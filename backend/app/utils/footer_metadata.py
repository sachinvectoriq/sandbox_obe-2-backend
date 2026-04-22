from azure.storage.blob import ContainerClient
# =========================
# FIND BLOB PATH BY FILENAME (NESTED FOLDERS)
# =========================
def find_blob_by_filename(connection_string: str, container_name: str, filename: str) -> str:
    """
    Scan all blobs in the container (including nested folders) and return the full blob path for the given filename.
    Returns None if not found.
    """
    container = ContainerClient.from_connection_string(connection_string, container_name)
    for blob in container.list_blobs():
        actual_name = blob.name.split("/")[-1]
        if actual_name == filename:
            return blob.name  # full path (including folders)
    return None
from fastapi import APIRouter
from typing import Dict, List, Any
from azure.storage.blob import BlobClient
import requests
import time
import re
import asyncio   # ✅ ADDED ONLY THIS

router = APIRouter()

# =========================
# CONFIG
# =========================
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=saocm20obedev001;AccountKey=tPuIekoPR/H5VK++7dtJ4MnIkmb2M46j47vEA8ooT91ixadSuE5V1PSpfn2zH0kByBBglkYXh8+++ASt3pnjwA==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = ""  # dynamic from skillset

DOC_INTELLIGENCE_ENDPOINT = "https://ais-ocm20-obe-dev-001.cognitiveservices.azure.com/"
DOC_INTELLIGENCE_KEY = "3D5frJJMSDniR21q5dtm7V5mtu0Zlwhp3pIWw1vqp5b2Y8j6CPYGJQQJ99CBACHYHv6XJ3w3AAAEACOGKBNk"


# =========================
# STEP 1: DOWNLOAD FILE
# =========================
def download_blob(file_name: str, container_name: str) -> bytes:
    # Use the new function to find the full blob path (including nested folders)
    blob_path = find_blob_by_filename(AZURE_STORAGE_CONNECTION_STRING, container_name, file_name)
    if not blob_path:
        raise FileNotFoundError(f"File '{file_name}' not found in container '{container_name}' (including nested folders)")
    blob = BlobClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING,
        container_name=container_name,
        blob_name=blob_path
    )
    return blob.download_blob().readall()


# =========================
# STEP 2: OCR (FIRST 4 PAGES)
# =========================
def run_ocr(file_bytes: str) -> str:
    url = f"{DOC_INTELLIGENCE_ENDPOINT}/formrecognizer/documentModels/prebuilt-layout:analyze?api-version=2023-07-31&pages=1-4"

    headers = {
        "Ocp-Apim-Subscription-Key": DOC_INTELLIGENCE_KEY,
        "Content-Type": "application/pdf"
    }

    response = requests.post(url, headers=headers, data=file_bytes)

    if response.status_code != 202:
        raise Exception(f"OCR submit failed: {response.text}")

    operation_url = response.headers["operation-location"]

    while True:
        result = requests.get(operation_url, headers={
            "Ocp-Apim-Subscription-Key": DOC_INTELLIGENCE_KEY
        })

        result_json = result.json()

        if result_json["status"] == "succeeded":
            break
        elif result_json["status"] == "failed":
            raise Exception("OCR failed")

        time.sleep(2)

    text = ""
    for page in result_json["analyzeResult"]["pages"]:
        for line in page["lines"]:
            text += line["content"] + "\n"

    return text


# =========================
# STEP 3: EXTRACTION (STRICT + FIXED)
# =========================
def extract_footer_values(text: str, label: str) -> List[str]:
    OPCOS = [
        "Actalent",
        "Actalent Services",
        "Aerotek",
        "Aerotek Services",
        "Aston Carter",
        "TEKsystems",
        "TEKsystems Global Services",
        "Allegis Corporate Services"
    ]

    PERSONAS = [
        "FSG",
        "CLS",
        "Sales and Recruiting",
        "Delivery and TA Services",
        "Front Office",
        "Back Office",
        "Corporate Services",
        "Talent"
    ]

    pattern = rf"{label}\s*:\s*(.*)"

    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return []

    raw = match.group(1).split("\n")[0].strip().lower()

    search_list = OPCOS if label == "Operating Companies" else PERSONAS

    tokens = re.split(r"[,\|]", raw)
    tokens = [t.strip().lower() for t in tokens if t.strip()]

    found = []
    for token in tokens:
        for item in search_list:
            if token == item.lower():
                if item not in found:
                    found.append(item)

    return found


# =========================
# POST PROCESSING
# =========================
def normalize_opco(v: str) -> str:
    return re.sub(r"[^a-z0-9]", "", v.lower())


def normalize_persona(v: str) -> str:
    return re.sub(r"[^a-z0-9]", "", v.lower())


# =========================
# TAG API CALL (ONLY ADDITION)
# =========================
def send_to_tag_api(payload: Dict[str, Any]):
    url = "https://app-ocm20-obe-dev-001-dgd8hbbyeyc2d2fv.eastus2-01.azurewebsites.net/api/tag-logs/submit"
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print("⚠️ Tag API failed:", e)


# =========================
# CORE LOGIC (FULL TEXT)
# =========================
def process_document(file_name: str, container_name: str) -> Dict[str, Any]:
    print(f"📥 Downloading: {file_name} from {container_name}")
    file_bytes = download_blob(file_name, container_name)

    print("🔍 Running OCR...")
    text = run_ocr(file_bytes)

    print("🧠 Extracting OPCO + Persona from FULL TEXT...")

    opcos = extract_footer_values(text, "Operating Companies")
    personas = (
        extract_footer_values(text, "Persona Categories") or
        extract_footer_values(text, "Personas")
    )

    # =========================
    # CLEAN NULL VALUES
    # =========================
    def clean_list(values: List[str]) -> List[str]:
        cleaned = []
        for v in values:
            if not v:
                continue
            if str(v).strip().lower() in ["null", "none", ""]:
                continue
            cleaned.append(v)
        return cleaned

    opcos = clean_list(opcos)
    personas = clean_list(personas)

    # =========================
    # FINAL NORMALIZED OUTPUT
    # =========================
    opcos = opcos if opcos else None
    personas = personas if personas else None

    result = {
        "opco_values_array": [normalize_opco(x) for x in opcos] if opcos else None,
        "persona_values_array": [normalize_persona(x) for x in personas] if personas else None,
        "isValid": bool(opcos and personas),
        "executed": True
    }

    # =========================
    # ABSENT LOGIC (STRICT ENUM)
    # =========================
    if opcos is None and personas is None:
        absent_value = "both"
    elif opcos is None:
        absent_value = "opco"
    elif personas is None:
        absent_value = "persona"
    else:
        absent_value = None

    # =========================
    # TAG API PAYLOAD
    # =========================
    try:
        payload = {
            "doc_id": file_name,
            "event_type": "footer_extraction",
            "source": "ocr_pipeline",
            "container_name": container_name,
            "document_name": file_name,
            "opcos": opcos,
            "personas": personas,
            "presence_type": "extracted",
            "absent": absent_value,
            "metadata": {
                "opcos": result["opco_values_array"],
                "personas": result["persona_values_array"],
                "isValid": result["isValid"],
                "container": container_name,
                "document": file_name
            }
        }

        send_to_tag_api(payload)

    except Exception as e:
        print("⚠️ Failed to send tag log:", e)

    return result


# =========================
# FASTAPI ENDPOINT
# =========================
@router.post("/footer-metadata")
async def footer_metadata_endpoint(payload: dict):
    results = []

    for item in payload.get("values", []):
        data_input = item.get("data", {})

        file_name = data_input.get("metadata_storage_name", "")
        container_name = data_input.get("metadata_storage_container", "")

        print("📄 FILE:", file_name)
        print("📦 CONTAINER:", container_name)

        try:
            data = process_document(file_name, container_name)
        except Exception as e:
            data = {
                "error": str(e),
                "isValid": False,
                "executed": False
            }

        results.append({
            "recordId": item["recordId"],
            "data": data
        })

    return {"values": results}
