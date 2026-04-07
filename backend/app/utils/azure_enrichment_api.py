
from fastapi import APIRouter
from typing import Dict, List, Any
from azure.storage.blob import BlobServiceClient
import os
import re

router = APIRouter()

# ================================
# CONFIG (use env or replace)
# ================================
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("BLOBSTORAGE_CONNECTION_STRING", "")

blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)


# ================================
# HELPERS
# ================================
def parse_values(raw: str) -> List[str]:
    """
    Converts:
    'fdg|tsg' OR 'fdg, tsg' → ['fdg', 'tsg']
    """
    if not raw:
        return []

    # normalize separators
    cleaned = re.split(r"[|,;]", raw)

    return [x.strip() for x in cleaned if x and x.strip()]


def extract_blob_metadata(container: str, blob_name: str) -> Dict[str, Any]:
    """
    Reads blob metadata from Azure Storage
    """
    container_client = blob_service_client.get_container_client(container)
    blob_client = container_client.get_blob_client(blob_name)

    props = blob_client.get_blob_properties()

    metadata = props.metadata or {}

    return metadata


# ================================
# MAIN WEB API SKILL ENDPOINT
# ================================
@router.post("/enrich")
async def enrich(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Azure WebApiSkill entry point
    """

    results = []

    for item in request.get("values", []):
        record_id = item.get("recordId")
        data = item.get("data", {})

        blob_name = data.get("metadata_storage_name")
        container = data.get("container") or data.get("metadata_storage_container")

        opco_values = []
        persona_values = []

        try:
            # 1. FETCH BLOB METADATA
            metadata = extract_blob_metadata(container, blob_name)

            # 2. TRY OPERATIONAL KEYS (priority order)
            raw_opco = (
                metadata.get("opco")
                or metadata.get("opco_values")
                or data.get("opco")
                or ""
            )

            raw_persona = (
                metadata.get("persona")
                or metadata.get("persona_values")
                or data.get("persona")
                or ""
            )

            # 3. PARSE INTO ARRAYS
            opco_values = parse_values(raw_opco)
            persona_values = parse_values(raw_persona)

        except Exception as e:
            # NEVER FAIL INDEXING
            opco_values = []
            persona_values = []

        results.append({
            "recordId": record_id,
            "data": {
                "opco_values": opco_values,
                "persona_values": persona_values
            }
        })

    return {"values": results}
