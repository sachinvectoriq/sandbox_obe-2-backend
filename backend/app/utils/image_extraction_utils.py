import requests
from fastapi import APIRouter
from typing import Dict, List, Any


# --- FastAPI router (for main.py) ---
router = APIRouter()


# --- Azure WebApiSkill batch models ---
class SkillRecordData:
    def __init__(self, systemMessage: str, userMessage: str, image: str):
        self.systemMessage = systemMessage
        self.userMessage = userMessage
        self.image = image


class SkillRecord:
    def __init__(self, recordId: str, data: SkillRecordData):
        self.recordId = recordId
        self.data = data


class SkillRequest:
    def __init__(self, values: List[SkillRecord]):
        self.values = values


# --- Your Azure OpenAI config ---
API_KEY = "1ZomQzKUgTD5SUYNQGfta1xc7ssr8ycArmc7TgKN9lgas0xEqETUJQQJ99CBACHYHv6XJ3w3AAAAACOGy3my"
ENDPOINT_URL = "https://aif-ocm20-obe-dev-001.openai.azure.com/openai/deployments/gpt-5.1-ocm20-obe-dev-001/chat/completions?api-version=2025-04-01-preview"


# --- FIX: Ensure image is in proper data URL format ---
def ensure_data_url(image_data: str) -> str:
    """
    Azure OpenAI expects image in data URL format:
    data:image/png;base64,<base64>
    """
    if not image_data:
        return image_data

    # Already correct
    if image_data.startswith("data:image"):
        return image_data

    # Default to PNG if not specified
    return f"data:image/png;base64,{image_data}"


# --- LLM call to Azure OpenAI GPT-vision ---
def call_openai_vision(
    system_message: str,
    user_message: str,
    image_data: str,
) -> str:
    """
    Call Azure OpenAI GPT-vision with the same semantics as the ChatCompletionSkill.
    Returns a string: the verbalized image description.
    """
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

    # ✅ FIX APPLIED HERE
    image_data = ensure_data_url(image_data)

    body = {
        "messages": [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data
                        },
                    },
                ],
            },
        ],
        "temperature": 0.7,
    }

    resp = requests.post(ENDPOINT_URL, headers=headers, json=body)

    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenAI call failed: {resp.status_code} {resp.text}"
        )

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
        return content
    except (KeyError, IndexError):
        raise RuntimeError("Invalid OpenAI response", data)


# --- WebApiSkill endpoint ---
@router.post("/image-verbalization")
async def image_verbalization_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input from Azure WebApiSkill:
    {
        "values": [
            {
                "recordId": "1",
                "data": {
                    "systemMessage": "...",
                    "userMessage": "...",
                    "image": "base64 or data URL"
                }
            }
        ]
    }

    Output:
    {
        "values": [
            {
                "recordId": "1",
                "data": {
                    "response": "text description"
                }
            }
        ]
    }
    """
    results = []

    for item in payload.get("values", []):
        data = item["data"]
        system_msg = data["systemMessage"]
        user_msg = data["userMessage"]
        image_data = data["image"]

        try:
            verbalized = call_openai_vision(system_msg, user_msg, image_data)
            response_field = verbalized
        except Exception as e:
            response_field = f"[ERROR: {e}]"

        results.append({
            "recordId": item["recordId"],
            "data": {
                "response": response_field
            }
        })

    return {"values": results}