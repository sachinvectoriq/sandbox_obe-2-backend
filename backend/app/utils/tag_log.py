"""
Tag Logs API

This module provides a standalone tag logging endpoint.
Frontend/backend can call this API directly to store tag logs in Cosmos DB.
"""

import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from azure.cosmos.aio import CosmosClient

from app.core.container import Container


router = APIRouter(prefix="/tag-logs", tags=["Tag Logs"])


# ---------------------------
# Request Model
# ---------------------------

class TagLogRequest(BaseModel):
    doc_id: str
    user_id: str | None = None
    event_type: str | None = None
    source: str | None = None
    metadata: dict | None = None

    # frontend passes only raw intent fields
    container_name: str | None = None
    document_name: str | None = None

    opcos: list[str] | None = None
    personas: list[str] | None = None

    presence_type: str | None = None

    # must be: opco | persona | both
    absent: str | None = None

    @field_validator("absent")
    @classmethod
    def validate_absent(cls, v):
        if v is None:
            return v
        v = v.lower().strip()
        allowed = {"opco", "persona", "both"}
        if v not in allowed:
            raise ValueError("absent must be one of: opco, persona, both")
        return v


# ---------------------------
# Dependency
# ---------------------------

def get_cosmos_client():
    container = Container()
    return container.cosmos_client()


# ---------------------------
# API Endpoint
# ---------------------------

@router.post("/submit")
async def submit_tag_log(
    request: TagLogRequest,
    cosmos_client: CosmosClient = Depends(get_cosmos_client),
):
    try:
        database_name = "tag_logs"
        container_name = "tag_logs_container"

        db = cosmos_client.get_database_client(database_name)
        container = db.get_container_client(container_name)

        # ---------------------------
        # TIME COMPUTATION (SERVER SIDE)
        # ---------------------------
        utc_now = datetime.now(timezone.utc)

        ist_now = utc_now.astimezone(ZoneInfo("Asia/Kolkata"))
        eastern_now = utc_now.astimezone(ZoneInfo("America/New_York"))

        document = {
            "id": str(uuid.uuid4()),
            "doc_id": request.doc_id,  # partition key
            "user_id": request.user_id,
            "event_type": request.event_type,
            "source": request.source,

            "metadata": request.metadata or {},

            # contextual fields
            "container_name": request.container_name,
            "document_name": request.document_name,

            # tagging info
            "opcos": request.opcos,
            "personas": request.personas,
            "presence_type": request.presence_type,
            "absent": request.absent,

            # computed times (NO FRONTEND INPUT)
            "utc_time": utc_now.isoformat(),
            "ist_time": ist_now.isoformat(),
            "eastern_time": eastern_now.isoformat(),

            # system fields
            "timestamp_utc": utc_now.isoformat(),
            "date": utc_now.date().isoformat(),
            "record_type": "tag_log"
        }

        # await container.create_item(document)  # COMMENTED OUT: logging disabled

        return {
            "message": "Tag log submitted successfully (logging disabled)",
            "id": document["id"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
