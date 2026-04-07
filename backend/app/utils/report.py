"""
Report API

This module provides endpoints to fetch and analyze
feedback data from Cosmos DB.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from azure.cosmos.aio import CosmosClient
from app.core.container import Container

router = APIRouter(prefix="/report", tags=["Report"])


# ---------------------------
# Dependency to Get Cosmos
# ---------------------------

def get_cosmos_client():
    container = Container()
    return container.cosmos_client()


# ---------------------------
# Get All Feedback (Paginated)
# ---------------------------

@router.get("/all")
async def get_all_feedback(
    limit: int = Query(100, ge=1),
    offset: int = Query(0, ge=0),
    cosmos_client: CosmosClient = Depends(get_cosmos_client),
):
    try:
        database_name = "feedback-table"
        container_name = "feedback-container"

        db = cosmos_client.get_database_client(database_name)
        container = db.get_container_client(container_name)

        query = f"""
        SELECT * FROM c
        WHERE c.record_type = 'feedback'
        OFFSET {offset} LIMIT {limit}
        """

        items = []
        async for item in container.query_items(query=query):
            items.append(item)

        return {
            "count": len(items),
            "limit": limit,
            "offset": offset,
            "data": items
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# Filter Feedback
# ---------------------------

@router.get("/filter")
async def filter_feedback(
    start_date: Optional[str] = None,   # YYYY-MM-DD
    end_date: Optional[str] = None,     # YYYY-MM-DD
    feedback_type: Optional[str] = None,
    persona: Optional[str] = None,
    opco: Optional[str] = None,
    user_name: Optional[str] = None,
    limit: int = Query(100, ge=1),
    offset: int = Query(0, ge=0),
    cosmos_client: CosmosClient = Depends(get_cosmos_client),
):
    try:
        database_name = "feedback-table"
        container_name = "feedback-container"

        db = cosmos_client.get_database_client(database_name)
        container = db.get_container_client(container_name)

        conditions = ["c.record_type = 'feedback'"]
        parameters = []

        if feedback_type:
            conditions.append("c.feedback_type = @feedback_type")
            parameters.append({"name": "@feedback_type", "value": feedback_type})

        if persona:
            conditions.append("c.persona = @persona")
            parameters.append({"name": "@persona", "value": persona})

        if opco:
            conditions.append("c.opco = @opco")
            parameters.append({"name": "@opco", "value": opco})

        if user_name:
            conditions.append("c.user_name = @user_name")
            parameters.append({"name": "@user_name", "value": user_name})

        if start_date:
            conditions.append("c.date >= @start_date")
            parameters.append({"name": "@start_date", "value": start_date})

        if end_date:
            conditions.append("c.date <= @end_date")
            parameters.append({"name": "@end_date", "value": end_date})

        where_clause = " AND ".join(conditions)

        query = f"""
        SELECT * FROM c
        WHERE {where_clause}
        OFFSET {offset} LIMIT {limit}
        """

        items = []
        async for item in container.query_items(
            query=query,
            parameters=parameters
        ):
            items.append(item)

        return {
            "count": len(items),
            "limit": limit,
            "offset": offset,
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
                "feedback_type": feedback_type,
                "persona": persona,
                "opco": opco,
                "user_name": user_name
            },
            "data": items
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# Feedback Summary (Grouped)
# ---------------------------

@router.get("/summary")
async def feedback_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    feedback_type: Optional[str] = None,
    user_name: Optional[str] = None,
    cosmos_client: CosmosClient = Depends(get_cosmos_client),
):
    try:
        database_name = "feedback-table"
        container_name = "feedback-container"

        db = cosmos_client.get_database_client(database_name)
        container = db.get_container_client(container_name)

        conditions = ["c.record_type = 'feedback'"]
        parameters = []

        if feedback_type:
            conditions.append("c.feedback_type = @feedback_type")
            parameters.append({"name": "@feedback_type", "value": feedback_type})

        if user_name:
            conditions.append("c.user_name = @user_name")
            parameters.append({"name": "@user_name", "value": user_name})

        if start_date:
            conditions.append("c.date >= @start_date")
            parameters.append({"name": "@start_date", "value": start_date})

        if end_date:
            conditions.append("c.date <= @end_date")
            parameters.append({"name": "@end_date", "value": end_date})

        where_clause = " AND ".join(conditions)

        query = f"""
        SELECT c.feedback_type, COUNT(1) as count
        FROM c
        WHERE {where_clause}
        GROUP BY c.feedback_type
        """

        results = []
        async for item in container.query_items(
            query=query,
            parameters=parameters
        ):
            results.append(item)

        return {
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
                "feedback_type": feedback_type,
                "user_name": user_name
            },
            "summary": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
