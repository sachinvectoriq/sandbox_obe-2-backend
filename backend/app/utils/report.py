from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from azure.cosmos.aio import CosmosClient
from app.core.container import Container
from datetime import datetime

router = APIRouter(prefix="/audit-report", tags=["Audit Report"])


# -----------------------------------
# Dependency
# -----------------------------------

def get_cosmos_client():
    container = Container()
    return container.cosmos_client()


# -----------------------------------
# Excluded Users
# -----------------------------------

EXCLUDED_USERS = [
    "Bhaskar, Solomon",
    "Sachin Bhusanurmath",
    "HardCodedUser",
    "Anonymous",
    "Test User"
]


# -----------------------------------
# Combined Audit + Feedback Report
# -----------------------------------

@router.get("/combined-report")
async def combined_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_name: Optional[str] = None,
    persona: Optional[str] = None,
    opco: Optional[str] = None,
    limit: int = Query(100, ge=1),
    offset: int = Query(0, ge=0),
    cosmos_client: CosmosClient = Depends(get_cosmos_client),
):

    try:

        # -----------------------------------
        # Audit Container
        # -----------------------------------

        audit_db = cosmos_client.get_database_client("audit-table")
        audit_container = audit_db.get_container_client("audit-container")

        # -----------------------------------
        # Feedback Container
        # -----------------------------------

        feedback_db = cosmos_client.get_database_client("feedback-table")
        feedback_container = feedback_db.get_container_client("feedback-container")

        # -----------------------------------
        # Build Audit Query
        # -----------------------------------

        conditions = []
        parameters = []

        excluded_users_query = ",".join(
            [f"'{user}'" for user in EXCLUDED_USERS]
        )

        conditions.append(
            f"c.user_name NOT IN ({excluded_users_query})"
        )

        if user_name:
            conditions.append("c.user_name = @user_name")
            parameters.append({
                "name": "@user_name",
                "value": user_name
            })

        if persona:
            conditions.append("c.persona = @persona")
            parameters.append({
                "name": "@persona",
                "value": persona
            })

        if opco:
            conditions.append("c.opco = @opco")
            parameters.append({
                "name": "@opco",
                "value": opco
            })

        if start_date:
            conditions.append("c.date >= @start_date")
            parameters.append({
                "name": "@start_date",
                "value": start_date
            })

        if end_date:
            conditions.append("c.date <= @end_date")
            parameters.append({
                "name": "@end_date",
                "value": end_date
            })

        where_clause = " AND ".join(conditions)

        audit_query = f"""
        SELECT
            c.id,
            c.chat_session_id,
            c.user_id,
            c.user_name,
            c.job_title,
            c.opco,
            c.persona,
            c.timestamp_utc,
            c.date,
            c.query,
            c.ai_response,
            c.citations
        FROM c
        WHERE {where_clause}
        ORDER BY c.timestamp_utc DESC
        OFFSET {offset} LIMIT {limit}
        """

        # -----------------------------------
        # Fetch Audit Data
        # -----------------------------------

        audit_items = []

        async for item in audit_container.query_items(
            query=audit_query,
            parameters=parameters,
            enable_cross_partition_query=True
        ):
            audit_items.append(item)

        # -----------------------------------
        # Fetch Feedback Data
        # -----------------------------------

        feedback_query = """
        SELECT
            c.chat_session_id,
            c.query,
            c.ai_response,
            c.feedback_type,
            c.feedback_note
        FROM c
        WHERE c.record_type = 'feedback'
        """

        feedback_items = []

        async for item in feedback_container.query_items(
            query=feedback_query,
            enable_cross_partition_query=True
        ):
            feedback_items.append(item)

        # -----------------------------------
        # Create Feedback Lookup Dictionary
        # -----------------------------------

        feedback_lookup = {}

        for fb in feedback_items:

            key = (
                fb.get("chat_session_id"),
                fb.get("query"),
                fb.get("ai_response")
            )

            feedback_lookup[key] = {
                "feedback_type": fb.get("feedback_type", "-"),
                "feedback_note": fb.get("feedback_note", "-")
            }

        # -----------------------------------
        # Merge Audit + Feedback
        # -----------------------------------

        final_results = []

        for audit in audit_items:

            key = (
                audit.get("chat_session_id"),
                audit.get("query"),
                audit.get("ai_response")
            )

            feedback_data = feedback_lookup.get(
                key,
                {
                    "feedback_type": "-",
                    "feedback_note": "-"
                }
            )

            # -----------------------------------
            # Format Timestamp
            # -----------------------------------

            formatted_timestamp = audit.get("timestamp_utc")

            try:
                if formatted_timestamp:
                    parsed_time = datetime.fromisoformat(
                        formatted_timestamp.replace("Z", "+00:00")
                    )

                    formatted_timestamp = parsed_time.strftime(
                        "%Y-%m-%d %H:%M:%S UTC"
                    )

            except Exception:
                pass

            # -----------------------------------
            # Final Combined Row
            # -----------------------------------

            combined_row = {
                "user_name": audit.get("user_name"),
                "job_title": audit.get("job_title"),
                "opco": audit.get("opco"),
                "persona": audit.get("persona"),
                "query": audit.get("query"),
                "ai_response": audit.get("ai_response"),
                "citations": audit.get("citations"),
                "date_and_time": formatted_timestamp,
                "feedback_type": feedback_data["feedback_type"],
                "feedback_note": feedback_data["feedback_note"]
            }

            final_results.append(combined_row)

        return {
            "count": len(final_results),
            "limit": limit,
            "offset": offset,
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
                "user_name": user_name,
                "persona": persona,
                "opco": opco
            },
            "data": final_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
