import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from app.api.routes.chat import chat as chat_route
from app.api.routes.chat import agentic_query as agentic_query_route
from app.api.schemas.chat import ChatRequest, QueryRequest, SearchFilters
from app.api.schemas.chat import ChatResponse, QueryResponse


@pytest.mark.asyncio
async def test_chat_requires_user_id_missing_returns_400():
    chat_service = AsyncMock()
    logger = MagicMock()

    request = ChatRequest(message="hello")

    with pytest.raises(HTTPException) as exc:
        await chat_route(
            request=request,
            chat_service=chat_service,
            logger=logger,
            token=None,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "user id is required"


@pytest.mark.asyncio
async def test_chat_requires_user_id_blank_returns_400():
    chat_service = AsyncMock()
    logger = MagicMock()

    request = ChatRequest(user_id="   ", message="hello")

    with pytest.raises(HTTPException) as exc:
        await chat_route(
            request=request,
            chat_service=chat_service,
            logger=logger,
            token=None,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "user id is required"





@pytest.mark.asyncio
async def test_chat_with_user_id_calls_service():
    chat_service = AsyncMock()
    logger = MagicMock()

    expected = ChatResponse(message="ok")
    chat_service.chat_async.return_value = expected

    request = ChatRequest(user_id="user123", message="hello")

    result = await chat_route(
        request=request,
        chat_service=chat_service,
        logger=logger,
        token=None,
    )

    assert result.message == "ok"
    chat_service.chat_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_agentic_query_with_user_id_calls_service():
    chat_service = AsyncMock()
    logger = MagicMock()

    expected = QueryResponse(answer="ok")
    chat_service.query_async.return_value = expected

    request = QueryRequest(
        query="what is cosmos db?",
        session_id="550e8400-e29b-41d4-a716-446655440000",
        user_id="user123",
        filters=SearchFilters(opco_values=["TEKsystems"])
    )

    result = await agentic_query_route(
        request=request,
        chat_service=chat_service,
        logger=logger,
    )

    assert result.answer == "ok"
    chat_service.query_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_agentic_query_with_filters_calls_service():
    """Test that filters are passed through to the service."""
    chat_service = AsyncMock()
    logger = MagicMock()

    expected = QueryResponse(answer="ok")
    chat_service.query_async.return_value = expected

    filters = SearchFilters(
        opco_values=["TEKsystems", "Aerotek"],
        persona_values=["Front Office"]
    )

    request = QueryRequest(
        query="what is cosmos db?",
        session_id="550e8400-e29b-41d4-a716-446655440000",
        user_id="user123",
        filters=filters
    )

    result = await agentic_query_route(
        request=request,
        chat_service=chat_service,
        logger=logger,
    )

    assert result.answer == "ok"
    chat_service.query_async.assert_awaited_once()
    
    # Verify filters were passed to service
    call_kwargs = chat_service.query_async.call_args.kwargs
    assert call_kwargs["filters"] == filters


@pytest.mark.asyncio
async def test_get_conversation_history_success():
    """Test getting conversation history for a session."""
    from app.api.routes.chat import get_conversation_history
    
    chat_service = AsyncMock()
    logger = MagicMock()

    expected_conversation = {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "user_id": "user123",
        "message_count": 4,
        "messages": [
            {"role": "user", "text": "What is Cosmos DB?", "id": "msg-1"},
            {"role": "assistant", "text": "Cosmos DB is...", "id": "msg-2"},
            {"role": "user", "text": "Tell me more", "id": "msg-3"},
            {"role": "assistant", "text": "Sure...", "id": "msg-4"},
        ]
    }
    
    chat_service.get_conversation.return_value = expected_conversation

    result = await get_conversation_history(
        session_id="550e8400-e29b-41d4-a716-446655440000",
        user_id="user123",
        chat_service=chat_service,
        logger=logger,
        max_messages=None,
    )

    assert result["session_id"] == "550e8400-e29b-41d4-a716-446655440000"
    assert result["user_id"] == "user123"
    assert result["returned_messages"] == 4
    assert len(result["messages"]) == 4
    chat_service.get_conversation.assert_awaited_once_with("user123", "550e8400-e29b-41d4-a716-446655440000")


@pytest.mark.asyncio
async def test_get_conversation_history_with_max_messages():
    """Test getting conversation history with max_messages limit."""
    from app.api.routes.chat import get_conversation_history
    
    chat_service = AsyncMock()
    logger = MagicMock()

    expected_conversation = {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "user_id": "user123",
        "message_count": 10,
        "messages": [{"role": "user", "text": f"Message {i}", "id": f"msg-{i}"} for i in range(10)]
    }
    
    chat_service.get_conversation.return_value = expected_conversation

    result = await get_conversation_history(
        session_id="550e8400-e29b-41d4-a716-446655440000",
        user_id="user123",
        chat_service=chat_service,
        logger=logger,
        max_messages=5,
    )

    assert result["total_messages"] == 10
    assert result["returned_messages"] == 5
    # Should return last 5 messages
    assert len(result["messages"]) == 5
    assert result["messages"][0]["id"] == "msg-5"
    assert result["messages"][-1]["id"] == "msg-9"


@pytest.mark.asyncio
async def test_get_conversation_history_not_found():
    """Test getting conversation history when conversation doesn't exist."""
    from app.api.routes.chat import get_conversation_history
    
    chat_service = AsyncMock()
    logger = MagicMock()
    
    chat_service.get_conversation.return_value = None

    with pytest.raises(HTTPException) as exc:
        await get_conversation_history(
            session_id="550e8400-e29b-41d4-a716-446655440000",
            user_id="user123",
            chat_service=chat_service,
            logger=logger,
            max_messages=None,
        )

    assert exc.value.status_code == 404
    assert "not found" in exc.value.detail.lower()
