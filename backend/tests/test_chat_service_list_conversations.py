import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.chat_service import ChatService
from app.services.search_service import ISearchService
from app.models import AzureOpenAIOptions, WorkflowOptions


@pytest.mark.asyncio
async def test_list_conversations_returns_list_not_none():
    mock_search_service = AsyncMock(spec=ISearchService)
    mock_logger = MagicMock()
    openai_options = AzureOpenAIOptions(
        resource_uri="https://test.openai.azure.com",
        api_key="test-api-key",
        text_embedding_model="text-embedding-3-large",
        chat_completion_model="gpt-4o",
    )
    workflow_options = WorkflowOptions(
        max_retrieval_iterations=3,        
    )

    conversation_service = AsyncMock()
    conversation_service.list_user_sessions.return_value = [{"session_id": "s1"}]

    service = ChatService(
        search_service=mock_search_service,
        openai_options=openai_options,
        logger=mock_logger,
        workflow_options=workflow_options,
        conversation_service=conversation_service,
    )

    sessions = await service.list_conversations("u1", 100)

    assert sessions == [{"session_id": "s1"}]


@pytest.mark.asyncio
async def test_list_conversations_none_from_store_becomes_empty_list():
    mock_search_service = AsyncMock(spec=ISearchService)
    mock_logger = MagicMock()
    openai_options = AzureOpenAIOptions(
        resource_uri="https://test.openai.azure.com",
        api_key="test-api-key",
        text_embedding_model="text-embedding-3-large",
        chat_completion_model="gpt-4o",
    )
    workflow_options = WorkflowOptions(
        max_retrieval_iterations=3,
    )

    conversation_service = AsyncMock()
    conversation_service.list_user_sessions.return_value = None

    service = ChatService(
        search_service=mock_search_service,
        openai_options=openai_options,
        logger=mock_logger,
        workflow_options=workflow_options,
        conversation_service=conversation_service,
    )

    sessions = await service.list_conversations("u1", 100)

    assert sessions == []
