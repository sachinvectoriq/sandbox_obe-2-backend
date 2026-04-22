"""Unit tests for search index service."""

import pytest
from unittest.mock import AsyncMock
from azure.search.documents.indexes.aio import SearchIndexClient

from app.ingestion.search_index_service import SearchIndexService
from app.models.config_options import AzureOpenAIOptions


@pytest.fixture
def mock_index_client():
    """Create a mock async SearchIndexClient."""
    client = AsyncMock(spec=SearchIndexClient)
    client.create_or_update_index = AsyncMock()
    return client


@pytest.fixture
def openai_options():
    """Create test Azure OpenAI options."""
    return AzureOpenAIOptions(
        resource_uri="https://test.openai.azure.com",
        api_key="test-key",
        text_embedding_model="text-embedding-3-large",
        chat_completion_model="gpt-4o",
        chat_completion_resource_uri="https://test-chat.openai.azure.com",
        chat_completion_api_key="test-chat-key",
    )


@pytest.fixture
def search_index_service(mock_index_client, openai_options, mock_logger):
    """Create a SearchIndexService instance for testing."""
    return SearchIndexService(
        index_client=mock_index_client,
        openai_options=openai_options,
        logger=mock_logger,
    )


@pytest.mark.asyncio
async def test_create_search_index_async_success(
    search_index_service, mock_index_client
):
    """Test successful search index creation."""
    index_name = "test-index"

    await search_index_service.create_search_index_async(index_name)

    # Verify the client was called
    mock_index_client.create_or_update_index.assert_called_once()

    # Get the index that was passed
    call_args = mock_index_client.create_or_update_index.call_args
    index = call_args[0][0]

    # Verify index properties
    assert index.name == index_name
    assert len(index.fields) > 0
    assert index.vector_search is not None
    assert index.semantic_search is not None


@pytest.mark.asyncio
async def test_create_search_index_fields_configuration(
    search_index_service, mock_index_client
):
    """Test that index fields are configured correctly."""
    index_name = "test-index"

    await search_index_service.create_search_index_async(index_name)

    call_args = mock_index_client.create_or_update_index.call_args
    index = call_args[0][0]

    # Verify key field exists
    content_id_field = next(f for f in index.fields if f.name == "content_id")
    assert content_id_field.key is True

    # Verify vector field exists
    embedding_field = next(f for f in index.fields if f.name == "content_embedding")
    assert embedding_field.searchable is True
    assert embedding_field.vector_search_dimensions == 3072

    # Verify ComplexField exists for location_metadata
    location_field = next(f for f in index.fields if f.name == "location_metadata")
    assert location_field is not None


@pytest.mark.asyncio
async def test_create_search_index_vector_search_config(
    search_index_service, mock_index_client, openai_options
):
    """Test that vector search is configured correctly."""
    index_name = "test-index"

    await search_index_service.create_search_index_async(index_name)

    call_args = mock_index_client.create_or_update_index.call_args
    index = call_args[0][0]

    # Verify vector search configuration
    assert len(index.vector_search.algorithms) > 0
    assert len(index.vector_search.vectorizers) > 0
    assert len(index.vector_search.profiles) > 0
    assert len(index.vector_search.compressions) > 0


@pytest.mark.asyncio
async def test_create_search_index_semantic_search_config(
    search_index_service, mock_index_client
):
    """Test that semantic search is configured correctly."""
    index_name = "test-index"

    await search_index_service.create_search_index_async(index_name)

    call_args = mock_index_client.create_or_update_index.call_args
    index = call_args[0][0]

    # Verify semantic search configuration
    assert index.semantic_search is not None
    assert index.semantic_search.default_configuration_name == "semanticconfig"
    assert len(index.semantic_search.configurations) > 0
