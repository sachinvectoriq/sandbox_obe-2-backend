"""Unit tests for skillset service."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from azure.search.documents.indexes.aio import SearchIndexerClient

from app.ingestion.skillset_service import SkillsetService
from app.models.config_options import (
    SearchServiceOptions,
    AzureOpenAIOptions,
    AIServicesOptions,
    BlobStorageOptions,
)


@pytest.fixture
def mock_indexer_client():
    """Create a mock async SearchIndexerClient."""
    client = AsyncMock(spec=SearchIndexerClient)
    client.create_or_update_skillset = AsyncMock()
    return client


@pytest.fixture
def search_options():
    """Create test search service options."""
    return SearchServiceOptions(
        endpoint="https://test.search.windows.net",
        api_key="test-key",
        index_name="test-index",
        indexer_name="test-indexer",
        data_source_name="test-datasource",
        skillset_name="test-skillset",
        skillset_api_version="2025-08-01-preview",
    )


@pytest.fixture
def openai_options():
    """Create test Azure OpenAI options."""
    return AzureOpenAIOptions(
        resource_uri="https://test.openai.azure.com",
        text_embedding_model="text-embedding-3-large",
        chat_completion_model="gpt-4o",
        chat_completion_resource_uri="https://test-chat.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview",
    )


@pytest.fixture
def ai_services_options():
    """Create test AI Services options."""
    return AIServicesOptions(
        cognitive_services_endpoint="https://test.cognitiveservices.azure.com"
    )


@pytest.fixture
def blob_options():
    """Create test blob storage options."""
    return BlobStorageOptions(
        resource_id="/subscriptions/test/resourceGroups/test/providers/Microsoft.Storage/storageAccounts/test",
        container_name="documents",
        images_container_name="images",
    )


@pytest.fixture
def skillset_service(
    mock_indexer_client,
    search_options,
    openai_options,
    ai_services_options,
    blob_options,
    mock_logger,
):
    """Create a SkillsetService instance for testing."""
    return SkillsetService(
        search_indexer_client=mock_indexer_client,
        search_options=search_options,
        openai_options=openai_options,
        ai_services_options=ai_services_options,
        blob_options=blob_options,
        logger=mock_logger,
    )


@pytest.mark.asyncio
async def test_create_skillset_using_sdk_async_success(
    skillset_service, mock_indexer_client
):
    """Test successful skillset creation using SDK."""
    skillset_name = "test-skillset"
    index_name = "test-index"

    await skillset_service.create_skillset_using_sdk_async(
        skillset_name, index_name
    )

    # Verify the client was called
    mock_indexer_client.create_or_update_skillset.assert_called_once()

    # Get the skillset that was passed
    call_args = mock_indexer_client.create_or_update_skillset.call_args
    skillset = call_args[0][0]

    # Verify skillset properties
    assert skillset.name == skillset_name
    assert skillset.description is not None
    assert len(skillset.skills) > 0
    assert skillset.cognitive_services_account is not None
    assert skillset.index_projection is not None


@pytest.mark.asyncio
async def test_create_skillset_using_sdk_async_text_only(
    skillset_service, mock_indexer_client
):
    """Test skillset creation with text-only mode (no image processing)."""
    skillset_name = "test-skillset"
    index_name = "test-index"

    await skillset_service.create_skillset_using_sdk_async(
        skillset_name, index_name
    )

    # Verify the client was called
    mock_indexer_client.create_or_update_skillset.assert_called_once()

    # Get the skillset that was passed
    call_args = mock_indexer_client.create_or_update_skillset.call_args
    skillset = call_args[0][0]

    # Verify skillset properties
    assert skillset.name == skillset_name
    # Skillset should have all skills including image processing
    assert len(skillset.skills) > 0
    # Knowledge store should always be configured for normalized images
    assert skillset.knowledge_store is not None


@pytest.mark.asyncio
async def test_create_skillset_using_rest_async_success(
    skillset_service, search_options, mocker
):
    """Test successful skillset creation using REST API."""
    skillset_name = "test-skillset"
    index_name = "test-index"

    # Mock httpx.AsyncClient
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"name": skillset_name}
    
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.put = AsyncMock(return_value=mock_response)
    
    mocker.patch("httpx.AsyncClient", return_value=mock_client)

    await skillset_service.create_skillset_using_rest_async(
        skillset_name, index_name
    )

    # Verify the HTTP client was called
    mock_client.put.assert_called_once()
    
    # Get the call arguments
    call_args = mock_client.put.call_args
    url = call_args[0][0]  # First positional argument is the URL
    kwargs = call_args[1]
    
    # Verify URL contains skillset name
    assert skillset_name in url
    assert url.endswith(f"/skillsets/{skillset_name}")
    
    # Verify API version is in query params
    params = kwargs.get("params", {})
    assert params.get("api-version") == search_options.skillset_api_version
    
    # Verify headers
    headers = kwargs["headers"]
    assert "api-key" in headers
    assert headers["api-key"] == search_options.api_key
    assert headers["Content-Type"] == "application/json"
    
    # Verify JSON payload exists
    json_payload = kwargs["json"]
    assert json_payload is not None
    assert json_payload["name"] == skillset_name
    assert json_payload["description"] is not None
    assert "skills" in json_payload
    assert len(json_payload["skills"]) > 0
    assert "indexProjections" in json_payload
    assert "knowledgeStore" in json_payload  # Should be present for multimodal mode


@pytest.mark.asyncio
async def test_create_skillset_using_rest_async_text_only(
    skillset_service, mocker
):
    """Test skillset creation via REST API with text-only mode."""
    skillset_name = "test-skillset"
    index_name = "test-index"

    # Mock httpx.AsyncClient
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"name": skillset_name}
    
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.put = AsyncMock(return_value=mock_response)
    
    mocker.patch("httpx.AsyncClient", return_value=mock_client)

    await skillset_service.create_skillset_using_rest_async(
        skillset_name, index_name
    )

    # Verify the HTTP client was called
    mock_client.put.assert_called_once()
    
    # Get the JSON payload
    call_args = mock_client.put.call_args
    json_payload = call_args[1]["json"]
    
    # Verify knowledge store is always configured for multimodal processing
    assert "knowledgeStore" in json_payload
    assert json_payload.get("knowledgeStore") is not None
    
    # Verify skills exist for multimodal processing
    assert "skills" in json_payload
    assert len(json_payload["skills"]) > 0


@pytest.mark.asyncio
async def test_create_skillset_using_rest_async_http_error(
    skillset_service, mocker
):
    """Test skillset creation via REST API handles HTTP errors."""
    skillset_name = "test-skillset"
    index_name = "test-index"

    # Mock httpx.AsyncClient with error response
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad request error"
    mock_response.raise_for_status.side_effect = Exception("HTTP 400 error")
    
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.put = AsyncMock(return_value=mock_response)
    
    mocker.patch("httpx.AsyncClient", return_value=mock_client)

    with pytest.raises(Exception):
        await skillset_service.create_skillset_using_rest_async(
            skillset_name, index_name
        )
