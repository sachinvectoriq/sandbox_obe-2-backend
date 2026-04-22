"""Unit tests for indexer service."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes.aio import SearchIndexerClient
from azure.search.documents.indexes.models import SearchIndexer, SearchIndexerStatus

from app.ingestion.indexer_service import IndexerService


@pytest.fixture
def mock_indexer_client():
    """Create a mock async SearchIndexerClient."""
    client = AsyncMock(spec=SearchIndexerClient)
    client.create_or_update_indexer = AsyncMock()
    client.run_indexer = AsyncMock()
    client.get_indexer_status = AsyncMock()
    client.reset_indexer = AsyncMock()
    return client


@pytest.fixture
def indexer_service(mock_indexer_client, mock_logger):
    """Create an IndexerService instance for testing."""
    return IndexerService(indexer_client=mock_indexer_client, logger=mock_logger)


@pytest.mark.asyncio
async def test_create_indexer_async_success(indexer_service, mock_indexer_client):
    """Test successful indexer creation."""
    indexer_name = "test-indexer"
    data_source_name = "test-datasource"
    target_index_name = "test-index"
    skillset_name = "test-skillset"

    await indexer_service.create_indexer_async(
        indexer_name, data_source_name, target_index_name, skillset_name
    )

    # Verify the client was called
    mock_indexer_client.create_or_update_indexer.assert_called_once()

    # Get the indexer that was passed
    call_args = mock_indexer_client.create_or_update_indexer.call_args
    indexer: SearchIndexer = call_args[0][0]

    # Verify indexer properties
    assert indexer.name == indexer_name
    assert indexer.data_source_name == data_source_name
    assert indexer.target_index_name == target_index_name
    assert indexer.skillset_name == skillset_name
    assert indexer.parameters is not None
    assert len(indexer.field_mappings) > 0


@pytest.mark.asyncio
async def test_create_indexer_async_field_mappings(
    indexer_service, mock_indexer_client
):
    """Test that indexer field mappings are configured correctly."""
    indexer_name = "test-indexer"

    await indexer_service.create_indexer_async(
        indexer_name, "datasource", "index", "skillset"
    )

    call_args = mock_indexer_client.create_or_update_indexer.call_args
    indexer: SearchIndexer = call_args[0][0]

    # Verify field mapping exists
    assert len(indexer.field_mappings) == 1
    assert indexer.field_mappings[0].source_field_name == "metadata_storage_name"
    assert indexer.field_mappings[0].target_field_name == "document_title"


@pytest.mark.asyncio
async def test_run_indexer_async_success(indexer_service, mock_indexer_client):
    """Test successful indexer run."""
    indexer_name = "test-indexer"

    await indexer_service.run_indexer_async(indexer_name)

    # Verify the client was called
    mock_indexer_client.run_indexer.assert_called_once_with(indexer_name)


@pytest.mark.asyncio
async def test_get_indexer_status_async_success(indexer_service, mock_indexer_client):
    """Test successful indexer status retrieval."""
    indexer_name = "test-indexer"
    mock_status = MagicMock(spec=SearchIndexerStatus)

    mock_indexer_client.get_indexer_status.return_value = mock_status

    status = await indexer_service.get_indexer_status_async(indexer_name)

    # Verify the client was called and status returned
    mock_indexer_client.get_indexer_status.assert_called_once_with(indexer_name)
    assert status == mock_status


@pytest.mark.asyncio
async def test_reset_indexer_async_success(indexer_service, mock_indexer_client):
    """Test successful indexer reset."""
    indexer_name = "test-indexer"

    await indexer_service.reset_indexer_async(indexer_name)

    # Verify the client was called
    mock_indexer_client.reset_indexer.assert_called_once_with(indexer_name)
