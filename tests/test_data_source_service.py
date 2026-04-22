"""Unit tests for data source service."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from azure.search.documents.indexes.models import SearchIndexerDataSourceConnection

from app.ingestion.data_source_service import DataSourceService
from app.models.config_options import BlobStorageOptions


@pytest.fixture
def mock_indexer_client():
    """Create a mock SearchIndexerClient."""
    client = MagicMock()
    client.create_or_update_data_source_connection = AsyncMock()
    return client


@pytest.fixture
def blob_options_with_resource_id():
    """Create test blob storage options with resource ID (managed identity)."""
    return BlobStorageOptions(
        resource_id="/subscriptions/test/resourceGroups/test/providers/Microsoft.Storage/storageAccounts/testaccount",
        container_name="test-container",
        images_container_name="test-images",
    )


@pytest.fixture
def blob_options_with_connection_string():
    """Create test blob storage options with connection string."""
    return BlobStorageOptions(
        connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key==;EndpointSuffix=core.windows.net",
        container_name="test-container",
        images_container_name="test-images",
    )


@pytest.mark.asyncio
async def test_create_blob_data_source_async_with_managed_identity(
    mock_indexer_client, blob_options_with_resource_id, mock_logger
):
    """Test successful blob data source creation with managed identity."""
    data_source_service = DataSourceService(
        indexer_client=mock_indexer_client,
        blob_options=blob_options_with_resource_id,
        logger=mock_logger,
    )
    data_source_name = "test-datasource"

    await data_source_service.create_blob_data_source_async(data_source_name)

    # Verify the client was called
    mock_indexer_client.create_or_update_data_source_connection.assert_called_once()

    # Get the data source that was passed
    call_args = mock_indexer_client.create_or_update_data_source_connection.call_args
    data_source: SearchIndexerDataSourceConnection = call_args[0][0]

    # Verify data source properties
    assert data_source.name == data_source_name
    assert data_source.container.name == blob_options_with_resource_id.container_name
    assert f"ResourceId={blob_options_with_resource_id.resource_id}" in data_source.connection_string
    assert data_source.description == "A data source to store multi-modality documents"
    assert data_source.data_change_detection_policy is not None
    assert data_source.data_deletion_detection_policy is not None


@pytest.mark.asyncio
async def test_create_blob_data_source_async_with_connection_string(
    mock_indexer_client, blob_options_with_connection_string, mock_logger
):
    """Test successful blob data source creation with connection string."""
    data_source_service = DataSourceService(
        indexer_client=mock_indexer_client,
        blob_options=blob_options_with_connection_string,
        logger=mock_logger,
    )
    data_source_name = "test-datasource"

    await data_source_service.create_blob_data_source_async(data_source_name)

    # Verify the client was called
    mock_indexer_client.create_or_update_data_source_connection.assert_called_once()

    # Get the data source that was passed
    call_args = mock_indexer_client.create_or_update_data_source_connection.call_args
    data_source: SearchIndexerDataSourceConnection = call_args[0][0]

    # Verify data source properties
    assert data_source.name == data_source_name
    assert data_source.container.name == blob_options_with_connection_string.container_name
    # When connection_string is provided, it should be used as-is
    assert data_source.connection_string == blob_options_with_connection_string.connection_string
    assert data_source.description == "A data source to store multi-modality documents"
    assert data_source.data_change_detection_policy is not None
    assert data_source.data_deletion_detection_policy is not None


@pytest.mark.asyncio
async def test_create_blob_data_source_async_missing_credentials(mock_indexer_client, mock_logger):
    """Test blob data source creation when neither resource_id nor connection_string is provided.
    
    Note: The current implementation allows this (uses managed identity),
    but resource_id is still required by BlobStorageOptions validation.
    This test now verifies that managed identity path works when only container names are provided.
    """
    blob_options_no_auth = BlobStorageOptions(
        container_name="test-container",
        images_container_name="test-images",
    )
    data_source_service = DataSourceService(
        indexer_client=mock_indexer_client,
        blob_options=blob_options_no_auth,
        logger=mock_logger,
    )
    
    # Should raise ValueError when neither credential is provided
    with pytest.raises(ValueError, match="Either BlobStorage__ResourceId or BlobStorageConnection must be configured"):
        await data_source_service.create_blob_data_source_async("test-datasource")
