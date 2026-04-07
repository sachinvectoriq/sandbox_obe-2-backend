"""Unit tests for search pipeline orchestrator."""

import pytest
from unittest.mock import AsyncMock
from azure.core.exceptions import ResourceNotFoundError

from app.ingestion.search_pipeline_orchestrator import SearchPipelineOrchestrator
from app.ingestion.data_source_service import IDataSourceService
from app.ingestion.search_index_service import ISearchIndexService
from app.ingestion.skillset_service import ISkillsetService
from app.ingestion.indexer_service import IIndexerService
from app.models.config_options import SearchServiceOptions


@pytest.fixture
def mock_data_source_service():
    """Create a mock data source service."""
    service = AsyncMock(spec=IDataSourceService)
    service.create_blob_data_source_async = AsyncMock()
    return service


@pytest.fixture
def mock_search_index_service():
    """Create a mock search index service."""
    service = AsyncMock(spec=ISearchIndexService)
    service.create_search_index_async = AsyncMock()
    return service


@pytest.fixture
def mock_skillset_service():
    """Create a mock skillset service."""
    service = AsyncMock(spec=ISkillsetService)
    service.create_skillset_using_sdk_async = AsyncMock()
    return service


@pytest.fixture
def mock_indexer_service():
    """Create a mock indexer service."""
    service = AsyncMock(spec=IIndexerService)
    service.create_indexer_async = AsyncMock()
    service.run_indexer_async = AsyncMock()
    service.get_indexer_status_async = AsyncMock()
    return service


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
        skillset_api_version="2024-07-01",
    )


@pytest.fixture
def orchestrator(
    mock_data_source_service,
    mock_search_index_service,
    mock_skillset_service,
    mock_indexer_service,
    search_options,
    mock_logger,
):
    """Create a SearchPipelineOrchestrator instance for testing."""
    return SearchPipelineOrchestrator(
        data_source_service=mock_data_source_service,
        search_index_service=mock_search_index_service,
        skillset_service=mock_skillset_service,
        indexer_service=mock_indexer_service,
        search_options=search_options,
        logger=mock_logger,
    )


@pytest.mark.asyncio
async def test_setup_pipeline_async_success(
    orchestrator,
    mock_data_source_service,
    mock_search_index_service,
    mock_skillset_service,
    mock_indexer_service,
    search_options,
):
    """Test successful pipeline setup with all components."""
    await orchestrator.setup_pipeline_async()

    # Verify all services were called in order
    mock_data_source_service.create_blob_data_source_async.assert_called_once_with(
        search_options.data_source_name
    )
    mock_search_index_service.create_search_index_async.assert_called_once_with(
        search_options.index_name
    )
    mock_skillset_service.create_skillset_using_sdk_async.assert_called_once_with(
        search_options.skillset_name, search_options.index_name
    )
    mock_indexer_service.create_indexer_async.assert_called_once_with(
        search_options.indexer_name,
        search_options.data_source_name,
        search_options.index_name,
        search_options.skillset_name,
    )


@pytest.mark.asyncio
async def test_setup_pipeline_async_partial_failure(
    orchestrator, mock_search_index_service
):
    """Test pipeline setup stops on component failure."""
    # Simulate failure in search index creation
    mock_search_index_service.create_search_index_async.side_effect = Exception(
        "Index creation failed"
    )

    with pytest.raises(Exception, match="Index creation failed"):
        await orchestrator.setup_pipeline_async()


@pytest.mark.asyncio
async def test_run_indexer_async_success(
    orchestrator, mock_indexer_service, search_options
):
    """Test successful indexer run."""
    await orchestrator.run_indexer_async()

    # Verify indexer service was called
    mock_indexer_service.run_indexer_async.assert_called_once_with(
        search_options.indexer_name
    )


@pytest.mark.asyncio
async def test_is_first_run_async_true(orchestrator, mock_indexer_service):
    """Test is_first_run returns True when indexer doesn't exist."""
    # Simulate indexer not found
    mock_indexer_service.get_indexer_status_async.side_effect = ResourceNotFoundError(
        "Indexer not found"
    )

    result = await orchestrator.is_first_run_async()

    assert result is True


@pytest.mark.asyncio
async def test_is_first_run_async_false(orchestrator, mock_indexer_service):
    """Test is_first_run returns False when indexer exists."""
    # Simulate indexer exists
    mock_status = AsyncMock()
    mock_indexer_service.get_indexer_status_async.return_value = mock_status

    result = await orchestrator.is_first_run_async()

    assert result is False


@pytest.mark.asyncio
async def test_pipeline_orchestration_order(
    orchestrator,
    mock_data_source_service,
    mock_search_index_service,
    mock_skillset_service,
    mock_indexer_service,
):
    """Test that pipeline components are created in the correct order."""
    call_order = []

    # Track call order
    mock_data_source_service.create_blob_data_source_async.side_effect = (
        lambda *args: call_order.append("datasource")
    )
    mock_search_index_service.create_search_index_async.side_effect = (
        lambda *args: call_order.append("index")
    )
    mock_skillset_service.create_skillset_using_sdk_async.side_effect = (
        lambda *args: call_order.append("skillset")
    )
    mock_indexer_service.create_indexer_async.side_effect = (
        lambda *args: call_order.append("indexer")
    )

    await orchestrator.setup_pipeline_async()

    # Verify correct order
    assert call_order == ["datasource", "index", "skillset", "indexer"]
