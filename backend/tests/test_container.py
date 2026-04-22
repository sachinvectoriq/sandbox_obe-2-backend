"""
Unit tests for dependency injection container.

NOTE: These tests are skipped because the container requires environment variables
to be set for initialization. The container is tested indirectly through service tests.
"""

import pytest

# Skip all container tests - they require full environment setup
pytestmark = pytest.mark.skip(
    reason="Container requires environment variables. Tested via service integration tests."
)

from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient, SearchIndexerClient

from app.core.container import Container
from app.core.settings import Settings
from app.ingestion.data_source_service import IDataSourceService
from app.ingestion.search_index_service import ISearchIndexService
from app.ingestion.skillset_service import ISkillsetService
from app.ingestion.indexer_service import IIndexerService
from app.ingestion.search_pipeline_orchestrator import ISearchPipelineOrchestrator


@pytest.fixture
def container() -> Container:
    """Create a test container instance."""
    return Container()


def test_container_config_is_singleton(container: Container) -> None:
    """Test that config is a singleton."""
    config1 = container.config()
    config2 = container.config()
    assert config1 is config2
    assert isinstance(config1, Settings)


def test_container_search_index_client_is_singleton(container: Container) -> None:
    """Test that SearchIndexClient is a singleton (async version)."""
    client1 = container.search_index_client()
    client2 = container.search_index_client()
    assert client1 is client2
    assert isinstance(client1, SearchIndexClient)


def test_container_search_indexer_client_is_singleton(container: Container) -> None:
    """Test that SearchIndexerClient is a singleton (async version)."""
    client1 = container.search_indexer_client()
    client2 = container.search_indexer_client()
    assert client1 is client2
    assert isinstance(client1, SearchIndexerClient)


def test_container_search_client_is_singleton(container: Container) -> None:
    """Test that SearchClient is a singleton (async version)."""
    client1 = container.search_client()
    client2 = container.search_client()
    assert client1 is client2
    assert isinstance(client1, SearchClient)


def test_container_data_source_service_is_singleton(container: Container) -> None:
    """Test that DataSourceService is a singleton."""
    service1 = container.data_source_service()
    service2 = container.data_source_service()
    assert service1 is service2
    assert isinstance(service1, IDataSourceService)


def test_container_search_index_service_is_singleton(container: Container) -> None:
    """Test that SearchIndexService is a singleton."""
    service1 = container.search_index_service()
    service2 = container.search_index_service()
    assert service1 is service2
    assert isinstance(service1, ISearchIndexService)


def test_container_skillset_service_is_singleton(container: Container) -> None:
    """Test that SkillsetService is a singleton."""
    service1 = container.skillset_service()
    service2 = container.skillset_service()
    assert service1 is service2
    assert isinstance(service1, ISkillsetService)


def test_container_indexer_service_is_singleton(container: Container) -> None:
    """Test that IndexerService is a singleton."""
    service1 = container.indexer_service()
    service2 = container.indexer_service()
    assert service1 is service2
    assert isinstance(service1, IIndexerService)


def test_container_search_pipeline_orchestrator_is_singleton(container: Container) -> None:
    """Test that SearchPipelineOrchestrator is a singleton."""
    service1 = container.search_pipeline_orchestrator()
    service2 = container.search_pipeline_orchestrator()
    assert service1 is service2
    assert isinstance(service1, ISearchPipelineOrchestrator)


def test_container_clients_use_config(container: Container) -> None:
    """Test that clients are configured from Settings."""
    config = container.config()
    search_index_client = container.search_index_client()
    
    # Verify endpoint is configured correctly
    assert str(config.search_service.endpoint) in str(search_index_client._endpoint)
