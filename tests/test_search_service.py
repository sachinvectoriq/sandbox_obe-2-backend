"""Unit tests for search service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.search_service import SearchService
from app.models import AzureOpenAIOptions


@pytest.fixture
def mock_openai_options():
    """Create mock Azure OpenAI options."""
    return AzureOpenAIOptions(
        resource_uri="https://test.openai.azure.com",
        api_key="test-api-key",
        text_embedding_model="text-embedding-3-large",
        chat_completion_model="gpt-4o",
    )


@pytest.fixture
def mock_search_client():
    """Create a mock async SearchClient."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_logger():
    """Create a mock Logger instance."""
    logger = MagicMock()
    return logger


@pytest.fixture
def search_service(mock_search_client, mock_openai_options, mock_logger):
    """Create a SearchService instance for testing."""
    return SearchService(
        search_client=mock_search_client,
        openai_options=mock_openai_options,
        logger=mock_logger,
    )


class TestSearchServiceFilterBuilder:
    """Tests for OData filter expression building."""

    def test_build_filter_no_filters(self, search_service):
        """Test that no filter returns None."""
        result = search_service._build_filter_expression(None, None)
        assert result is None

    def test_build_filter_single_opco(self, search_service):
        """Test single opco filter."""
        result = search_service._build_filter_expression(
            filters={"opco_values": ["TEKsystems"]},
            exclude_ids=None,
        )
        assert result == "(opco_values/any(o: o eq 'TEKsystems'))"

    def test_build_filter_multiple_opco(self, search_service):
        """Test multiple opco values (OR condition)."""
        result = search_service._build_filter_expression(
            filters={"opco_values": ["TEKsystems", "Aston Carter"]},
            exclude_ids=None,
        )
        assert "opco_values/any(o: o eq 'TEKsystems')" in result
        assert "opco_values/any(o: o eq 'Aston Carter')" in result
        assert " or " in result

    def test_build_filter_single_persona(self, search_service):
        """Test single persona filter."""
        result = search_service._build_filter_expression(
            filters={"persona_values": ["Developer"]},
            exclude_ids=None,
        )
        assert result == "(persona_values/any(p: p eq 'Developer'))"

    def test_build_filter_combined(self, search_service):
        """Test combined opco and persona filters (AND condition)."""
        result = search_service._build_filter_expression(
            filters={"opco_values": ["TEKsystems"], "persona_values": ["Developer"]},
            exclude_ids=None,
        )
        assert "opco_values/any(o: o eq 'TEKsystems')" in result
        assert "persona_values/any(p: p eq 'Developer')" in result
        assert " and " in result


class TestSearchServiceEmbedding:
    """Tests for embedding generation."""

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, search_service):
        """Test successful embedding generation."""
        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 3072)]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        with patch.object(
            search_service, "_get_openai_client", return_value=mock_client
        ):
            result = await search_service.generate_embedding_async("test query")

        assert len(result) == 3072
        assert result[0] == 0.1
        mock_client.embeddings.create.assert_called_once()


class TestSearchServiceSearch:
    """Tests for hybrid search functionality."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, search_service, mock_search_client):
        """Test that search returns properly formatted results."""
        # Mock embedding generation
        mock_embedding = [0.1] * 3072

        # Mock search results (async iterator)
        mock_results = [
            {
                "text_document_id": "doc-1",
                "content_id": "doc-1-chunk-1",
                "content_text": "This is test content about Azure.",
                "document_title": "Azure Guide.pdf",
                "content_path": "/documents/azure-guide.pdf",
                "location_metadata": {"pageNumber": 5},
                "@search.score": 0.95,
                "@search.reranker_score": 3.2,
            },
            {
                "text_document_id": "doc-2",
                "content_id": "doc-2-chunk-1",
                "content_text": "More content about cloud services.",
                "document_title": "Cloud Overview.pdf",
                "content_path": "/documents/cloud-overview.pdf",
                "location_metadata": {},
                "@search.score": 0.85,
                "@search.reranker_score": 2.8,
            },
        ]

        # Create async iterator for mock results
        async def mock_search_iter():
            for result in mock_results:
                yield result

        mock_search_client.search = AsyncMock(return_value=mock_search_iter())

        with patch.object(
            search_service,
            "generate_embedding_async",
            return_value=mock_embedding,
        ):
            results = await search_service.search_async(
                query="What is Azure?",
                top_k=5,
            )

        assert len(results) == 2
        assert results[0].document_id == "doc-1"
        assert results[0].content_id == "doc-1-chunk-1"
        assert results[0].content == "This is test content about Azure."
        assert results[0].title == "Azure Guide.pdf"
        assert results[0].page_number == 5
        assert results[0].score == 0.95
        assert results[0].metadata.get("reranker_score") == 3.2

        # Second result should have None page_number
        assert results[1].page_number is None

    @pytest.mark.asyncio
    async def test_search_with_filters(self, search_service, mock_search_client):
        """Test that search applies filters correctly."""
        mock_embedding = [0.1] * 3072

        async def mock_search_iter():
            if False:  # pragma: no cover
                yield  # Empty async generator

        mock_search_client.search = AsyncMock(return_value=mock_search_iter())

        with patch.object(
            search_service,
            "generate_embedding_async",
            return_value=mock_embedding,
        ):
            await search_service.search_async(
                query="test",
                top_k=5,
                filters={"opco_values": ["TEKsystems"], "persona_values": ["Developer"]},
            )

        # Verify filter was passed to search
        call_kwargs = mock_search_client.search.call_args.kwargs
        assert call_kwargs["filter"] is not None
        assert "opco_values" in call_kwargs["filter"]
        assert "persona_values" in call_kwargs["filter"]

    @pytest.mark.asyncio
    async def test_search_empty_results(self, search_service, mock_search_client):
        """Test search with no results."""
        mock_embedding = [0.1] * 3072

        async def mock_search_iter():
            if False:  # pragma: no cover
                yield  # Empty async generator

        mock_search_client.search = AsyncMock(return_value=mock_search_iter())

        with patch.object(
            search_service,
            "generate_embedding_async",
            return_value=mock_embedding,
        ):
            results = await search_service.search_async(query="nonexistent", top_k=5)

        assert len(results) == 0
