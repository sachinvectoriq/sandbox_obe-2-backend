"""Unit tests for chat service."""

import uuid
import pytest
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime

from app.services.chat_service import ChatService
from app.services.search_service import ISearchService
from app.models import RetrievedDocument, AzureOpenAIOptions, AgenticRAGState, Citation
from app.api.schemas.chat import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    SearchFilters,
)


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
def mock_search_service():
    """Create a mock search service."""
    service = AsyncMock(spec=ISearchService)
    return service


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer.
    
    Uses a simplified approximation of 1 token per 4 characters.
    This is sufficient for unit tests as we're testing the logic flow
    and boundary conditions, not the exact tokenization behavior.
    Real tokenization varies by content type, language, and model.
    """
    tokenizer = Mock()
    # Mock encode to return a list of tokens based on text length
    # Rough approximation: 1 token per 4 characters
    tokenizer.encode = lambda text: [0] * (len(text) // 4 + 1)
    return tokenizer


@pytest.fixture
def mock_logger():
    """Create a mock Logger instance."""
    logger = MagicMock()
    return logger


@pytest.fixture
def chat_service(mock_search_service, mock_openai_options, mock_tokenizer, mock_logger, workflow_options):
    """Create a ChatService instance for testing."""
    service = ChatService(
        search_service=mock_search_service,
        openai_options=mock_openai_options,
        logger=mock_logger,
        workflow_options=workflow_options,
    )
    # Pre-set the tokenizer to avoid network calls
    service._tokenizer = mock_tokenizer
    return service


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing."""
    return [
        RetrievedDocument(
            document_id="doc-1-chunk-1",
            content_id="content-1",
            content="Azure Cosmos DB is a fully managed NoSQL database service.",
            title="Azure Guide.pdf",
            source="/documents/azure-guide.pdf",
            page_number=5,
            score=0.95,
            metadata={"page_number": 5, "reranker_score": 3.2},
        ),
        RetrievedDocument(
            document_id="doc-2-chunk-1",
            content_id="content-2",
            content="Cosmos DB offers multiple consistency levels.",
            title="Database Basics.pdf",
            source="/documents/database-basics.pdf",
            page_number=12,
            score=0.85,
            metadata={"page_number": 12, "reranker_score": 2.8},
        ),
    ]


class TestChatServiceContextBuilder:
    """Tests for context building from search results."""

    def test_build_context_with_results(self, chat_service, sample_search_results):
        """Test context building with search results."""
        context = chat_service._build_context_from_results(sample_search_results)

        assert "Azure Guide.pdf" in context
        assert "Page 5" in context
        assert "Azure Cosmos DB is a fully managed NoSQL database service." in context
        assert "Database Basics.pdf" in context
        assert "Source 1" in context
        assert "Source 2" in context

    def test_build_context_empty_results(self, chat_service):
        """Test context building with no results."""
        context = chat_service._build_context_from_results([])
        assert context == "No relevant documents found."

    def test_build_context_missing_page_number(self, chat_service):
        """Test context building when page number is None."""
        results = [
            RetrievedDocument(
                document_id="doc-1",
                content_id="content-1",
                content="Some content",
                title="Doc.pdf",
                source="/documents/doc.pdf",
                score=0.9,
                metadata={},
            )
        ]
        context = chat_service._build_context_from_results(results)
        assert "Doc.pdf" in context
        assert "Page" not in context

    def test_build_context_with_token_limit(self, chat_service):
        """Test context building respects token limits."""
        # Create results with content that would exceed a small token limit
        large_content = "A" * 1000  # Large content
        results = [
            RetrievedDocument(
                document_id="doc-1",
                content_id="content-1",
                content=large_content,
                title="Doc1.pdf",
                source="/documents/doc1.pdf",
                score=0.9,
            ),
            RetrievedDocument(
                document_id="doc-2",
                content_id="content-2",
                content=large_content,
                title="Doc2.pdf",
                source="/documents/doc2.pdf",
                score=0.85,
            ),
        ]
        
        # Build context with a small token limit
        context = chat_service._build_context_from_results(results, max_tokens=100)
        
        # Should still have some content
        assert len(context) > 0
        # Should be truncated (won't include both full documents)
        assert len(context) < len(large_content) * 2

    def test_build_context_truncates_at_document_boundary(self, chat_service):
        """Test that context truncation happens at document boundaries when possible."""
        results = [
            RetrievedDocument(
                document_id="doc-1",
                content_id="content-1",
                content="Short content",
                title="Doc1.pdf",
                source="/documents/doc1.pdf",
                score=0.9,
            ),
            RetrievedDocument(
                document_id="doc-2",
                content_id="content-2",
                content="A" * 10000,  # Very large content
                title="Doc2.pdf",
                source="/documents/doc2.pdf",
                score=0.85,
            ),
        ]
        
        # Set a token limit that should allow first doc but not second
        context = chat_service._build_context_from_results(results, max_tokens=50)
        
        # Should include first document
        assert "Doc1.pdf" in context
        assert "Short content" in context


class TestChatServiceTokenCounting:
    """Tests for token counting functionality."""

    def test_count_tokens(self, chat_service):
        """Test token counting for text."""
        text = "This is a test message."
        token_count = chat_service._count_tokens(text)
        assert token_count > 0
        assert token_count < 100  # Should be a small number

    def test_get_max_context_tokens(self, chat_service):
        """Test calculation of max context tokens."""
        max_tokens = chat_service._get_max_context_tokens()
        # Should be less than model limit but reasonable
        assert max_tokens > 10000
        assert max_tokens < 130000  # Less than gpt-4o's 128k limit

    def test_estimate_message_tokens(self, chat_service):
        """Test estimation of message tokens."""
        message = "What is Azure?"
        tokens = chat_service._estimate_message_tokens(message)
        assert tokens > 0

    def test_estimate_message_tokens_with_history(self, chat_service):
        """Test token estimation with conversation history."""
        message = "Follow up question"
        history = [
            ChatMessage(role=MessageRole.USER, content="First question"),
            ChatMessage(role=MessageRole.ASSISTANT, content="First answer"),
        ]
        
        tokens_with_history = chat_service._estimate_message_tokens(message, history)
        tokens_without_history = chat_service._estimate_message_tokens(message)
        
        # Should have more tokens with history
        assert tokens_with_history > tokens_without_history


class TestChatServiceMessageBuilder:
    """Tests for OpenAI message array building."""

    def test_build_messages_basic(self, chat_service):
        """Test basic message building without history."""
        messages = chat_service._build_messages(
            message="What is Azure?",
            context="Azure is a cloud platform.",
            conversation_history=None,
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Azure is a cloud platform." in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is Azure?"

    def test_build_messages_with_history(self, chat_service):
        """Test message building with conversation history."""
        history = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
        ]

        messages = chat_service._build_messages(
            message="Follow up question",
            context="Some context",
            conversation_history=history,
        )

        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hi there!"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Follow up question"


class TestChatServiceAgenticSessionId:
    """Tests for agentic session ID validation."""

    async def test_query_async_validates_session_id_format(
        self,
        mock_search_service,
        mock_openai_options,
        mock_logger,
        workflow_options,
    ):
        """Test that query_async validates session_id is a valid UUID."""
        service = ChatService(
            search_service=mock_search_service,
            openai_options=mock_openai_options,
            logger=mock_logger,
            workflow_options=workflow_options,
            workflow=cast(Any, MagicMock()),
        )

        # Test with invalid UUID format
        with pytest.raises(ValueError, match="Invalid session_id format"):
            await service.query_async(
                query="What is Cosmos DB?",
                user_id="user123",
                session_id="not-a-valid-uuid",
                conversation_history=None,
                filters=None,
            )

    async def test_query_async_accepts_valid_session_id(
        self,
        mock_search_service,
        mock_openai_options,
        mock_logger,
        workflow_options,
        monkeypatch,
    ):
        """Test that query_async accepts valid UUID session_id."""
        class DummyWorkflowOutputEvent:
            def __init__(self, data):
                self.data = data

        class DummyAgentRunUpdateEvent:
            def __init__(self, executor_id: str = "dummy"):
                self.executor_id = executor_id

        # Patch agent_framework event types used by ChatService.query_async
        import app.services.chat_service as chat_service_module

        monkeypatch.setattr(chat_service_module, "WorkflowOutputEvent", DummyWorkflowOutputEvent)
        monkeypatch.setattr(chat_service_module, "AgentRunUpdateEvent", DummyAgentRunUpdateEvent)

        final_state = AgenticRAGState(
            query="What is Cosmos DB?",
            user_id="user123",
            session_id="550e8400-e29b-41d4-a716-446655440000",
            current_attempt=1,
            answer="Test answer",
            citations=[Citation(document_id="doc1", content_id="content-1", document_title="Doc 1", page_number=None)],
            vetted_results=[],
            thought_process=[],
            search_history=[],
            decisions=[],
        )

        class DummyWorkflow:
            async def run_stream(self, initial_state):
                yield DummyWorkflowOutputEvent(final_state)

        service = ChatService(
            search_service=mock_search_service,
            openai_options=mock_openai_options,
            logger=mock_logger,
            workflow_options=workflow_options,
            workflow=cast(Any, DummyWorkflow()),
        )

        valid_session_id = "550e8400-e29b-41d4-a716-446655440000"
        response = await service.query_async(
            query="What is Cosmos DB?",
            user_id="user123",
            session_id=valid_session_id,
            conversation_history=None,
            filters=None,
        )

        assert response.session_id == valid_session_id

    async def test_query_async_with_filters(
        self,
        mock_search_service,
        mock_openai_options,
        mock_logger,
        workflow_options,
        monkeypatch,
    ):
        """Test that query_async passes filters to workflow state."""
        class DummyWorkflowOutputEvent:
            def __init__(self, data):
                self.data = data

        class DummyAgentRunUpdateEvent:
            def __init__(self, executor_id: str = "dummy"):
                self.executor_id = executor_id

        # Patch agent_framework event types
        import app.services.chat_service as chat_service_module

        monkeypatch.setattr(chat_service_module, "WorkflowOutputEvent", DummyWorkflowOutputEvent)
        monkeypatch.setattr(chat_service_module, "AgentRunUpdateEvent", DummyAgentRunUpdateEvent)

        final_state = AgenticRAGState(
            query="What is Cosmos DB?",
            user_id="user123",
            session_id="550e8400-e29b-41d4-a716-446655440000",
            filters={"opco_values": ["TEKsystems"], "persona_values": ["Front Office"]},
            current_attempt=1,
            answer="Test answer with filters applied",
            citations=[Citation(document_id="doc1", content_id="content-1", document_title="Doc 1", page_number=None)],
            vetted_results=[],
            thought_process=[],
            search_history=[],
            decisions=[],
        )

        # Track the initial state passed to workflow
        captured_initial_state = None

        class DummyWorkflow:
            async def run_stream(self, initial_state):
                nonlocal captured_initial_state
                captured_initial_state = initial_state
                yield DummyWorkflowOutputEvent(final_state)

        service = ChatService(
            search_service=mock_search_service,
            openai_options=mock_openai_options,
            logger=mock_logger,
            workflow_options=workflow_options,
            workflow=cast(Any, DummyWorkflow()),
        )

        valid_session_id = "550e8400-e29b-41d4-a716-446655440000"
        
        filters = SearchFilters(
            opco_values=["TEKsystems"],
            persona_values=["Front Office"]
        )
        
        response = await service.query_async(
            query="What is Cosmos DB?",
            user_id="user123",
            session_id=valid_session_id,
            conversation_history=None,
            filters=filters,
        )

        # Verify filters were passed to the workflow state
        assert captured_initial_state is not None
        assert captured_initial_state.filters is not None
        assert captured_initial_state.filters["opco_values"] == ["TEKsystems"]
        assert captured_initial_state.filters["persona_values"] == ["Front Office"]
        assert response.answer == "Test answer with filters applied"


class TestChatServiceCitations:
    """Tests for citation creation."""

    def test_create_citations(self, chat_service, sample_search_results):
        """Test citation creation from search results."""
        citations = chat_service._citation_tracker.create_citations(sample_search_results)

        assert len(citations) == 2
        assert citations[0].content_id == "content-1"
        assert citations[0].document_title == "Azure Guide.pdf"
        assert citations[0].page_number == 5
        # Citation model includes basic identifiers + page metadata only

    def test_create_citations_handles_long_content(self, chat_service):
        """Test that long content does not break citation creation."""
        long_content = "A" * 600  # More than 500 characters
        results = [
            RetrievedDocument(
                document_id="doc-1",
                content_id="content-1",
                content=long_content,
                title="Long Doc.pdf",
                source="/documents/long-doc.pdf",
                score=0.9,
            )
        ]

        citations = chat_service._citation_tracker.create_citations(results)

        assert len(citations) == 1
        assert citations[0].content_id == "content-1"
        assert citations[0].document_title == "Long Doc.pdf"

    def test_create_citations_no_reranker_score(self, chat_service):
        """Test citation when only search score is available."""
        results = [
            RetrievedDocument(
                document_id="doc-1",
                content_id="content-1",
                content="Content",
                title="Doc.pdf",
                source="/documents/doc.pdf",
                score=0.85,
                metadata={},
            )
        ]

        citations = chat_service._citation_tracker.create_citations(results)
        assert len(citations) == 1
        assert citations[0].content_id == "content-1"


class TestChatServiceChat:
    """Tests for the main chat functionality."""

    @pytest.mark.asyncio
    async def test_chat_success(
        self, chat_service, mock_search_service, sample_search_results
    ):
        """Test successful chat interaction."""
        # Setup mock search service
        mock_search_service.search_async = AsyncMock(return_value=sample_search_results)

        # Setup mock OpenAI client
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content="Azure Cosmos DB is a great database!"))
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        with patch.object(
            chat_service, "_get_openai_client", return_value=mock_client
        ):
            response = await chat_service.chat_async(
                message="What is Azure Cosmos DB?",
                top_k=5,
            )

        assert isinstance(response, ChatResponse)
        assert response.message == "Azure Cosmos DB is a great database!"
        assert len(response.citations) == 2
        assert response.timestamp is not None

        # Verify search was called
        mock_search_service.search_async.assert_called_once_with(
            query="What is Azure Cosmos DB?",
            top_k=5,
            search_mode="hybrid",
            filters=None,
            use_semantic_ranking=True,
            deduplicate=True,
        )

    @pytest.mark.asyncio
    async def test_chat_with_filters(
        self, chat_service, mock_search_service, sample_search_results
    ):
        """Test chat with search filters."""
        mock_search_service.search_async = AsyncMock(return_value=sample_search_results)

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Response"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        filters = SearchFilters(
            opco_values=["TEKsystems"],
            persona_values=["Developer"],
        )

        with patch.object(
            chat_service, "_get_openai_client", return_value=mock_client
        ):
            await chat_service.chat_async(
                message="Test question",
                filters=filters,
                top_k=3,
            )

        # Verify filters were passed to search
        mock_search_service.search_async.assert_called_once_with(
            query="Test question",
            top_k=3,
            search_mode="hybrid",
            filters={"opco_values": ["TEKsystems"], "persona_values": ["Developer"]},
            use_semantic_ranking=True,
            deduplicate=True,
        )

    @pytest.mark.asyncio
    async def test_chat_with_conversation_history(
        self, chat_service, mock_search_service, sample_search_results
    ):
        """Test chat with conversation history."""
        mock_search_service.search_async = AsyncMock(return_value=sample_search_results)

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Follow up response"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        history = [
            ChatMessage(role=MessageRole.USER, content="Previous question"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Previous answer"),
        ]

        with patch.object(
            chat_service, "_get_openai_client", return_value=mock_client
        ):
            response = await chat_service.chat_async(
                message="Follow up",
                conversation_history=history,
            )

        assert response.message == "Follow up response"

        # Verify OpenAI was called with history in messages
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 4  # system + 2 history + current user

    @pytest.mark.asyncio
    async def test_chat_no_search_results(self, chat_service, mock_search_service):
        """Test chat when search returns no results."""
        mock_search_service.search_async = AsyncMock(return_value=[])

        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content="I couldn't find relevant information."))
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        with patch.object(
            chat_service, "_get_openai_client", return_value=mock_client
        ):
            response = await chat_service.chat_async(message="Unknown topic")

        assert response.message == "I couldn't find relevant information."
        assert len(response.citations) == 0
