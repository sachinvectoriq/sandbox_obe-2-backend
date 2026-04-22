"""Unit tests for ConversationService."""
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from azure.cosmos import exceptions
from app.services.conversation_service import ConversationService
from agent_framework import ChatMessage, Role
@pytest.fixture
def mock_cosmos_client():
    """Create a mock Cosmos DB client."""
    client = MagicMock()
    db = MagicMock()
    container = AsyncMock()
    # Make these synchronous calls
    client.get_database_client = MagicMock(return_value=db)
    db.get_container_client = MagicMock(return_value=container)
    return client, db, container
@pytest.fixture
def mock_logger():
    """Create a mock Logger instance."""
    return MagicMock()
@pytest.fixture
def conversation_service(mock_cosmos_client, mock_logger):
    """Create a ConversationService instance for testing."""
    client, db, container = mock_cosmos_client
    service = ConversationService(
        cosmos_client=client,
        database_name="test_db",
        container_name="test_container",
        logger=mock_logger,
    )
    # Store container reference for test assertions
    service._test_container = container
    return service
class TestConversationServiceRoleConversion:
    """Tests for role conversion utilities."""
    def test_role_to_str_with_enum(self, conversation_service):
        """Test converting Role enum to string."""
        result = conversation_service._role_to_str(Role.USER)
        assert result == "user"
    def test_role_to_str_with_string(self, conversation_service):
        """Test converting string role."""
        result = conversation_service._role_to_str("assistant")
        assert result == "assistant"
    def test_payload_to_chat_message(self, conversation_service):
        """Test converting payload dict to ChatMessage."""
        payload = {
            "id": "msg123",
            "role": "user",
            "text": "Hello, world!"
        }
        message = conversation_service._payload_to_chat_message(payload)
        assert isinstance(message, ChatMessage)
        assert message.role == Role.USER
        assert message.text == "Hello, world!"
    def test_payload_to_chat_message_missing_fields(self, conversation_service):
        """Test converting payload with missing optional fields."""
        payload = {"role": "assistant", "text": "Response"}
        message = conversation_service._payload_to_chat_message(payload)
        assert message.role == Role.ASSISTANT
        assert message.text == "Response"
class TestSanitizeMessageText:
    """Tests for message text sanitization functionality."""
    def test_sanitize_message_text_numeric_citations_single(self):
        """Test sanitizing single numeric citations."""
        from app.services.conversation_service import ConversationService
        text = "Azure Cosmos DB is a database [1]."
        result = ConversationService.sanitize_message_text(text)
        assert result == "Azure Cosmos DB is a database."
    def test_sanitize_message_text_numeric_citations_multiple(self):
        """Test sanitizing multiple numeric citations."""
        from app.services.conversation_service import ConversationService
        text = "Azure supports multiple APIs [1][2][3] for flexibility."
        result = ConversationService.sanitize_message_text(text)
        assert result == "Azure supports multiple APIs for flexibility."
    def test_sanitize_message_text_numeric_citations_scattered(self):
        """Test sanitizing scattered numeric citations."""
        from app.services.conversation_service import ConversationService
        text = "First point [1]. Second point [2]. Third point [3]."
        result = ConversationService.sanitize_message_text(text)
        assert result == "First point. Second point. Third point."
    def test_sanitize_message_text_content_id_citations_single(self):
        """Test sanitizing single content ID citations."""
        from app.services.conversation_service import ConversationService
        text = "CRG works through Connected {e6e5902f3f2d_aHR0cHM6Ly9zdHJn} system."
        result = ConversationService.sanitize_message_text(text)
        assert result == "CRG works through Connected system."
    def test_sanitize_message_text_content_id_citations_long(self):
        """Test sanitizing long content ID citations."""
        from app.services.conversation_service import ConversationService
        text = "Process starts {e6e5902f3f2d_aHR0cHM6Ly9zdHJnYWxsZWdpc29jbWthMDAxNWE2MDAuYmxvYi5jb3JlLndpbmRvd3MubmV0L2RvY3VtZW50cy9FbnRlcmluZyUyMENSRyUyMGluZm8lMjBpbnRvJTIwQ29ubmVjdGVkJTIwKHJlbGF0ZWQlMjByZWNvcmRzKV90YWdnZWQucGRm0_normalized_images_7} in the system."
        result = ConversationService.sanitize_message_text(text)
        assert result == "Process starts in the system."
    def test_sanitize_message_text_mixed_citations(self):
        """Test sanitizing both numeric and content ID citations together."""
        from app.services.conversation_service import ConversationService
        text = "First point [1][2]. Second point {abc123_xyz}. Third point [3]."
        result = ConversationService.sanitize_message_text(text)
        assert result == "First point. Second point. Third point."
    def test_sanitize_message_text_many_citations_single_sentence(self):
        """Test sanitizing many citations (10+) from a single sentence."""
        from app.services.conversation_service import ConversationService
        text = "This sentence has many citations [1][2][3][4][5][6][7][8][9][10][11][12] in the middle. [13][14][15][16][17][18][19][20]"
        result = ConversationService.sanitize_message_text(text)
        assert result == "This sentence has many citations in the middle."
        # Verify all 20 citations removed
        assert "[" not in result
        assert "]" not in result
    def test_sanitize_message_text_with_multiple_spaces(self):
        """Test that multiple spaces are collapsed after citation removal."""
        from app.services.conversation_service import ConversationService
        text = "Word1 [1]  [2]  word2."
        result = ConversationService.sanitize_message_text(text)
        assert result == "Word1 word2."
    def test_sanitize_message_text_with_space_before_newline(self):
        """Test that spaces before newlines are removed."""
        from app.services.conversation_service import ConversationService
        text = "First sentence [1] \nSecond sentence."
        result = ConversationService.sanitize_message_text(text)
        assert result == "First sentence\nSecond sentence."
    def test_sanitize_message_text_with_space_after_newline(self):
        """Test that spaces after newlines are removed."""
        from app.services.conversation_service import ConversationService
        text = "First sentence.\n [1]Second sentence."
        result = ConversationService.sanitize_message_text(text)
        assert result == "First sentence.\nSecond sentence."
    def test_sanitize_message_text_with_space_before_period(self):
        """Test that spaces before periods are removed."""
        from app.services.conversation_service import ConversationService
        text = "This is a sentence [1] ."
        result = ConversationService.sanitize_message_text(text)
        assert result == "This is a sentence."
    def test_sanitize_message_text_with_space_before_comma(self):
        """Test that spaces before commas are removed."""
        from app.services.conversation_service import ConversationService
        text = "Item one [1] , item two [2] , item three."
        result = ConversationService.sanitize_message_text(text)
        assert result == "Item one, item two, item three."
    def test_sanitize_message_text_with_multiple_newlines(self):
        """Test that multiple consecutive newlines are reduced to max 2."""
        from app.services.conversation_service import ConversationService
        text = "Paragraph one [1].\n\n\n\nParagraph two [2]."
        result = ConversationService.sanitize_message_text(text)
        assert result == "Paragraph one.\n\nParagraph two."
    def test_sanitize_message_text_empty_text(self):
        """Test sanitizing empty text."""
        from app.services.conversation_service import ConversationService
        text = ""
        result = ConversationService.sanitize_message_text(text)
        assert result == ""
    def test_sanitize_message_text_no_citations(self):
        """Test text without citations remains unchanged (except whitespace normalization)."""
        from app.services.conversation_service import ConversationService
        text = "This is plain text without citations."
        result = ConversationService.sanitize_message_text(text)
        assert result == "This is plain text without citations."
    def test_sanitize_message_text_only_citations(self):
        """Test text that is only citations."""
        from app.services.conversation_service import ConversationService
        text = "[1][2][3]{abc123}"
        result = ConversationService.sanitize_message_text(text)
        assert result == ""
    def test_sanitize_message_text_preserves_brackets_in_text(self):
        """Test that non-citation brackets are preserved."""
        from app.services.conversation_service import ConversationService
        text = "The array [a, b, c] contains values [1]."
        result = ConversationService.sanitize_message_text(text)
        # [1] is removed but [a, b, c] should remain... actually no, [a, b, c] has no digits
        # so it should be preserved as it doesn't match \d+
        assert result == "The array [a, b, c] contains values."
    def test_sanitize_message_text_real_world_example(self):
        """Test with a real-world example from the issue."""
        from app.services.conversation_service import ConversationService
        text = """CRG works through a structured workflow in the Connected system, where users input customer onboarding, compliance, and ancillary requirements [1][2][3]. The process begins in the Initiated Stage {e6e5902f3f2d_aHR0cHM6Ly9zdHJnYWxsZWdpc29jbWthMDAxNWE2MDAuYmxvYi5jb3JlLndpbmRvd3MubmV0L2RvY3VtZW50cy9FbnRlcmluZyUyMENSRyUyMGluZm8lMjBpbnRvJTIwQ29ubmVjdGVkJTIwKHJlbGF0ZWQlMjByZWNvcmRzKV90YWdnZWQucGRm0_normalized_images_7}[4][5]. Each section within the CRG is completed based on requirements [1][2][3]."""
        result = ConversationService.sanitize_message_text(text)
        # Check that all citations are removed
        assert "[1]" not in result
        assert "[2]" not in result
        assert "[3]" not in result
        assert "[4]" not in result
        assert "[5]" not in result
        assert "{e6e5902f3f2d" not in result
        assert "}" not in result or result.count("}") == 0
        # Check that content is preserved
        assert "CRG works through a structured workflow" in result
        assert "Connected system" in result
        assert "Initiated Stage" in result
        assert "completed based on requirements" in result
class TestConversationServiceGetHistory:
    """Tests for retrieving conversation history."""
    @pytest.mark.asyncio
    async def test_get_conversation_history_success(self, conversation_service):
        """Test successful retrieval of conversation history."""
        # Setup mock data
        mock_items = [
            {
                "id": "user123_session456_msg1",
                "user_id": "user123",
                "session_id": "session456",
                "timestamp": "2025-01-01T10:00:00Z",
                "serialized_message": json.dumps({
                    "id": "msg1",
                    "role": "user",
                    "text": "Hello",
                    "author_name": None
                }),
                "message_text": "Hello",
                "message_id": "msg1",
                "role": "user"
            },
            {
                "id": "user123_session456_msg2",
                "user_id": "user123",
                "session_id": "session456",
                "timestamp": "2025-01-01T10:01:00Z",
                "serialized_message": json.dumps({
                    "id": "msg2",
                    "role": "assistant",
                    "text": "Hi there!",
                    "author_name": None
                }),
                "message_text": "Hi there!",
                "message_id": "msg2",
                "role": "assistant"
            }
        ]
        # Mock async iterator
        async def mock_query_items(*args, **kwargs):
            for item in mock_items:
                yield item
        conversation_service._test_container.query_items = mock_query_items
        # Execute
        messages = await conversation_service.get_conversation_history(
            session_id="session456",
            user_id="user123"
        )
        # Assert
        assert len(messages) == 2
        assert messages[0].role == Role.USER
        assert messages[0].text == "Hello"
        assert messages[1].role == Role.ASSISTANT
        assert messages[1].text == "Hi there!"
    @pytest.mark.asyncio
    async def test_get_conversation_history_with_max_messages(self, conversation_service):
        """Test conversation history with max_messages limit."""
        # Setup mock data with 5 messages
        mock_items = [
            {
                "id": f"user123_session456_msg{i}",
                "user_id": "user123",
                "session_id": "session456",
                "timestamp": f"2025-01-01T10:0{i}:00Z",
                "serialized_message": json.dumps({
                    "id": f"msg{i}",
                    "role": "user" if i % 2 == 0 else "assistant",
                    "text": f"Message {i}",
                    "author_name": None
                }),
                "message_text": f"Message {i}",
                "message_id": f"msg{i}",
                "role": "user" if i % 2 == 0 else "assistant"
            }
            for i in range(5)
        ]
        async def mock_query_items(*args, **kwargs):
            for item in mock_items:
                yield item
        conversation_service._test_container.query_items = mock_query_items
        # Execute with max_messages=3
        messages = await conversation_service.get_conversation_history(
            session_id="session456",
            user_id="user123",
            max_messages=3
        )
        # Assert - should only return last 3 messages
        assert len(messages) == 3
        assert messages[0].text == "Message 2"
        assert messages[2].text == "Message 4"
    @pytest.mark.asyncio
    async def test_get_conversation_history_empty(self, conversation_service):
        """Test retrieving history when no messages exist."""
        async def mock_query_items(*args, **kwargs):
            return
            yield  # Make it a generator but yield nothing
        conversation_service._test_container.query_items = mock_query_items
        messages = await conversation_service.get_conversation_history(
            session_id="session456",
            user_id="user123"
        )
        assert messages == []
    @pytest.mark.asyncio
    async def test_get_conversation_history_cosmos_error(self, conversation_service):
        """Test handling Cosmos DB errors during retrieval."""
        async def mock_query_items(*args, **kwargs):
            raise exceptions.CosmosHttpResponseError(message="DB error")
            yield
        conversation_service._test_container.query_items = mock_query_items
        messages = await conversation_service.get_conversation_history(
            session_id="session456",
            user_id="user123"
        )
        # Should return empty list on error
        assert messages == []
    @pytest.mark.asyncio
    async def test_get_conversation_history_corrupted_message(self, conversation_service):
        """Test handling corrupted serialized_message."""
        mock_items = [
            {
                "id": "user123_session456_msg1",
                "user_id": "user123",
                "session_id": "session456",
                "timestamp": "2025-01-01T10:00:00Z",
                "serialized_message": "INVALID_JSON",
                "message_id": "msg1",
                "message_text": "Fallback text",
                "role": "user"
            }
        ]
        async def mock_query_items(*args, **kwargs):
            for item in mock_items:
                yield item
        conversation_service._test_container.query_items = mock_query_items
        messages = await conversation_service.get_conversation_history(
            session_id="session456",
            user_id="user123"
        )
        # Should use fallback data
        assert len(messages) == 1
        assert messages[0].text == "Fallback text"
        assert messages[0].role == Role.USER
class TestConversationServiceAddMessage:
    """Tests for adding messages to conversation history."""
    @pytest.mark.asyncio
    async def test_add_message_success(self, conversation_service):
        """Test successfully adding a message."""
        conversation_service._test_container.upsert_item = AsyncMock()
        await conversation_service.add_message(
            session_id="session456",
            user_id="user123",
            role="user",
            content="Hello, how are you?"
        )
        # Verify upsert was called
        conversation_service._test_container.upsert_item.assert_called_once()
        # Check the document structure
        call_args = conversation_service._test_container.upsert_item.call_args
        doc = call_args.kwargs["body"]
        assert doc["user_id"] == "user123"
        assert doc["session_id"] == "session456"
        assert doc["role"] == "user"
        assert doc["message_text"] == "Hello, how are you?"
        assert "serialized_message" in doc
        # Check serialized message
        serialized = json.loads(doc["serialized_message"])
        assert serialized["role"] == "user"
        assert serialized["text"] == "Hello, how are you?"
    @pytest.mark.asyncio
    async def test_add_message_with_metadata(self, conversation_service):
        """Test adding a message with metadata."""
        conversation_service._test_container.upsert_item = AsyncMock()
        metadata = {
            "citations": ["doc1", "doc2"],
            "document_count": 5
        }
        await conversation_service.add_message(
            session_id="session456",
            user_id="user123",
            role="assistant",
            content="Here's the answer with [1] and [2] citations.",
            metadata=metadata
        )
        # Check serialized message includes metadata
        call_args = conversation_service._test_container.upsert_item.call_args
        doc = call_args.kwargs["body"]
        serialized = json.loads(doc["serialized_message"])
        # Metadata should be in serialized_message
        assert serialized["citations"] == ["doc1", "doc2"]
        assert serialized["document_count"] == 5
        # Text should have citations stripped
        assert serialized["text"] == "Here's the answer with and citations."
    @pytest.mark.asyncio
    async def test_add_message_cosmos_error(self, conversation_service):
        """Test handling Cosmos DB errors when adding message."""
        conversation_service._test_container.upsert_item = AsyncMock(
            side_effect=exceptions.CosmosHttpResponseError(message="DB error")
        )
        # Should raise exception
        with pytest.raises(exceptions.CosmosHttpResponseError):
            await conversation_service.add_message(
                session_id="session456",
                user_id="user123",
                role="user",
                content="Test message"
            )


class TestConversationServiceListSessions:
    """Tests for listing user sessions."""
    @pytest.mark.asyncio
    async def test_list_user_sessions_success(self, conversation_service):
        """Test listing all sessions for a user."""
        mock_results = [
            {"session_id": "session1"},
            {"session_id": "session2"},
            {"session_id": "session3"}
        ]
        async def mock_query_items(*args, **kwargs):
            for item in mock_results:
                yield item
        conversation_service._test_container.query_items = mock_query_items
        sessions = await conversation_service.list_user_sessions(
            user_id="user123"
        )
        assert len(sessions) == 3
        assert sessions[0]["session_id"] == "session1"
        assert sessions[1]["session_id"] == "session2"
        assert sessions[2]["session_id"] == "session3"
    @pytest.mark.asyncio
    async def test_list_user_sessions_with_max_results(self, conversation_service):
        """Test listing sessions with max_results limit."""
        mock_results = [
            {"session_id": f"session{i}"}
            for i in range(10)
        ]
        async def mock_query_items(*args, **kwargs):
            for item in mock_results:
                yield item
        conversation_service._test_container.query_items = mock_query_items
        sessions = await conversation_service.list_user_sessions(
            user_id="user123",
            max_results=5
        )
        assert len(sessions) == 5
    @pytest.mark.asyncio
    async def test_list_user_sessions_empty(self, conversation_service):
        """Test listing sessions when user has none."""
        async def mock_query_items(*args, **kwargs):
            return
            yield
        conversation_service._test_container.query_items = mock_query_items
        sessions = await conversation_service.list_user_sessions(
            user_id="user123"
        )
        assert sessions == []
    @pytest.mark.asyncio
    async def test_list_user_sessions_cosmos_error(self, conversation_service):
        """Test handling Cosmos DB errors when listing sessions."""
        async def mock_query_items(*args, **kwargs):
            raise exceptions.CosmosHttpResponseError(message="DB error")
            yield
        conversation_service._test_container.query_items = mock_query_items
        sessions = await conversation_service.list_user_sessions(
            user_id="user123"
        )
        assert sessions == []
class TestConversationServiceDeleteSession:
    """Tests for deleting conversation threads."""
    @pytest.mark.asyncio
    async def test_delete_session_success(self, conversation_service):
        """Test successfully deleting a session."""
        mock_messages = [
            {"id": "user123_session456_msg1"},
            {"id": "user123_session456_msg2"},
            {"id": "user123_session456_msg3"}
        ]
        async def mock_query_items(*args, **kwargs):
            for item in mock_messages:
                yield item
        conversation_service._test_container.query_items = mock_query_items
        conversation_service._test_container.delete_item = AsyncMock()
        await conversation_service.delete_session(
            session_id="session456",
            user_id="user123"
        )
        # Should delete all 3 messages
        assert conversation_service._test_container.delete_item.call_count == 3
    @pytest.mark.asyncio
    async def test_delete_session_empty(self, conversation_service):
        """Test deleting a session with no messages."""
        async def mock_query_items(*args, **kwargs):
            return
            yield
        conversation_service._test_container.query_items = mock_query_items
        conversation_service._test_container.delete_item = AsyncMock()
        await conversation_service.delete_session(
            session_id="session456",
            user_id="user123"
        )
        # Should not call delete if no messages
        conversation_service._test_container.delete_item.assert_not_called()
    @pytest.mark.asyncio
    async def test_delete_session_cosmos_error(self, conversation_service):
        """Test handling Cosmos DB errors when deleting session."""
        async def mock_query_items(*args, **kwargs):
            raise exceptions.CosmosHttpResponseError(message="DB error")
            yield
        conversation_service._test_container.query_items = mock_query_items
        with pytest.raises(exceptions.CosmosHttpResponseError):
            await conversation_service.delete_session(
                session_id="session456",
                user_id="user123"
            )
class TestConversationServiceClearUserHistory:
    """Tests for clearing all user conversation history."""
    @pytest.mark.asyncio
    async def test_clear_user_history_success(self, conversation_service):
        """Test successfully clearing all user history."""
        mock_items = [
            {"id": "user123_session1_msg1", "user_id": "user123", "session_id": "session1"},
            {"id": "user123_session1_msg2", "user_id": "user123", "session_id": "session1"},
            {"id": "user123_session2_msg1", "user_id": "user123", "session_id": "session2"},
            {"id": "user123_session3_msg1", "user_id": "user123", "session_id": "session3"}
        ]
        async def mock_query_items(*args, **kwargs):
            for item in mock_items:
                yield item
        conversation_service._test_container.query_items = mock_query_items
        conversation_service._test_container.delete_item = AsyncMock()
        deleted_count = await conversation_service.clear_user_history(
            user_id="user123"
        )
        assert deleted_count == 4
        assert conversation_service._test_container.delete_item.call_count == 4
    @pytest.mark.asyncio
    async def test_clear_user_history_empty(self, conversation_service):
        """Test clearing history when user has no messages."""
        async def mock_query_items(*args, **kwargs):
            return
            yield
        conversation_service._test_container.query_items = mock_query_items
        conversation_service._test_container.delete_item = AsyncMock()
        deleted_count = await conversation_service.clear_user_history(
            user_id="user123"
        )
        assert deleted_count == 0
        conversation_service._test_container.delete_item.assert_not_called()
    @pytest.mark.asyncio
    async def test_clear_user_history_cosmos_error(self, conversation_service):
        """Test handling Cosmos DB errors when clearing history."""
        async def mock_query_items(*args, **kwargs):
            raise exceptions.CosmosHttpResponseError(message="DB error")
            yield
        conversation_service._test_container.query_items = mock_query_items
        with pytest.raises(exceptions.CosmosHttpResponseError):
            await conversation_service.clear_user_history(
                user_id="user123"
            )
