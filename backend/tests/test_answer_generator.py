"""Unit tests for AnswerGenerator._extract_cited_documents method."""

import pytest
from unittest.mock import MagicMock
from app.agents.answer_generator import AnswerGenerator
from app.models import RetrievedDocument


@pytest.fixture
def mock_answer_generator(mock_logger):
    """Create a mock AnswerGenerator for testing."""
    # Mock the dependencies
    mock_settings = MagicMock()
    mock_settings.use_managed_identity = False
    mock_settings.azure_openai.endpoint = "https://test.openai.azure.com"
    mock_settings.azure_openai.api_version = "2024-02-01"
    mock_settings.azure_openai.deployment_name = "test-deployment"
    
    mock_citation_tracker = MagicMock()
    
    # Create instance without actually initializing the ChatAgent
    generator = object.__new__(AnswerGenerator)
    generator.settings = mock_settings
    generator.logger = mock_logger
    generator.citation_tracker = mock_citation_tracker
    generator.tracer = MagicMock()
    generator.agent = MagicMock()
    
    return generator


@pytest.fixture
def sample_documents():
    """Create sample RetrievedDocument objects for testing."""
    return [
        RetrievedDocument(
            content_id="doc1_abc123",
            document_id="doc1",
            title="Azure Cosmos DB Guide",
            content="Azure Cosmos DB is a globally distributed database.",
            source="cosmos_guide.pdf",            
            score=0.95,
            page_number=1
        ),
        RetrievedDocument(
            content_id="doc2_def456",
            document_id="doc2",
            title="Azure Functions Overview",
            content="Azure Functions is a serverless compute service.",
            source="functions_guide.pdf",            
            score=0.88,
            page_number=2
        ),
        RetrievedDocument(
            content_id="doc3_ghi789",
            document_id="doc3",
            title="Azure Storage Guide",
            content="Azure Storage provides scalable cloud storage.",
            source="storage_guide.pdf",            
            score=0.82,
            page_number=3
        ),
    ]


class TestExtractCitedDocuments:
    """Tests for _extract_cited_documents method."""
    
    def test_no_citations_in_answer(self, mock_answer_generator, sample_documents):
        """Test when answer contains no citations."""
        answer_text = "This is an answer without any citations."
        
        result = mock_answer_generator._extract_cited_documents(answer_text, sample_documents)
        
        assert result == []
        mock_answer_generator.logger.warning.assert_called_once()
    
    def test_single_citation(self, mock_answer_generator, sample_documents):
        """Test extracting a single citation."""
        answer_text = "Azure Cosmos DB is a database {doc1_abc123}."
        
        result = mock_answer_generator._extract_cited_documents(answer_text, sample_documents)
        
        assert len(result) == 1
        assert result[0].content_id == "doc1_abc123"
        assert result[0].title == "Azure Cosmos DB Guide"
    
    def test_multiple_citations(self, mock_answer_generator, sample_documents):
        """Test extracting multiple citations in order."""
        answer_text = (
            "Azure Cosmos DB is a database {doc1_abc123}. "
            "Azure Functions is serverless {doc2_def456}. "
            "Azure Storage is scalable {doc3_ghi789}."
        )
        
        result = mock_answer_generator._extract_cited_documents(answer_text, sample_documents)
        
        assert len(result) == 3
        assert result[0].content_id == "doc1_abc123"
        assert result[1].content_id == "doc2_def456"
        assert result[2].content_id == "doc3_ghi789"
    
    def test_duplicate_citations_only_counted_once(self, mock_answer_generator, sample_documents):
        """Test that duplicate citations are only counted once."""
        answer_text = (
            "First mention {doc1_abc123}. "
            "Second mention {doc1_abc123}. "
            "Third mention {doc1_abc123}."
        )
        
        result = mock_answer_generator._extract_cited_documents(answer_text, sample_documents)
        
        assert len(result) == 1
        assert result[0].content_id == "doc1_abc123"
    
    def test_unmatched_citation(self, mock_answer_generator, sample_documents):
        """Test when citation does not match any document (LLM hallucination)."""
        answer_text = "This cites a non-existent document {doc999_xyz}."
        
        result = mock_answer_generator._extract_cited_documents(answer_text, sample_documents)
        
        assert result == []
        # Should log warning about unmatched IDs
        assert mock_answer_generator.logger.warning.call_count >= 1
    
    def test_mixed_matched_and_unmatched_citations(self, mock_answer_generator, sample_documents):
        """Test mix of valid and invalid citations."""
        answer_text = (
            "Valid citation {doc1_abc123}. "
            "Invalid citation {doc999_hallucinated}. "
            "Another valid {doc2_def456}."
        )
        
        result = mock_answer_generator._extract_cited_documents(answer_text, sample_documents)
        
        assert len(result) == 2
        assert result[0].content_id == "doc1_abc123"
        assert result[1].content_id == "doc2_def456"
        # Should log warnings for unmatched
        assert mock_answer_generator.logger.warning.call_count >= 1
    
    def test_long_content_ids(self, mock_answer_generator):
        """Test with very long content IDs (like base64 encoded paths)."""
        long_id = "e6e5902f3f2d_aHR0cHM6Ly9zdHJnYWxsZWdpc29jbWthMDAxNWE2MDAuYmxvYi5jb3JlLndpbmRvd3MubmV0L2RvY3VtZW50cy9FbnRlcmluZyUyMENSRyUyMGluZm8lMjBpbnRvJTIwQ29ubmVjdGVkJTIwKHJlbGF0ZWQlMjByZWNvcmRzKV90YWdnZWQucGRm0_normalized_images_7"
        
        documents = [
            RetrievedDocument(
                content_id=long_id,
                document_id="doc1",
                title="Long ID Document",
                content="Content here",
                source="file.pdf",                
                score=0.9,
                page_number=1
            )
        ]
        
        answer_text = f"This cites a long ID {{{long_id}}}."
        
        result = mock_answer_generator._extract_cited_documents(answer_text, documents)
        
        assert len(result) == 1
        assert result[0].content_id == long_id
    
    def test_preserves_citation_order(self, mock_answer_generator, sample_documents):
        """Test that citations are preserved in order of first appearance."""
        answer_text = (
            "Third doc {doc3_ghi789}. "
            "First doc {doc1_abc123}. "
            "Second doc {doc2_def456}."
        )
        
        result = mock_answer_generator._extract_cited_documents(answer_text, sample_documents)
        
        # Should be in order of appearance, not document order
        assert len(result) == 3
        assert result[0].content_id == "doc3_ghi789"
        assert result[1].content_id == "doc1_abc123"
        assert result[2].content_id == "doc2_def456"
    
    def test_empty_documents_list(self, mock_answer_generator):
        """Test when documents list is empty."""
        answer_text = "This has citations {doc1_abc123} but no documents."
        
        result = mock_answer_generator._extract_cited_documents(answer_text, [])
        
        assert result == []
    
    def test_citation_with_special_characters(self, mock_answer_generator):
        """Test citations with special characters in content IDs."""
        special_id = "doc-1_test@2024#section_1"
        
        documents = [
            RetrievedDocument(
                content_id=special_id,
                document_id="doc1",
                title="Special Chars",
                content="Content",
                source="file.pdf",                
                score=0.9,
                page_number=1
            )
        ]
        
        answer_text = f"Citation with special chars {{{special_id}}}."
        
        result = mock_answer_generator._extract_cited_documents(answer_text, documents)
        
        assert len(result) == 1
        assert result[0].content_id == special_id
    
    def test_consecutive_citations(self, mock_answer_generator, sample_documents):
        """Test consecutive citations without text between them."""
        answer_text = "Multiple services {doc1_abc123}{doc2_def456}{doc3_ghi789} available."
        
        result = mock_answer_generator._extract_cited_documents(answer_text, sample_documents)
        
        assert len(result) == 3
        assert result[0].content_id == "doc1_abc123"
        assert result[1].content_id == "doc2_def456"
        assert result[2].content_id == "doc3_ghi789"


class TestReplaceContentWithIndices:
    """Tests for _replace_content_with_indices method."""
    
    def test_single_citation_replacement(self, mock_answer_generator, sample_documents):
        """Test replacing a single content ID with numeric citation."""
        answer_text = "Azure Cosmos DB is a database {doc1_abc123}."
        cited_docs = [sample_documents[0]]  # doc1_abc123
        
        result = mock_answer_generator._replace_content_with_indices(answer_text, cited_docs)
        
        assert result == "Azure Cosmos DB is a database [1]."
    
    def test_multiple_citations_replacement(self, mock_answer_generator, sample_documents):
        """Test replacing multiple content IDs with numeric citations."""
        answer_text = (
            "Azure Cosmos DB {doc1_abc123} is a database. "
            "Azure Functions {doc2_def456} is serverless. "
            "Azure Storage {doc3_ghi789} is scalable."
        )
        cited_docs = sample_documents  # All three docs
        
        result = mock_answer_generator._replace_content_with_indices(answer_text, cited_docs)
        
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result
        assert "{doc1_abc123}" not in result
        assert "{doc2_def456}" not in result
        assert "{doc3_ghi789}" not in result
    
    def test_duplicate_citations_same_number(self, mock_answer_generator, sample_documents):
        """Test that duplicate content IDs get the same citation number."""
        answer_text = "First mention {doc1_abc123}. Second mention {doc1_abc123}."
        cited_docs = [sample_documents[0]]  # doc1_abc123
        
        result = mock_answer_generator._replace_content_with_indices(answer_text, cited_docs)
        
        assert result == "First mention [1]. Second mention [1]."
    
    def test_unmatched_citation_removed(self, mock_answer_generator, sample_documents):
        """Test that unmatched citations are removed."""
        answer_text = "Valid {doc1_abc123} and invalid {doc999_hallucinated} citations."
        cited_docs = [sample_documents[0]]  # Only doc1
        
        result = mock_answer_generator._replace_content_with_indices(answer_text, cited_docs)
        
        assert "[1]" in result
        assert "{doc999_hallucinated}" not in result
        # Should clean up whitespace
        assert "  " not in result
    
    def test_whitespace_cleanup_after_removal(self, mock_answer_generator, sample_documents):
        """Test that whitespace is cleaned up after removing unmatched citations."""
        answer_text = "Text {unmatched}  \nmore text."
        cited_docs = []
        
        result = mock_answer_generator._replace_content_with_indices(answer_text, cited_docs)
        
        assert result == "Text\nmore text."
        assert "  " not in result
    
    def test_consecutive_citations_sorted(self, mock_answer_generator, sample_documents):
        """Test that consecutive citations are sorted numerically."""
        answer_text = "Multiple points {doc3_ghi789}{doc1_abc123}{doc2_def456}."
        cited_docs = sample_documents  # [doc1, doc2, doc3]
        
        result = mock_answer_generator._replace_content_with_indices(answer_text, cited_docs)
        
        # Should sort [3][1][2] -> [1][2][3]
        assert result == "Multiple points [1][2][3]."
    
    def test_non_consecutive_citations_not_sorted(self, mock_answer_generator, sample_documents):
        """Test that non-consecutive citations are not sorted together."""
        answer_text = "First {doc2_def456}. Second {doc1_abc123}."
        cited_docs = sample_documents  # [doc1=1, doc2=2, doc3=3]
        
        result = mock_answer_generator._replace_content_with_indices(answer_text, cited_docs)
        
        # Should remain as [2] and [1] (not sorted across sentences)
        assert result == "First [2]. Second [1]."
    
    def test_empty_cited_docs(self, mock_answer_generator):
        """Test with empty cited_docs list."""
        answer_text = "Text with {doc1_abc123} citation."
        cited_docs = []
        
        result = mock_answer_generator._replace_content_with_indices(answer_text, cited_docs)
        
        assert "{doc1_abc123}" not in result
        assert "Text with citation." == result
    
    def test_no_citations_in_text(self, mock_answer_generator, sample_documents):
        """Test text without any citations."""
        answer_text = "Plain text without citations."
        cited_docs = sample_documents
        
        result = mock_answer_generator._replace_content_with_indices(answer_text, cited_docs)
        
        assert result == "Plain text without citations."
    
    def test_citation_order_by_cited_docs_order(self, mock_answer_generator, sample_documents):
        """Test that citation numbers are assigned based on cited_docs order."""
        answer_text = "Mentions {doc3_ghi789} and {doc1_abc123}."
        # Reverse order: doc3 becomes [1], doc1 becomes [2]
        cited_docs = [sample_documents[2], sample_documents[0]]
        
        result = mock_answer_generator._replace_content_with_indices(answer_text, cited_docs)
        
        assert result == "Mentions [1] and [2]."
    
    def test_long_content_id_replacement(self, mock_answer_generator):
        """Test replacement of very long content IDs."""
        long_id = "e6e5902f3f2d_aHR0cHM6Ly9zdHJnYWxsZWdpc29jbWthMDAxNWE2MDAuYmxvYi5jb3JlLndpbmRvd3MubmV0L2RvY3VtZW50cy9FbnRlcmluZyUyMENSRyUyMGluZm8lMjBpbnRvJTIwQ29ubmVjdGVkJTIwKHJlbGF0ZWQlMjByZWNvcmRzKV90YWdnZWQucGRm0_normalized_images_7"
        
        documents = [
            RetrievedDocument(
                content_id=long_id,
                document_id="doc1",
                title="Long ID",
                content="Content",
                source="file.pdf",                
                score=0.9,
                page_number=1
            )
        ]
        
        answer_text = f"Text with long ID {{{long_id}}} here."
        
        result = mock_answer_generator._replace_content_with_indices(answer_text, documents)
        
        assert result == "Text with long ID [1] here."
    
    def test_mixed_matched_and_unmatched(self, mock_answer_generator, sample_documents):
        """Test mix of matched and unmatched citations."""
        answer_text = "Valid {doc1_abc123} invalid {fake} valid {doc2_def456} invalid {another}."
        cited_docs = [sample_documents[0], sample_documents[1]]
        
        result = mock_answer_generator._replace_content_with_indices(answer_text, cited_docs)
        
        assert "[1]" in result
        assert "[2]" in result
        assert "{fake}" not in result
        assert "{another}" not in result
    
    def test_space_before_period_cleanup(self, mock_answer_generator):
        """Test whitespace handling when citation is removed before period."""
        answer_text = "Sentence {unmatched} ."
        cited_docs = []
        
        result = mock_answer_generator._replace_content_with_indices(answer_text, cited_docs)
        
        # The method removes {unmatched} and cleans up double spaces to single space
        # It doesn't specifically remove space before period, so result will be "Sentence ."
        assert "{unmatched}" not in result
        assert "Sentence ." == result
    
    def test_multiple_consecutive_groups_sorted_separately(self, mock_answer_generator, sample_documents):
        """Test that multiple groups of consecutive citations are sorted separately."""
        answer_text = "First group {doc3_ghi789}{doc1_abc123}. Second group {doc2_def456}{doc3_ghi789}."
        cited_docs = sample_documents
        
        result = mock_answer_generator._replace_content_with_indices(answer_text, cited_docs)
        
        # First group: [3][1] -> [1][3]
        # Second group: [2][3] -> [2][3] (already sorted)
        assert "First group [1][3]." in result
        assert "Second group [2][3]." in result


class TestSortConsecutiveCitations:
    """Tests for _sort_consecutive_citations method."""
    
    def test_two_citations_unsorted(self, mock_answer_generator):
        """Test sorting two consecutive citations."""
        answer_text = "Azure services [2][1] are great."
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        assert result == "Azure services [1][2] are great."
    
    def test_three_citations_unsorted(self, mock_answer_generator):
        """Test sorting three consecutive citations."""
        answer_text = "Features [3][1][2] available."
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        assert result == "Features [1][2][3] available."
    
    def test_reverse_order_citations(self, mock_answer_generator):
        """Test sorting citations in complete reverse order."""
        answer_text = "Points [5][4][3][2][1] mentioned."
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        assert result == "Points [1][2][3][4][5] mentioned."
    
    def test_already_sorted_citations(self, mock_answer_generator):
        """Test that already sorted citations remain unchanged."""
        answer_text = "Services [1][2][3] available."
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        assert result == "Services [1][2][3] available."
    
    def test_single_citation_unchanged(self, mock_answer_generator):
        """Test that single citation is not modified."""
        answer_text = "Azure Cosmos DB [1] is great."
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        assert result == "Azure Cosmos DB [1] is great."
    
    def test_non_consecutive_citations_not_sorted(self, mock_answer_generator):
        """Test that non-consecutive citations are not sorted together."""
        answer_text = "First point [3]. Second point [1]. Third point [2]."
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        # Should remain unchanged since citations are not consecutive
        assert result == "First point [3]. Second point [1]. Third point [2]."
    
    def test_multiple_groups_sorted_independently(self, mock_answer_generator):
        """Test multiple groups of consecutive citations sorted independently."""
        answer_text = "Group A [3][1][2] and Group B [4][2] available."
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        assert result == "Group A [1][2][3] and Group B [2][4] available."
    
    def test_no_citations(self, mock_answer_generator):
        """Test text without citations remains unchanged."""
        answer_text = "This is plain text without citations."
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        assert result == "This is plain text without citations."
    
    def test_citations_with_text_between_not_sorted(self, mock_answer_generator):
        """Test that citations with text between them are not sorted together."""
        answer_text = "Point [3] and point [1] mentioned."
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        # Should not sort because there's text between them
        assert result == "Point [3] and point [1] mentioned."
    
    def test_citations_with_only_space_between(self, mock_answer_generator):
        """Test citations separated by space are not considered consecutive."""
        answer_text = "Citations [2] [1] here."
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        # Space between citations means they're not consecutive
        assert result == "Citations [2] [1] here."
    
    def test_partially_sorted_group(self, mock_answer_generator):
        """Test sorting a partially sorted group."""
        answer_text = "Items [1][3][2][4] listed."
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        assert result == "Items [1][2][3][4] listed."
    
    def test_duplicate_numbers_in_consecutive_citations(self, mock_answer_generator):
        """Test sorting consecutive citations with duplicate numbers."""
        answer_text = "References [3][1][2][1] found."
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        # Should sort including duplicates
        assert result == "References [1][1][2][3] found."
    
    def test_large_citation_numbers(self, mock_answer_generator):
        """Test sorting with large citation numbers."""
        answer_text = "Many sources [100][25][50] used."
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        assert result == "Many sources [25][50][100] used."
    
    def test_two_digit_and_single_digit_mix(self, mock_answer_generator):
        """Test sorting mix of single and double-digit citations."""
        answer_text = "Sources [12][3][7][1] referenced."
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        assert result == "Sources [1][3][7][12] referenced."
    
    def test_at_start_of_text(self, mock_answer_generator):
        """Test consecutive citations at the start of text."""
        answer_text = "[3][1][2] These are the main points."
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        assert result == "[1][2][3] These are the main points."
    
    def test_at_end_of_text(self, mock_answer_generator):
        """Test consecutive citations at the end of text."""
        answer_text = "The main sources are [4][2][1]"
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        assert result == "The main sources are [1][2][4]"
    
    def test_entire_text_is_citations(self, mock_answer_generator):
        """Test when entire text is just citations."""
        answer_text = "[5][3][1][4][2]"
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        assert result == "[1][2][3][4][5]"
    
    def test_complex_multi_group_scenario(self, mock_answer_generator):
        """Test complex scenario with multiple groups in various states."""
        answer_text = (
            "First [3][1] point. "
            "Second [2] alone. "
            "Third [5][4][2][1][3] many. "
            "Fourth [1] single."
        )
        
        result = mock_answer_generator._sort_consecutive_citations(answer_text)
        
        assert result == (
            "First [1][3] point. "
            "Second [2] alone. "
            "Third [1][2][3][4][5] many. "
            "Fourth [1] single."
        )
