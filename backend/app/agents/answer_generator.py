"""
Answer Generator Agent for synthesizing answers with citations.
Uses MAF ChatAgent for LLM interactions.
"""
import re
from typing import List, Dict, Optional
from azure.identity import DefaultAzureCredential

from agent_framework import ChatAgent, ChatMessage, Role
from agent_framework.azure import AzureOpenAIChatClient
from opentelemetry import trace

from app.core.settings import Settings
from app.core.logger import Logger
from app.models import (
    RetrievedDocument,
    GeneratedAnswer
)
from app.prompts.templates import AnswerGeneratorPrompts
from app.utils.citation_tracker import CitationTracker


class AnswerGenerator:
    """
    Agent for synthesizing answers with proper citation handling.
    Uses MAF ChatAgent for LLM interactions following framework patterns.
    
    Responsibilities:
    - Assemble context from retrieved documents
    - Generate comprehensive answers grounded in sources
    - Insert citations linking claims to sources
    - Handle cases with insufficient information
    - Control answer length based on query complexity
    """
    
    # Compiled regex patterns for efficiency
    _CITATION_PATTERN = re.compile(r"\{([^}]+)\}")
    _CITATION_NUMBER_PATTERN = re.compile(r'\[(\d+)\]')
    _CONSECUTIVE_CITATIONS_PATTERN = re.compile(r'(?:\[\d+\]){2,}')
    _REF_TOKEN_PATTERN = re.compile(r'^\s*ref[-_\s]?(\d+)\s*$', re.IGNORECASE)
    
    def __init__(self, settings: Settings, logger: Logger, citation_tracker: CitationTracker):
        """
        Initialize the answer generator agent using MAF ChatAgent.
        
        Args:
            settings: Application settings with Azure AI configuration
            logger: Injected logging service
            citation_tracker: Citation tracking utility for source attribution
        """
        self.settings = settings
        self.logger = logger
        self.citation_tracker = citation_tracker
        self.tracer = trace.get_tracer("AnswerGeneratorAgent")
        
        # Initialize MAF ChatAgent
        credential = DefaultAzureCredential() if settings.use_managed_identity else None
        
        chat_client = AzureOpenAIChatClient(
            credential=credential,
            endpoint=settings.azure_openai.endpoint,
            api_version=settings.azure_openai.api_version,
            deployment_name=settings.azure_openai.deployment_name
        )
        
        self.agent = ChatAgent(
            chat_client=chat_client,
            name="AnswerGeneratorAgent",
            instructions=AnswerGeneratorPrompts.ANSWER_GENERATOR_SYSTEM_PROMPT
        )
        
        self.logger.info(f"AnswerGenerator initialized with MAF ChatAgent: {settings.azure_openai.deployment_name}")
    
    async def generate_answer(
        self,
        query: str,
        documents: List[RetrievedDocument],
        generated_answer_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> GeneratedAnswer:
        """
        Generate answer from retrieved documents with citations using MAF ChatAgent.                
        
        Args:
            query: User's original query
            documents: Retrieved documents to use as sources
            generated_answer_prompt: Prompt for the answer generation
            conversation_history: Optional conversation history for context
        
        Returns:
            GeneratedAnswer with text, citations, and metadata
        """
        try:
            # Log entry point to track duplicate calls
            self.logger.info(f"[GENERATE_ANSWER] Called with query: '{query[:50]}...', {len(documents)} documents")
            
            # Add custom attributes to current MAF span
            self.logger.add_span_attributes(
                operation="answer_generation",
                query_length=len(query),
                document_count=len(documents),
                agent_name="AnswerGeneratorAgent"
            )
            
            # Handle case with no documents
            if not documents:
                return self._generate_fallback_answer(query)
            
            # Generate answer using MAF ChatAgent
            response = await self._call_llm(
                generated_answer_prompt=generated_answer_prompt, 
                conversation_history=conversation_history
            )
            
            # Extract which documents were actually cited
            cited_docs = self._extract_cited_documents(response, documents)

            # Safety net: if LLM produced no usable citations, retry once with a
            # stricter reminder so that the final answer always carries citations.
            if not cited_docs:
                self.logger.warning(
                    "[Citation] No valid citations in initial response - retrying with stricter prompt"
                )
                retry_prompt = (
                    generated_answer_prompt
                    + "\n\n=== CITATION ENFORCEMENT (RETRY) ===\n"
                    "Your previous response was missing citations. "
                    "You MUST cite EVERY factual sentence by appending the short Citation Token "
                    "for the supporting Vetted Result in curly braces immediately after the "
                    "sentence-ending period. The tokens are listed above as `ref-1`, `ref-2`, etc. "
                    "Cite by writing exactly `{ref-1}`, `{ref-2}`, ... — do NOT invent other "
                    "tokens. Sentences without a citation will be discarded."
                )
                retry_response = await self._call_llm(
                    generated_answer_prompt=retry_prompt,
                    conversation_history=conversation_history
                )
                retry_cited_docs = self._extract_cited_documents(retry_response, documents)
                if retry_cited_docs:
                    response = retry_response
                    cited_docs = retry_cited_docs

            # Replace {Content Id} with [1], [2], [3] and sort consecutive citations
            final_answer = self._replace_content_with_indices(response, cited_docs, documents)
            
            # Create citations only for documents that were cited
            citations = self.citation_tracker.create_citations(cited_docs)
            
            generated_answer = GeneratedAnswer(
                answer_text=final_answer,
                citations=citations,
                metadata={
                    "document_count": len(documents),
                    "cited_count": len(cited_docs)
                }
            )
                                    
            return generated_answer
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}", exc_info=True)
            return self._generate_fallback_answer(query, error=str(e))
        
    
    def _generate_fallback_answer(
        self,
        query: str,
        error: Optional[str] = None
    ) -> GeneratedAnswer:
        """
        Generate fallback answer when documents unavailable.
        
        Args:
            query: User query
            error: Optional error message
        
        Returns:
            Fallback GeneratedAnswer
        """
        if error:
            answer_text = (
                f"I apologize, but I encountered an error while generating an answer: {error}. "
                "Please try rephrasing your question or try again later."
            )
        else:
            answer_text = (
                "I don't have enough information in the available documents to answer your question. "
                "The search didn't return any relevant documents. "
                "Please try rephrasing your question, adjusting your filters, or providing more context."
            )
                        
        self.logger.warning(f"Returning fallback answer for query: {query[:100]}")
        
        return GeneratedAnswer(
            answer_text=answer_text,
            citations=[],
            metadata={
                "fallback": True,
                "error": error
            }
        )
    
    async def _call_llm(self, generated_answer_prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None, max_tokens: int = 8000) -> str:
        """Call LLM using MAF ChatAgent with generated answer prompt.
        
        The system prompt (closed-book rules, citation rules, etc.) is set at
        agent initialization. The generated_answer_prompt contains the user question,
        vetted results, and reflection analysis as a single user message.
        
        Args:
            generated_answer_prompt: Combined user question, vetted results, and reflection analysis
            conversation_history: Optional conversation history for additional context
            max_tokens: Maximum tokens for response
        
        Returns:
            LLM response text
        """
        self.logger.info(f"[CALL_LLM] Invoking MAF ChatAgent with {max_tokens} max tokens")
        
        # Build messages list starting with conversation history
        messages = []
        
        # Add conversation history if provided (citations already stripped at save time)
        if conversation_history:
            for msg in conversation_history:
                role = Role.USER if msg.get("role") == "user" else Role.ASSISTANT
                messages.append(ChatMessage(role=role, text=msg.get("content", "")))
        
        # Add generated answer prompt (includes user question + vetted results) as user message
        messages.append(ChatMessage(role=Role.USER, text=generated_answer_prompt))
        
        # Run agent with constructed messages
        result = await self.agent.run(
            messages=messages,
            temperature=0.1,  # Low temperature for factual answers
            max_tokens=max_tokens
        )
        
        return result.messages[-1].text
    
    def _resolve_cited_id(
        self,
        cited_content_id: str,
        content_id_map: Dict[str, RetrievedDocument],
        documents: Optional[List[RetrievedDocument]] = None,
    ) -> Optional[RetrievedDocument]:
        """
        Resolve a cited token to a real document.

        Supports three forms (in priority order):
        1. Short ref tokens like "ref-1", "ref_2", "ref 3" → maps to documents[N-1].
        2. Exact match against the full content_id.
        3. Tolerant substring/overlap match for slightly mangled content_ids.
        Returns None if no plausible match is found.
        """
        # 1. Short ref token (preferred new format). Use a simple, regex-free
        # check so we never depend on regex flag handling. Accepts forms like
        # "ref-1", "ref_2", "ref 3", "REF1", with optional surrounding spaces
        # and optional surrounding braces (in case the LLM nested them).
        if cited_content_id and documents:
            stripped = cited_content_id.strip().strip("{}").strip().lower()
            if stripped.startswith("ref"):
                tail = stripped[3:].lstrip("-_ ").strip()
                if tail.isdigit():
                    idx = int(tail) - 1
                    self.logger.info(
                        f"[Citation] ref-token '{cited_content_id}' -> index {idx} "
                        f"(documents available: {len(documents)})"
                    )
                    if 0 <= idx < len(documents):
                        return documents[idx]
                    return None

        # 2. Exact content_id match.
        if cited_content_id in content_id_map:
            return content_id_map[cited_content_id]

        # 3. Tolerant fallback for mangled content_ids.
        best_doc: Optional[RetrievedDocument] = None
        best_overlap = 0
        min_overlap = max(12, len(cited_content_id) // 3)

        for real_id, doc in content_id_map.items():
            if cited_content_id in real_id or real_id in cited_content_id:
                overlap = min(len(cited_content_id), len(real_id))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_doc = doc

        if best_doc is not None and best_overlap >= min_overlap:
            return best_doc
        return None

    def _extract_cited_documents(self, answer_text: str, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Extract which documents were actually cited in the answer.
        
        Parses {Content ID} patterns and matches to documents.
        
        Args:
            answer_text: Generated answer with {Content ID} citations
            documents: All documents
        
        Returns:
            List of documents that were cited
        """
        # Extract all {Content ID} patterns using pre-compiled regex
        cited_content_ids = self._CITATION_PATTERN.findall(answer_text)
        
        if not cited_content_ids:
            self.logger.warning("[Citation] No {Content ID} citations found in answer")
            return []
        
        # Create content ID to document mapping for fast lookup
        content_id_map = {doc.content_id: doc for doc in documents}
        
        # Get unique cited IDs preserving order
        unique_cited_ids = list(dict.fromkeys(cited_content_ids))
        
        # Log extracted content IDs (truncated for readability)
        self.logger.info(f"[Citation] Extracted {len(unique_cited_ids)} unique content IDs from answer:")
        for i, cid in enumerate(unique_cited_ids, 1):
            display_id = cid if len(cid) <= 50 else f"{cid[:25]}...{cid[-25:]}"
            self.logger.info(f"  [{i}] {display_id}")
        
        # Log available document content IDs
        self.logger.info(f"[Citation] Available documents: {len(documents)}")
        for i, doc in enumerate(documents, 1):
            display_id = doc.content_id if len(doc.content_id) <= 50 else f"{doc.content_id[:25]}...{doc.content_id[-25:]}"
            self.logger.info(f"  Doc[{i}] content_id: {display_id}")
        
        # Match cited content IDs to documents (in order of first appearance).
        # Use the tolerant resolver so that minor LLM mistakes in copying the
        # long content_id still attach a citation instead of silently dropping it.
        cited_docs = []
        seen_doc_keys = set()
        unmatched_ids = set()

        for cited_content_id in cited_content_ids:
            doc = self._resolve_cited_id(cited_content_id, content_id_map, documents)
            if doc is None:
                unmatched_ids.add(cited_content_id)
                continue
            if doc.content_id in seen_doc_keys:
                continue
            cited_docs.append(doc)
            seen_doc_keys.add(doc.content_id)
        
        # Log unmatched content IDs
        if unmatched_ids:
            self.logger.warning(f"[Citation] {len(unmatched_ids)} content IDs could not be matched to documents:")
            for cid in unmatched_ids:
                display_id = cid if len(cid) <= 50 else f"{cid[:25]}...{cid[-25:]}"
                self.logger.warning(f"  UNMATCHED: {display_id}")
        
        self.logger.info(f"[Citation] Found {len(cited_docs)} cited documents from {len(cited_content_ids)} citations")
        return cited_docs
    
    def _replace_content_with_indices(
        self,
        answer_text: str,
        cited_docs: List[RetrievedDocument],
        all_documents: Optional[List[RetrievedDocument]] = None,
    ) -> str:
        """
        Replace {Content ID} patterns with [n] citation indices and sort consecutive citations.
        
        Args:
            answer_text: Answer text with {Content ID} patterns
            cited_docs: List of cited documents in order (determines citation numbering)
            all_documents: Full set of retrieved documents, used to tolerantly
                resolve {Content ID} markers whose IDs don't match exactly.
        
        Returns:
            Answer text with sorted [1], [2], [3] citations
        """
        # Create content ID to index mapping (1-based)
        content_id_to_index = {doc.content_id: i + 1 for i, doc in enumerate(cited_docs)}
        # Build a tolerant lookup over ALL retrieved documents so we can rescue
        # citations whose content_id was slightly mangled by the LLM.
        full_id_map: Dict[str, RetrievedDocument] = {
            d.content_id: d for d in (all_documents or cited_docs)
        }
        
        self.logger.info(f"[Citation] Building index mapping for {len(cited_docs)} cited documents")
        
        # Track replacement stats
        replacements_made = 0
        replacements_failed = set()  # Use set for O(1) operations
        
        def replace_content_id(match):
            nonlocal replacements_made
            cited_content_id = match.group(1)

            # Exact match against the cited-doc index map.
            if cited_content_id in content_id_to_index:
                replacements_made += 1
                return f"[{content_id_to_index[cited_content_id]}]"

            # Tolerant resolution: map a mangled ID to a real document.
            resolved = self._resolve_cited_id(cited_content_id, full_id_map, all_documents)
            if resolved is not None:
                # Ensure the resolved doc has an index; if not, append it so
                # the citation can still be rendered to the user.
                if resolved.content_id not in content_id_to_index:
                    cited_docs.append(resolved)
                    content_id_to_index[resolved.content_id] = len(cited_docs)
                replacements_made += 1
                return f"[{content_id_to_index[resolved.content_id]}]"

            replacements_failed.add(cited_content_id)
            return ""
        
        # Replace all {Content ID} patterns with [n] (or remove if unmatched) using pre-compiled regex
        result = self._CITATION_PATTERN.sub(replace_content_id, answer_text)
        
        # Clean up extra whitespace from removed citations
        result = re.sub(r' {2,}', ' ', result)    # Multiple spaces → single space
        result = re.sub(r' \n', '\n', result)     # Space before newline → just newline
        result = re.sub(r'\n ', '\n', result)     # Space after newline → just newline
        
        # Log replacement summary
        self.logger.info(f"[Citation] Replacements: {replacements_made} successful, {len(replacements_failed)} failed")
        if replacements_failed:
            self.logger.warning(f"[Citation] Removed {len(replacements_failed)} unmatched citations (LLM hallucination - cited non-existent content IDs)")
            # Log only first few unmatched IDs to avoid log spam
            for cid in list(replacements_failed)[:5]:
                display_id = cid if len(cid) <= 50 else f"{cid[:25]}...{cid[-25:]}"
                self.logger.warning(f"  UNMATCHED: {display_id}")
            if len(replacements_failed) > 5:
                self.logger.warning(f"  ... and {len(replacements_failed) - 5} more")
        
        # Sort consecutive citations (e.g., [2][1] becomes [1][2])
        return self._sort_consecutive_citations(result)
    
    def _sort_consecutive_citations(self, answer_text: str) -> str:
        """
        Sort consecutive citation numbers to ensure proper ordering.
        
        Converts [2][1] to [1][2], [3][1][2] to [1][2][3], etc.
        
        Args:
            answer_text: Answer text with [n] citations
        
        Returns:
            Answer text with sorted consecutive citations
        """
        def sort_citations(match):
            # Extract all citation numbers from consecutive [n][n][n] pattern using pre-compiled regex
            citation_numbers = self._CITATION_NUMBER_PATTERN.findall(match.group(0))
            
            # Sort numerically and reconstruct as [1][2][3]
            return ''.join(f'[{n}]' for n in sorted(map(int, citation_numbers)))
        
        # Match consecutive citations using pre-compiled regex
        return self._CONSECUTIVE_CITATIONS_PATTERN.sub(sort_citations, answer_text)
