"""
Query Rewriter Agent for multi-strategy query expansion and rewriting.
Uses MAF ChatAgent for LLM interactions.
"""

from typing import List, Dict, Any, Optional
from azure.identity import DefaultAzureCredential

from agent_framework import ChatAgent, ChatMessage, Role
from agent_framework.azure import AzureOpenAIChatClient
from opentelemetry import trace

from app.core.settings import Settings
from app.core.logger import Logger
from app.prompts import QueryRewriterPrompts
from app.models.chat import RewrittenQuery


class QueryRewriter:
    """
    Agent for HyDE (Hypothetical Document Embedding) query generation.
    
    Generates hypothetical document passages for semantic search, creating
    search queries that represent what the answer content might look like
    rather than traditional keyword-based queries.
    """
    
    def __init__(self, settings: Settings, logger: Logger):
        """
        Initialize the query rewriter agent using MAF ChatAgent.
        
        Args:
            settings: Application settings with Azure AI configuration
            logger: Injected logging service
        """
        self.settings = settings
        self.logger = logger
        self.tracer = trace.get_tracer("QueryRewriterAgent")
        
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
            name="QueryRewriterAgent",
            instructions=QueryRewriterPrompts.HYDE_SYSTEM_PROMPT
        )
        
        self.logger.info(f"QueryRewriter initialized with MAF ChatAgent: {settings.azure_openai.deployment_name}")
    
    async def generate_hyde_search_query(
        self,
        user_query: str,
        search_history: Optional[List[Dict[str, Any]]] = None,
        previous_reviews: Optional[List[str]] = None
    ) -> str:
        """
        Generate HyDE (Hypothetical Document Embedding) search query.
        Creates a hypothetical paragraph of what the answer/content might look like.
        For subsequent searches, diversifies strategy based on previous attempts.
        
        Args:
            user_query: Original user question
            search_history: List of previous search attempts with queries
            previous_reviews: List of review feedback from previous searches
        
        Returns:
            HyDE search query text
        """
        self.logger.info(f"Generating HyDE search query for: {user_query[:100]}...")
        
        # Build context for prompt
        context_parts = [f"User Question: {user_query}"]
        
        # Add search history for subsequent attempts
        if search_history and previous_reviews:
            context_parts.append("\\n\\n### Previous Search Attempts ###")
            for i, (search, review) in enumerate(zip(search_history, previous_reviews), 1):
                context_parts.append(f"\\n<Attempt {i}>")
                context_parts.append(f"Query: {search.get('query', '')}")
                context_parts.append(f"Review: {review}")
                context_parts.append("</Attempt>")
            
            context_parts.append("\\n\\nCRITICAL: Since this is NOT the first search, you MUST diversify your approach:")
            context_parts.append("- Use different terminology, synonyms, or technical vs. layman terms")
            context_parts.append("- Focus on different aspects, time periods, or perspectives")
            context_parts.append("- Explore related concepts, causes, effects, or stakeholder viewpoints")
        
        context_parts.append("\\n\\nGenerate a hypothetical paragraph of what you expect to find in the target documents.")
        context_parts.append("Make it sound like the actual content, NOT like a search query.")
        
        try:
            context_text = '\n'.join(context_parts)
            user_prompt = f"\n{context_text}"
            
            rewritten_query = await self._call_llm(user_prompt)
            
            self.logger.info(f"Generated HyDE query: {rewritten_query.hypothetical_passage[:150]}...")
            self.logger.info(f"Reasoning: {rewritten_query.reasoning}")
            
            return rewritten_query.hypothetical_passage
            
        except Exception as e:
            self.logger.error(f"HyDE generation failed: {e}")
            # Fallback to original query
            return user_query
    
    async def _call_llm(self, user_prompt: str) -> RewrittenQuery:
        """
        Call LLM for query rewriting using MAF ChatAgent.
        
        Args:
            user_prompt: User message for query rewriting
        
        Returns:
            RewrittenQuery model with hypothetical_passage and reasoning
        """
        # Create MAF ChatMessage
        message = ChatMessage(role=Role.USER, text=user_prompt)
        
        # Run agent with JSON mode for structured output
        result = await self.agent.run(
            messages=[message],
            response_format=RewrittenQuery,
            max_tokens=500,
            temperature=0.3  # Slightly creative for rewrites
        )
        
        # Parse the structured response
        return RewrittenQuery.model_validate_json(result.messages[-1].text)
