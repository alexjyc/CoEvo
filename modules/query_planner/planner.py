"""
Query Planner Module

Handles query decomposition or reformulation for improved retrieval.
Uses a frozen LLM with an optimizable prompt.

Optimization Target: Query planning prompt
Evaluation Metrics: context_precision, context_recall (via retrieval results)
"""

import os
from pathlib import Path
from typing import Optional, Union, List

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from modules.base import (
    Module,
    ModuleType,
    QueryPlannerInput,
    QueryPlannerOutput,
)


class QueryPlannerResponse(BaseModel):
    """Structured response from query planner LLM"""
    mode: str = Field(
        description="The mode of the query planner. Either 'decomposition' or 'reformulation'."
    )
    query: Union[str, List[str]] = Field(
        description="The query to be planned. If decomposition, list of sub-queries. If reformulation, single query."
    )


class QueryPlannerModule(Module[QueryPlannerInput, QueryPlannerOutput]):
    """
    Query Planner Module for RAG Pipeline

    This module decides whether to decompose a complex query into sub-queries
    or reformulate a simple query for better retrieval.

    Independence: Can be evaluated standalone with ground truth queries
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        prompt_path: Optional[Path] = None,
    ):
        super().__init__(ModuleType.QUERY_PLANNER)
        self.model_name = model_name

        # Initialize LLM (frozen - only prompt changes)
        if model_name in ["gpt-4o-mini", "gpt-4o", "gpt-5-nano"]:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        elif model_name == "gemini-2.5-flash":
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                google_api_key=os.getenv("GEMINI_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Load prompt
        if prompt_path and prompt_path.exists():
            self.load_prompt(prompt_path)
        else:
            self._prompt = self.get_default_prompt()

    def get_default_prompt(self) -> str:
        """Return the seed prompt for query planning"""
        seed_path = Path(__file__).parent / "prompts" / "seed.txt"
        
        if seed_path.exists():
            return seed_path.read_text()

        # Fallback if seed.txt doesn't exist
        return """You are an expert in information-retrieval query handling.

Your task is to decide whether to:
- Decompose the query into sub-queries (for complex multi-intent queries)
- Reformulate the query (for simple but vague queries)

Decision Logic:
- Multiple intents/constraints → Decomposition
- Single intent but vague → Reformulation

Provide mode (decomposition/reformulation) and query/queries:"""

    async def run(self, input: QueryPlannerInput) -> QueryPlannerOutput:
        """
        Execute query planning.

        Args:
            input: QueryPlannerInput with query and optional feedback

        Returns:
            QueryPlannerOutput with mode and processed queries
        """
        try:
            prompt = self._prompt

            # Get structured output from LLM
            structured_llm = self.llm.with_structured_output(QueryPlannerResponse)

            response: QueryPlannerResponse = await structured_llm.ainvoke([
                {"role": "system", "content": prompt},
                {"role": "user", "content": input.query}
            ])

            # Normalize output to list of queries
            if isinstance(response.query, str):
                queries = [response.query]
            else:
                queries = response.query

            return QueryPlannerOutput(
                status="success",
                mode=response.mode,
                queries=queries,
                original_query=input.query,
                metadata={
                    "model": self.model_name,
                    "num_queries": len(queries),
                }
            )

        except Exception as e:
            return QueryPlannerOutput(
                status="error",
                error_message=str(e),
                mode="reformulation",
                queries=[input.query],  # Fallback to original query
                original_query=input.query,
            )

    def __repr__(self) -> str:
        return f"QueryPlannerModule(model={self.model_name})"
