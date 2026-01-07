"""
Reranker Module

Listwise reranking of retrieved documents using a frozen LLM with optimizable prompt.

Optimization Target: Reranking prompt
Evaluation Metrics: context_relevancy, answer_relevancy (post-rerank)
"""

import os
from pathlib import Path
from typing import Optional, List

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from modules.base import (
    Module,
    ModuleType,
    RerankerInput,
    RerankerOutput,
)


class DocumentRerankingResponse(BaseModel):
    """Structured response from reranker LLM"""
    ranked_documents: List[str] = Field(
        description="List of documents ranked by relevance, most relevant first"
    )


class RerankerModule(Module[RerankerInput, RerankerOutput]):
    """
    Reranker Module for RAG Pipeline

    This module takes retrieved documents and reranks them by relevance
    to the query using an LLM with a listwise approach.

    Independence: Can be evaluated with fixed retrieved documents
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        prompt_path: Optional[Path] = None,
    ):
        super().__init__(ModuleType.RERANKER)
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
        """Return the seed prompt for reranking"""
        seed_path = Path(__file__).parent / "prompts" / "seed.txt"

        if seed_path.exists():
            return seed_path.read_text()

        # Fallback if seed.txt doesn't exist
        return """You are an expert at evaluating document relevance.
Your task is to RERANK ALL {num_documents} documents by relevance to the query.

Return ALL documents in order from most relevant to least relevant.
Return ONLY the complete document texts, not numbers.

Ranking Criteria:
- Direct answers get highest priority
- Comprehensive explanations rank second
- Supporting context ranks third
- Off-topic content ranks lowest

Query: {query}

Rerank these documents:"""

    async def run(self, input: RerankerInput) -> RerankerOutput:
        """
        Execute listwise reranking.

        Args:
            input: RerankerInput with query, documents, and optional feedback

        Returns:
            RerankerOutput with reranked documents
        """
        try:
            if not input.documents:
                return RerankerOutput(
                    status="success",
                    ranked_documents=[],
                    scores=[],
                    metadata={"num_documents": 0}
                )

            # Format prompt
            prompt = self._prompt.format(
                query=input.query,
                num_documents=len(input.documents)
            )

            # Format documents for input
            numbered_docs = [f"[{i+1}] {doc}" for i, doc in enumerate(input.documents)]
            documents_text = "\n\n".join(numbered_docs)

            # Get structured output from LLM
            structured_llm = self.llm.with_structured_output(DocumentRerankingResponse)

            response: DocumentRerankingResponse = await structured_llm.ainvoke([
                {"role": "system", "content": prompt},
                {"role": "user", "content": documents_text}
            ])

            # Generate scores based on ranking position (1.0 for first, decreasing)
            num_docs = len(response.ranked_documents)
            scores = [1.0 - (i / num_docs) for i in range(num_docs)] if num_docs > 0 else []

            return RerankerOutput(
                status="success",
                ranked_documents=response.ranked_documents,
                scores=scores,
                metadata={
                    "model": self.model_name,
                    "num_input_documents": len(input.documents),
                    "num_output_documents": len(response.ranked_documents),
                }
            )

        except Exception as e:
            # On error, return original order
            return RerankerOutput(
                status="error",
                error_message=str(e),
                ranked_documents=input.documents,
                scores=[1.0 / (i + 1) for i in range(len(input.documents))],
            )

    def __repr__(self) -> str:
        return f"RerankerModule(model={self.model_name})"
