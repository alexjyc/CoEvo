"""
Generator Module

Answer generation using a frozen LLM with optimizable prompt.

Optimization Target: Generation prompt
Evaluation Metrics: faithfulness, answer_relevancy, answer_correctness
"""

import os
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from modules.base import (
    Module,
    ModuleType,
    GeneratorInput,
    GeneratorOutput,
)


class AnswerGenerationResponse(BaseModel):
    """Structured response from generator LLM"""
    answer: str = Field(
        description="The generated answer to the query"
    )
    reference: str = Field(
        description="Evidence from the context that supports the answer"
    )
    rationale: str = Field(
        description="Explanation of how the answer was generated"
    )


class GeneratorModule(Module[GeneratorInput, GeneratorOutput]):
    """
    Generator Module for RAG Pipeline

    This module generates answers based on the query and reranked context
    using an LLM with an optimizable prompt.

    Independence: Can be evaluated with fixed context
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        prompt_path: Optional[Path] = None,
    ):
        super().__init__(ModuleType.GENERATOR)
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
        """Return the seed prompt for generation"""
        seed_path = Path(__file__).parent / "prompts" / "seed.txt"

        if seed_path.exists():
            return seed_path.read_text()

        # Fallback if seed.txt doesn't exist
        return """You are an AI assistant providing expert-level answers.
Generate accurate, comprehensive responses based on the provided context.

Guidelines:
- Base your answer on the provided context
- Include specific details when available
- Acknowledge limitations if context is insufficient

Context: {context}

Question: {query}

Provide a thorough answer:"""

    async def run(self, input: GeneratorInput) -> GeneratorOutput:
        """
        Execute answer generation.

        Args:
            input: GeneratorInput with query, context, and optional feedback

        Returns:
            GeneratorOutput with answer, reference, and rationale
        """
        try:
            # Format prompt
            prompt = self._prompt.format(
                context=input.context,
                query=input.query
            )

            # Get structured output from LLM
            structured_llm = self.llm.with_structured_output(AnswerGenerationResponse)

            response: AnswerGenerationResponse = await structured_llm.ainvoke([
                {"role": "system", "content": prompt},
                {"role": "user", "content": input.query}
            ])

            return GeneratorOutput(
                status="success",
                answer=response.answer,
                reference=response.reference,
                rationale=response.rationale,
                metadata={
                    "model": self.model_name,
                    "answer_length": len(response.answer),
                    "context_length": len(input.context),
                }
            )

        except Exception as e:
            return GeneratorOutput(
                status="error",
                error_message=str(e),
                answer="",
                reference="",
                rationale="",
            )

    def __repr__(self) -> str:
        return f"GeneratorModule(model={self.model_name})"
