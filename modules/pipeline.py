"""
Modular RAG Pipeline

Orchestrates all modules in a clean, composable way.
Each module is independently testable and optimizable.
"""

from dataclasses import dataclass
from typing import Any

from modules.base import (
    GeneratorInput,
    PipelineConfig,
    QueryPlannerInput,
    RerankerInput,
    RetrievalInput,
)
from modules.evaluation import RAGASEvaluator
from modules.generator import GeneratorModule
from modules.query_planner import HybridRetriever, QueryPlannerModule
from modules.reranker import RerankerModule


@dataclass
class PipelineResult:
    """Result from RAG pipeline execution"""

    answer: str
    rationale: str
    contexts: list[str]
    ranked_documents: list[str]
    metrics: dict[str, float]
    metadata: dict[str, Any]


class ModularRAGPipeline:
    """
    Modular RAG Pipeline with independent, optimizable components.

    Architecture:
        Query → [Module 1: Query Planner + Retrieval] → Documents
             ↓
             → [Module 2: Reranker] → Refined Documents
             ↓
             → [Module 3: Generator] → Answer

    Each module:
    - Is independently testable
    - Has an optimizable prompt (via GEPA)
    - Is evaluated with RAGAS metrics
    """

    def __init__(
        self,
        preprocessor,  # DocumentPreprocessor instance
        config: PipelineConfig | None = None,
    ):
        self.config = config or PipelineConfig()
        self.preprocessor = preprocessor

        # Initialize modules
        self.query_planner = QueryPlannerModule(
            model_name=self.config.llm_model,
            prompt_path=self.config.prompt_dir / "query_planner" / "prompts" / "current.txt",
        )

        self.retriever = HybridRetriever(
            preprocessor=preprocessor,
            default_dense_weight=0.5,
        )

        self.reranker = RerankerModule(
            model_name=self.config.llm_model,
            prompt_path=self.config.prompt_dir / "reranker" / "prompts" / "current.txt",
        )

        self.generator = GeneratorModule(
            model_name=self.config.llm_model,
            prompt_path=self.config.prompt_dir / "generator" / "prompts" / "current.txt",
        )

        self.evaluator = RAGASEvaluator(model=self.config.llm_model)

    async def run(
        self,
        query: str,
        ground_truth: str | None = None,
        relevant_chunk_indices: list[int] | None = None,
        retrieval_k: int | None = None,
        final_k: int | None = None,
    ) -> PipelineResult:
        """
        Execute the full RAG pipeline.

        Args:
            query: User query
            ground_truth: Optional ground truth answer for evaluation
            relevant_chunk_indices: Optional list of relevant chunk indices
            retrieval_k: Number of chunks to retrieve (default from config)
            final_k: Number of chunks after reranking (default from config)

        Returns:
            PipelineResult with answer, contexts, and metrics
        """
        retrieval_k = retrieval_k or self.config.retrieval_k
        final_k = final_k or self.config.final_k

        # Module 1: Query Planning
        query_input = QueryPlannerInput(query=query)
        query_output = await self.query_planner.run(query_input)

        # Module 1: Retrieval
        retrieval_input = RetrievalInput(queries=query_output.queries, top_k=retrieval_k)
        retrieval_output = await self.retriever.run(retrieval_input)
        print("retrieval_output", retrieval_output)

        # Module 2: Reranking
        reranker_input = RerankerInput(
            query=query,
            documents=retrieval_output.document_texts,
        )
        reranker_output = await self.reranker.run(reranker_input)
        print("reranker_output", reranker_output)

        # Limit to final_k
        ranked_docs = reranker_output.ranked_documents[:final_k]
        context_text = "\n\n".join(ranked_docs)

        # Module 3: Generation
        generator_input = GeneratorInput(
            query=query,
            context=context_text,
        )
        generator_output = await self.generator.run(generator_input)
        print("generator_output", generator_output)

        # Evaluation
        metrics = await self.evaluator.evaluate_end_to_end(
            query=query,
            contexts=ranked_docs,
            answer=generator_output.answer,
            ground_truth=ground_truth,
            retrieved_chunk_indices=retrieval_output.chunk_indices,
            relevant_chunk_indices=relevant_chunk_indices,
        )

        return PipelineResult(
            answer=generator_output.answer,
            rationale=generator_output.rationale,
            contexts=retrieval_output.document_texts,
            ranked_documents=ranked_docs,
            metrics=metrics,
            metadata={
                "query_mode": query_output.mode,
                "num_queries": len(query_output.queries),
                "num_retrieved": len(retrieval_output.document_texts),
                "num_ranked": len(ranked_docs),
                "retrieved_chunk_indices": retrieval_output.chunk_indices,
            },
        )

    async def run_module_1(
        self,
        query: str,
        retrieval_k: int = 20,
        ground_truth: str | None = None,
    ) -> dict[str, Any]:
        """
        Run only Module 1 (Query Planner + Retrieval) for independent evaluation.
        """
        query_input = QueryPlannerInput(query=query)
        query_output = await self.query_planner.run(query_input)

        retrieval_input = RetrievalInput(queries=query_output.queries, top_k=retrieval_k)
        retrieval_output = await self.retriever.run(retrieval_input)

        # Evaluate retrieval
        retrieval_eval = await self.evaluator._evaluate_retrieval(
            retrieval_input,
            retrieval_output,
            {"query": query, "reference": ground_truth} if ground_truth else None,
        )

        return {
            "query_output": query_output,
            "retrieval_output": retrieval_output,
            "metrics": retrieval_eval,
        }

    async def run_module_2(
        self,
        query: str,
        documents: list[str],
        ground_truth: str | None = None,
    ) -> dict[str, Any]:
        """
        Run only Module 2 (Reranker) for independent evaluation.

        Args:
            query: User query
            documents: Pre-retrieved documents to rerank
            ground_truth: Ground truth answer for evaluation
        """
        reranker_input = RerankerInput(
            query=query,
            documents=documents,
        )
        reranker_output = await self.reranker.run(reranker_input)

        # Evaluate reranking
        reranker_eval = await self.evaluator._evaluate_reranker(
            reranker_input,
            reranker_output,
            {"query": query, "reference": ground_truth} if ground_truth else None,
        )

        return {
            "reranker_output": reranker_output,
            "metrics": reranker_eval,
        }

    async def run_module_3(
        self,
        query: str,
        context: str,
        ground_truth: str | None = None,
    ) -> dict[str, Any]:
        """
        Run only Module 3 (Generator) for independent evaluation.

        Args:
            query: User query
            context: Context string (concatenated relevant documents)
            ground_truth: Ground truth answer for evaluation
        """
        generator_input = GeneratorInput(
            query=query,
            context=context,
        )
        generator_output = await self.generator.run(generator_input)

        # Evaluate generation
        contexts = context.split("\n\n") if context else []
        generator_eval = await self.evaluator._evaluate_generator(
            generator_input,
            generator_output,
            {"query": query, "contexts": contexts, "reference": ground_truth}
            if ground_truth
            else None,
        )

        return {
            "generator_output": generator_output,
            "metrics": generator_eval,
        }

    def save_prompts(self, version: str = "v1") -> None:
        """Save all module prompts with version tag"""
        base_path = self.config.prompt_dir

        self.query_planner.save_prompt(base_path / "query_planner" / "prompts", version)
        self.reranker.save_prompt(base_path / "reranker" / "prompts", version)
        self.generator.save_prompt(base_path / "generator" / "prompts", version)

    def load_prompts(self, version: str = "v1") -> None:
        """Load all module prompts with version tag"""
        base_path = self.config.prompt_dir

        self.query_planner.load_prompt(base_path / "query_planner" / "prompts" / f"{version}.txt")
        self.reranker.load_prompt(base_path / "reranker" / "prompts" / f"{version}.txt")
        self.generator.load_prompt(base_path / "generator" / "prompts" / f"{version}.txt")

    def get_module_prompts(self) -> dict[str, str]:
        """Get all current module prompts"""
        return {
            "query_planner": self.query_planner.prompt or "",
            "reranker": self.reranker.prompt or "",
            "generator": self.generator.prompt or "",
        }

    def set_module_prompt(self, module_name: str, prompt: str) -> None:
        """Set prompt for a specific module"""
        if module_name == "query_planner":
            self.query_planner.prompt = prompt
        elif module_name == "reranker":
            self.reranker.prompt = prompt
        elif module_name == "generator":
            self.generator.prompt = prompt
        else:
            raise ValueError(f"Unknown module: {module_name}")

    def restore_all_prompts(self) -> None:
        """Restore all modules to their original prompts"""
        self.query_planner.restore_prompt()
        self.reranker.restore_prompt()
        self.generator.restore_prompt()
