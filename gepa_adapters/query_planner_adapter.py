"""
Query Planner GEPA Adapter (Module 1)

Optimizes query decomposition/reformulation prompts to improve retrieval quality.

Module I/O:
- Input: QueryPlannerInput(query: str, feedback: Optional[str])
- Output: QueryPlannerOutput(status, mode, queries: List[str], original_query, metadata)

Target Metrics: context_precision, context_recall (via retrieval results)
Optimization Goal: Generate queries that retrieve more relevant documents

GEPA Optimization Notes:
- Reflective feedback includes ground truth context for better prompt evolution
- Uses NDCG as primary metric (more sensitive to ranking quality than F1)
- Provides contrastive examples showing what queries should have targeted
"""

import re

# Import module types
import sys
from pathlib import Path
from typing import Any

from gepa_adapters.base import (
    RAGDataInst,
    RAGModuleAdapter,
    RAGRolloutOutput,
    RAGTrajectory,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.base import QueryPlannerInput, RetrievalInput
from modules.evaluation.metrics import RetrievalMetrics


class QueryPlannerAdapter(RAGModuleAdapter):
    """
    GEPA Adapter for Query Planner Module.

    This adapter optimizes the query planning prompt to improve:
    - Query decomposition for multi-intent queries
    - Query reformulation for ambiguous queries

    The optimization target is to maximize retrieval quality
    (context_precision, context_recall) downstream.

    Key Features for GEPA Optimization:
    - Rich reflective feedback with ground truth context
    - Contrastive examples showing missed concepts
    - Configurable primary metric (NDCG recommended)
    - Adaptive feedback thresholds
    """

    # Stopwords to filter when extracting key concepts
    STOPWORDS = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "although",
        "though",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "its",
        "it",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "their",
        "my",
        "your",
        "our",
        "we",
        "you",
        "i",
        "me",
        "him",
        "us",
        "answer",
        "question",
    }

    def __init__(
        self,
        query_planner_module,
        retriever_module,  # HybridRetriever for downstream evaluation
        evaluator,
        failure_score: float = 0.0,
        primary_metric: str = "ndcg",  # "ndcg", "mrr", or "f1"
    ):
        """
        Initialize Query Planner Adapter.

        Args:
            query_planner_module: QueryPlannerModule instance
            retriever_module: HybridRetriever for evaluating query quality
            evaluator: RAGASEvaluator instance
            failure_score: Score on execution failure
            primary_metric: Which metric to optimize ("ndcg", "mrr", or "f1")
                           NDCG recommended as it's more sensitive to ranking quality
        """
        super().__init__(
            module=query_planner_module,
            evaluator=evaluator,
            component_name="query_planner_prompt",
            failure_score=failure_score,
        )
        self.retriever = retriever_module
        self.primary_metric = primary_metric.lower()

        if self.primary_metric not in ("ndcg", "mrr", "f1"):
            raise ValueError(
                f"primary_metric must be 'ndcg', 'mrr', or 'f1', got '{primary_metric}'"
            )

    async def _run_single_async(
        self,
        data: RAGDataInst,
    ) -> tuple[RAGRolloutOutput, dict[str, Any], dict[str, Any]]:
        """
        Execute query planner and retriever on a single example.

        Process:
        1. Run query planner to get decomposed/reformulated queries
        2. Run retriever with generated queries
        3. Return output for evaluation

        Args:
            data: RAGDataInst with query, ground_truth, relevant_chunk_indices

        Returns:
            Tuple of (output, module_input, module_output)
        """
        query = data["query"]

        # Step 1: Run query planner
        planner_input = QueryPlannerInput(query=query)
        planner_output = await self.module.run(planner_input)

        # Step 2: Run retriever with generated queries
        retrieval_input = RetrievalInput(
            queries=planner_output.queries,
            top_k=20,  # Standard retrieval depth
        )
        retrieval_output = await self.retriever.run(retrieval_input)

        # Package output
        output: RAGRolloutOutput = {
            "result": {
                "planner_mode": planner_output.mode,
                "generated_queries": planner_output.queries,
                "retrieved_documents": retrieval_output.document_texts,
                "chunk_indices": retrieval_output.chunk_indices,
            },
            "success": planner_output.status == "success" and retrieval_output.status == "success",
            "error": planner_output.error_message or retrieval_output.error_message,
        }

        module_input = {
            "query": query,
            "feedback": None,
        }

        module_output = {
            "mode": planner_output.mode,
            "queries": planner_output.queries,
            "num_queries": len(planner_output.queries),
            "retrieved_chunk_indices": retrieval_output.chunk_indices,
            "num_retrieved": len(retrieval_output.document_texts),
        }

        return output, module_input, module_output

    async def _compute_score_async(
        self,
        data: RAGDataInst,
        module_output: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        """
        Compute retrieval quality score using chunk-level metrics.

        Metrics:
        - context_precision: Relevant chunks in retrieved set
        - context_recall: Coverage of relevant chunks
        - F1: Harmonic mean of precision and recall
        - NDCG: Normalized Discounted Cumulative Gain (ranking-sensitive)
        - MRR: Mean Reciprocal Rank (first relevant item position)

        Args:
            data: Original input with relevant_chunk_indices
            module_output: Module output with retrieved_chunk_indices

        Returns:
            Tuple of (primary_score, detailed_metrics)
        """
        retrieved_indices = module_output.get("retrieved_chunk_indices", [])
        relevant_indices = data.get("relevant_chunk_indices", [])

        if not relevant_indices:
            # No ground truth available, return neutral score
            return 0.5, {
                "context_precision": 0.5,
                "context_recall": 0.5,
                "f1": 0.5,
                "ndcg": 0.5,
                "mrr": 0.5,
            }

        # Use RetrievalMetrics for chunk-level evaluation
        eval_result = RetrievalMetrics.evaluate(
            retrieved_indices=retrieved_indices,
            relevant_indices=relevant_indices,
        )

        precision = eval_result.get("precision", 0.0)
        recall = eval_result.get("recall", 0.0)
        f1 = eval_result.get("f1", 0.0)
        ndcg = eval_result.get("ndcg", 0.0)
        mrr = eval_result.get("mrr", 0.0)

        metrics = {
            "context_precision": precision,
            "context_recall": recall,
            "f1": f1,
            "mrr": mrr,
            "ndcg": ndcg,
            "num_queries": module_output.get("num_queries", 1),
            "num_retrieved": module_output.get("num_retrieved", 0),
            "num_relevant": len(relevant_indices),
            "num_hits": len(set(retrieved_indices) & set(relevant_indices)),
        }

        # Return the configured primary metric as the optimization target
        if self.primary_metric == "ndcg":
            primary_score = ndcg
        elif self.primary_metric == "mrr":
            primary_score = mrr
        else:  # f1
            primary_score = f1

        return primary_score, metrics

    # -------------------------------------------------------------------------
    # Key Concept Extraction (for contrastive feedback)
    # -------------------------------------------------------------------------

    def _extract_key_concepts(self, text: str, max_concepts: int = 10) -> list[str]:
        """
        Extract key concepts/terms from text for contrastive feedback.

        This helps GEPA understand what terms SHOULD have been in the queries.

        Args:
            text: Ground truth answer or document text
            max_concepts: Maximum number of concepts to extract

        Returns:
            List of key terms/concepts
        """
        if not text:
            return []

        # Tokenize and clean
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]*[a-zA-Z0-9]\b|\b[a-zA-Z]\b", text.lower())

        # Filter stopwords and short words
        meaningful_words = [w for w in words if w not in self.STOPWORDS and len(w) > 2]

        # Count frequency
        word_freq: dict[str, int] = {}
        for w in meaningful_words:
            word_freq[w] = word_freq.get(w, 0) + 1

        # Sort by frequency and return top concepts
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        return [w for w, _ in sorted_words[:max_concepts]]

    def _find_missing_concepts(
        self,
        generated_queries: list[str],
        ground_truth: str,
    ) -> list[str]:
        """
        Find concepts from ground truth that are missing from generated queries.

        Args:
            generated_queries: The queries generated by the planner
            ground_truth: The expected answer

        Returns:
            List of concepts that should have been targeted but weren't
        """
        if not ground_truth:
            return []

        # Get concepts from ground truth
        gt_concepts = set(self._extract_key_concepts(ground_truth, max_concepts=15))

        # Get concepts covered by generated queries
        query_text = " ".join(generated_queries)
        query_concepts = set(self._extract_key_concepts(query_text, max_concepts=20))

        # Find missing concepts
        missing = gt_concepts - query_concepts

        return list(missing)[:8]  # Limit to 8 most important missing concepts

    # -------------------------------------------------------------------------
    # Reflective Dataset Generation (Enhanced for GEPA)
    # -------------------------------------------------------------------------

    def _format_trace_for_reflection(
        self,
        trajectory: RAGTrajectory,
    ) -> dict[str, Any]:
        """
        Format trajectory into GEPA reflection record with rich feedback.

        GEPA Best Practices Applied:
        1. Include ground truth context so reflection LLM knows the target
        2. Show what concepts were missed (contrastive feedback)
        3. Provide specific, actionable improvement suggestions
        4. Include retrieval statistics for context

        Returns:
            Dict with Inputs, Generated Outputs, Expected Answer, Retrieval Result, Feedback
        """
        data = trajectory["data"]
        module_input = trajectory["module_input"]
        module_output = trajectory["module_output"]
        score = trajectory["score"]
        metrics = trajectory["metrics"]

        # Extract key information
        query = module_input.get("query", data.get("query", ""))
        ground_truth = data.get("ground_truth", "")
        mode = module_output.get("mode", "unknown")
        queries = module_output.get("queries", [])

        # Retrieval statistics
        num_hits = metrics.get("num_hits", 0)
        num_relevant = metrics.get("num_relevant", 0)
        num_retrieved = metrics.get("num_retrieved", 0)
        precision = metrics.get("context_precision", 0)
        recall = metrics.get("context_recall", 0)
        ndcg = metrics.get("ndcg", 0)

        # Format inputs
        inputs_text = f"User Query: {query}"

        # Format generated outputs
        outputs_text = f"Planning Mode: {mode}\n"
        outputs_text += f"Generated Queries ({len(queries)}):\n"
        for i, q in enumerate(queries, 1):
            outputs_text += f"  {i}. {q}\n"

        # Format expected answer (truncated for brevity)
        if ground_truth:
            gt_preview = ground_truth[:400] + "..." if len(ground_truth) > 400 else ground_truth
            expected_text = f"Expected Answer: {gt_preview}"
        else:
            expected_text = "Expected Answer: Not available"

        # Format retrieval result
        retrieval_text = (
            f"Retrieval Result: Found {num_hits}/{num_relevant} relevant documents "
            f"(Precision: {precision:.1%}, Recall: {recall:.1%}, NDCG: {ndcg:.2f})"
        )

        # Generate rich feedback with ground truth context
        feedback = self._generate_rich_feedback(
            score=score,
            metrics=metrics,
            ground_truth=ground_truth,
            generated_queries=queries,
            mode=mode,
        )

        return {
            "Inputs": inputs_text,
            "Generated Outputs": outputs_text,
            "Expected Answer": expected_text,
            "Retrieval Result": retrieval_text,
            "Feedback": feedback,
            "Score": f"{score:.3f}",
        }

    def _generate_rich_feedback(
        self,
        score: float,
        metrics: dict[str, float],
        ground_truth: str,
        generated_queries: list[str],
        mode: str,
    ) -> str:
        """
        Generate rich, contrastive feedback for GEPA reflection.

        This is the key improvement: provide specific, actionable feedback
        with examples of what should have been done differently.

        Args:
            score: The primary metric score
            metrics: All evaluation metrics
            ground_truth: The expected answer
            generated_queries: Queries produced by the planner
            mode: decomposition or reformulation

        Returns:
            Detailed feedback string with contrastive examples
        """
        precision = metrics.get("context_precision", 0)
        recall = metrics.get("context_recall", 0)
        ndcg = metrics.get("ndcg", 0)
        num_hits = metrics.get("num_hits", 0)
        num_relevant = metrics.get("num_relevant", 0)
        num_queries = len(generated_queries)

        # Determine performance tier using adaptive thresholds
        # (relative to typical RAG performance, not absolute 0.7)
        if score >= 0.5:
            return self._positive_feedback_rich(
                score, metrics, ground_truth, generated_queries, mode
            )
        if score >= 0.2:
            return self._partial_feedback_rich(
                score, metrics, ground_truth, generated_queries, mode
            )
        return self._negative_feedback_rich(score, metrics, ground_truth, generated_queries, mode)

    def _positive_feedback_rich(
        self,
        score: float,
        metrics: dict[str, float],
        ground_truth: str,
        generated_queries: list[str],
        mode: str,
    ) -> str:
        """Generate positive feedback with reinforcement of what worked"""
        precision = metrics.get("context_precision", 0)
        recall = metrics.get("context_recall", 0)
        num_hits = metrics.get("num_hits", 0)
        num_relevant = metrics.get("num_relevant", 0)

        feedback = f"GOOD: Score={score:.2f}. "

        # Explain what worked
        if mode == "decomposition" and len(generated_queries) > 1:
            feedback += f"Decomposition into {len(generated_queries)} sub-queries effectively covered the information need. "
        else:
            feedback += "Query reformulation successfully captured key concepts. "

        feedback += f"Retrieved {num_hits}/{num_relevant} relevant documents. "

        # Extract what concepts were captured
        if ground_truth:
            gt_concepts = self._extract_key_concepts(ground_truth, max_concepts=5)
            query_text = " ".join(generated_queries)
            matched = [c for c in gt_concepts if c.lower() in query_text.lower()]
            if matched:
                feedback += f"Key concepts captured: {', '.join(matched[:3])}. "

        feedback += "Continue this approach for similar queries."

        return feedback

    def _partial_feedback_rich(
        self,
        score: float,
        metrics: dict[str, float],
        ground_truth: str,
        generated_queries: list[str],
        mode: str,
    ) -> str:
        """Generate feedback for partial success with specific improvements"""
        precision = metrics.get("context_precision", 0)
        recall = metrics.get("context_recall", 0)
        num_hits = metrics.get("num_hits", 0)
        num_relevant = metrics.get("num_relevant", 0)

        feedback = f"PARTIAL: Score={score:.2f}. "

        # Diagnose the issue
        if precision < recall:
            feedback += "Queries too broad - retrieved many irrelevant documents. "
        elif recall < precision:
            feedback += "Queries too narrow - missed relevant documents. "
        else:
            feedback += "Moderate retrieval quality. "

        # Find missing concepts
        if ground_truth:
            missing = self._find_missing_concepts(generated_queries, ground_truth)
            if missing:
                feedback += f"MISSING CONCEPTS to include: {', '.join(missing[:5])}. "

        # Suggest mode change if appropriate
        if mode == "reformulation" and num_relevant > 3 and recall < 0.4:
            feedback += "TRY: Decompose into sub-queries to capture multiple aspects. "
        elif mode == "decomposition" and len(generated_queries) > 3 and precision < 0.3:
            feedback += "TRY: Fewer, more focused sub-queries. "

        return feedback

    def _negative_feedback_rich(
        self,
        score: float,
        metrics: dict[str, float],
        ground_truth: str,
        generated_queries: list[str],
        mode: str,
    ) -> str:
        """Generate detailed negative feedback with contrastive examples"""
        precision = metrics.get("context_precision", 0)
        recall = metrics.get("context_recall", 0)
        num_hits = metrics.get("num_hits", 0)
        num_relevant = metrics.get("num_relevant", 0)

        feedback = f"POOR: Score={score:.2f}. Only {num_hits}/{num_relevant} relevant docs found. "

        # Diagnose primary failure mode
        if recall < 0.1:
            feedback += "CRITICAL: Almost no relevant documents retrieved. "
        elif precision < 0.1:
            feedback += "CRITICAL: Retrieved documents mostly irrelevant. "

        # Provide contrastive examples from ground truth
        if ground_truth:
            missing = self._find_missing_concepts(generated_queries, ground_truth)
            if missing:
                feedback += f"MUST INCLUDE these concepts: {', '.join(missing[:6])}. "

            # Show what the answer contains
            gt_preview = ground_truth[:150].replace("\n", " ")
            feedback += f"TARGET: The answer discusses '{gt_preview}...'. "

        # Concrete suggestions
        if len(generated_queries) == 1:
            feedback += "SUGGESTION: Break into 2-3 sub-queries targeting specific aspects. "
        elif len(generated_queries) > 3:
            feedback += "SUGGESTION: Too fragmented. Consolidate into 2-3 comprehensive queries. "
        else:
            feedback += "SUGGESTION: Add specific domain terms and named entities. "

        return feedback

    # -------------------------------------------------------------------------
    # Legacy feedback methods (kept for compatibility, delegate to rich versions)
    # -------------------------------------------------------------------------

    def _positive_feedback(self, score: float, metrics: dict[str, float]) -> str:
        """Generate positive feedback for high-scoring query planning"""
        return self._positive_feedback_rich(score, metrics, "", [], "reformulation")

    def _negative_feedback(self, score: float, metrics: dict[str, float]) -> str:
        """Generate improvement suggestions for low-scoring query planning"""
        return self._negative_feedback_rich(score, metrics, "", [], "reformulation")


# =============================================================================
# Utility: Create adapter with standard configuration
# =============================================================================


def create_query_planner_adapter(
    query_planner_module,
    retriever_module,
    evaluator,
    primary_metric: str = "ndcg",
) -> QueryPlannerAdapter:
    """
    Factory function to create QueryPlannerAdapter.

    Args:
        query_planner_module: QueryPlannerModule instance
        retriever_module: HybridRetriever instance
        evaluator: RAGASEvaluator instance
        primary_metric: Which metric to optimize ("ndcg", "mrr", or "f1")
                       NDCG is recommended as it's more sensitive to ranking quality

    Returns:
        Configured QueryPlannerAdapter
    """
    return QueryPlannerAdapter(
        query_planner_module=query_planner_module,
        retriever_module=retriever_module,
        evaluator=evaluator,
        failure_score=0.0,
        primary_metric=primary_metric,
    )
