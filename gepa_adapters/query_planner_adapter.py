"""
Query Planner GEPA Adapter (Module 1)

Optimizes query decomposition/reformulation prompts to improve retrieval quality.

Module I/O:
- Input: QueryPlannerInput(query: str, feedback: Optional[str])
- Output: QueryPlannerOutput(status, mode, queries: List[str], original_query, metadata)

Target Metrics: context_precision, context_recall (via retrieval results)
Optimization Goal: Generate queries that retrieve more relevant documents

GEPA Optimization Notes:
- Uses metric-based diagnostic feedback to avoid overfitting
- Analyzes query structure and retrieval behavior patterns
- Provides strategy-focused suggestions (not ground-truth-specific concepts)
- Uses NDCG as primary metric (more sensitive to ranking quality than F1)
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
    - Metric-based diagnostic feedback (no ground truth in reflection)
    - Query structure analysis (diversity, coverage, specificity)
    - Retrieval behavior pattern diagnosis
    - Strategy-focused suggestions for improvement
    - Configurable primary metric (NDCG recommended)
    """

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
    # Query Structure Analysis (for diagnostic feedback without ground truth)
    # -------------------------------------------------------------------------

    def _analyze_query_structure(
        self,
        queries: list[str],
        original_query: str,
    ) -> dict[str, Any]:
        """
        Analyze query patterns without using ground truth.

        Args:
            queries: Generated queries from the planner
            original_query: The original user query

        Returns:
            Dict with structural analysis metrics
        """
        if not queries:
            return {
                "num_queries": 0,
                "avg_query_length": 0,
                "query_diversity": 0.0,
                "original_term_coverage": 0.0,
            }

        # Basic statistics
        num_queries = len(queries)
        avg_length = sum(len(q.split()) for q in queries) / num_queries

        # Calculate diversity between queries
        diversity = self._calculate_query_diversity(queries)

        # Check coverage of original query terms
        original_terms = set(original_query.lower().split())
        query_terms = set(" ".join(queries).lower().split())
        coverage = len(original_terms & query_terms) / max(len(original_terms), 1)

        return {
            "num_queries": num_queries,
            "avg_query_length": avg_length,
            "query_diversity": diversity,
            "original_term_coverage": coverage,
        }

    def _calculate_query_diversity(self, queries: list[str]) -> float:
        """
        Calculate average Jaccard diversity between query pairs.

        Higher diversity means queries cover different aspects.

        Args:
            queries: List of query strings

        Returns:
            Average diversity score (0-1, higher = more diverse)
        """
        if len(queries) < 2:
            return 0.0

        diversities = []
        for i, q1 in enumerate(queries):
            for q2 in queries[i + 1 :]:
                terms1 = set(q1.lower().split())
                terms2 = set(q2.lower().split())
                if terms1 or terms2:
                    # Jaccard distance (1 - similarity)
                    intersection = len(terms1 & terms2)
                    union = len(terms1 | terms2)
                    diversity = 1 - (intersection / union) if union > 0 else 0
                    diversities.append(diversity)

        return sum(diversities) / len(diversities) if diversities else 0.0

    def _analyze_retrieval_behavior(self, metrics: dict[str, float]) -> str:
        """
        Diagnose retrieval patterns from metrics alone (no ground truth needed).

        Args:
            metrics: Evaluation metrics dict

        Returns:
            Diagnostic string describing the retrieval pattern
        """
        precision = metrics.get("context_precision", 0)
        recall = metrics.get("context_recall", 0)
        ndcg = metrics.get("ndcg", 0)
        mrr = metrics.get("mrr", 0)

        # Diagnose based on metric patterns
        if precision > 0.6 and recall < 0.3:
            return (
                "PATTERN: High precision, low recall. "
                "Queries are specific but miss coverage. "
                "TRY: Add broader synonyms, related terms, or alternative phrasings."
            )
        elif precision < 0.3 and recall > 0.5:
            return (
                "PATTERN: Low precision, decent recall. "
                "Queries are too broad. "
                "TRY: Add more specific qualifiers, named entities, or domain terms."
            )
        elif precision < 0.3 and recall < 0.3:
            return (
                "PATTERN: Both precision and recall low. "
                "Queries may be off-topic or too vague. "
                "TRY: Reformulate to directly address the question's core intent."
            )
        elif ndcg < mrr * 0.7 and mrr > 0.3:
            return (
                "PATTERN: Good first hit but poor ranking tail. "
                "TRY: Generate queries for each distinct aspect of the question."
            )
        elif ndcg > 0.5 and recall < 0.5:
            return (
                "PATTERN: Good ranking but incomplete coverage. "
                "TRY: Add queries targeting different facets of the information need."
            )
        else:
            return (
                f"Retrieval metrics: Precision={precision:.0%}, Recall={recall:.0%}, NDCG={ndcg:.2f}. "
                "Consider query structure and term coverage."
            )

    # -------------------------------------------------------------------------
    # Reflective Dataset Generation (Enhanced for GEPA)
    # -------------------------------------------------------------------------

    def _format_trace_for_reflection(
        self,
        trajectory: RAGTrajectory,
    ) -> dict[str, Any]:
        """
        Format trajectory into GEPA reflection record with diagnostic feedback.

        GEPA Best Practices Applied:
        1. NO ground truth in reflection (prevents overfitting)
        2. Query structure analysis (diversity, coverage)
        3. Metric-based retrieval behavior diagnosis
        4. Strategy-focused improvement suggestions

        Returns:
            Dict with Inputs, Generated Outputs, Retrieval Analysis, Feedback, Score
        """
        data = trajectory["data"]
        module_input = trajectory["module_input"]
        module_output = trajectory["module_output"]
        score = trajectory["score"]
        metrics = trajectory["metrics"]

        # Extract key information
        query = module_input.get("query", data.get("query", ""))
        mode = module_output.get("mode", "unknown")
        queries = module_output.get("queries", [])

        # Retrieval statistics
        num_hits = metrics.get("num_hits", 0)
        num_relevant = metrics.get("num_relevant", 0)
        precision = metrics.get("context_precision", 0)
        recall = metrics.get("context_recall", 0)
        ndcg = metrics.get("ndcg", 0)

        # Query structure analysis
        query_analysis = self._analyze_query_structure(queries, query)

        # Format inputs
        inputs_text = f"User Query: {query}"

        # Format generated outputs with structure analysis
        outputs_text = f"Planning Mode: {mode}\n"
        outputs_text += f"Generated Queries ({len(queries)}):\n"
        for i, q in enumerate(queries, 1):
            outputs_text += f"  {i}. {q}\n"
        outputs_text += f"\nQuery Analysis: {query_analysis['num_queries']} queries, "
        outputs_text += f"{query_analysis['query_diversity']:.0%} diversity, "
        outputs_text += f"{query_analysis['original_term_coverage']:.0%} original term coverage"

        # Format retrieval result with metrics
        retrieval_text = (
            f"Retrieval Result: Found {num_hits}/{num_relevant} relevant documents "
            f"(Precision: {precision:.1%}, Recall: {recall:.1%}, NDCG: {ndcg:.2f})"
        )

        # Generate diagnostic feedback (no ground truth)
        feedback = self._generate_rich_feedback(
            score=score,
            metrics=metrics,
            generated_queries=queries,
            original_query=query,
            mode=mode,
        )

        return {
            "Inputs": inputs_text,
            "Generated Outputs": outputs_text,
            "Retrieval Analysis": retrieval_text,
            "Feedback": feedback,
            "Score": f"{score:.3f}",
        }

    def _generate_rich_feedback(
        self,
        score: float,
        metrics: dict[str, float],
        generated_queries: list[str],
        original_query: str,
        mode: str,
    ) -> str:
        """
        Generate diagnostic feedback for GEPA reflection (no ground truth).

        Uses metric patterns and query structure analysis to provide
        strategy-focused improvement suggestions.

        Args:
            score: The primary metric score
            metrics: All evaluation metrics
            generated_queries: Queries produced by the planner
            original_query: The original user query
            mode: decomposition or reformulation

        Returns:
            Diagnostic feedback string with strategy suggestions
        """
        # Determine performance tier using adaptive thresholds
        if score >= 0.5:
            return self._positive_feedback_rich(
                score, metrics, generated_queries, original_query, mode
            )
        if score >= 0.2:
            return self._partial_feedback_rich(
                score, metrics, generated_queries, original_query, mode
            )
        return self._negative_feedback_rich(
            score, metrics, generated_queries, original_query, mode
        )

    def _positive_feedback_rich(
        self,
        score: float,
        metrics: dict[str, float],
        generated_queries: list[str],
        original_query: str,
        mode: str,
    ) -> str:
        """Generate positive feedback with reinforcement of what worked."""
        precision = metrics.get("context_precision", 0)
        recall = metrics.get("context_recall", 0)
        num_hits = metrics.get("num_hits", 0)
        num_relevant = metrics.get("num_relevant", 0)

        feedback = f"GOOD: Score={score:.2f}. "

        # Explain what worked based on mode and metrics
        if mode == "decomposition" and len(generated_queries) > 1:
            feedback += f"Decomposition into {len(generated_queries)} sub-queries effectively covered the information need. "
        else:
            feedback += "Query reformulation successfully captured key aspects. "

        feedback += f"Retrieved {num_hits}/{num_relevant} relevant documents. "

        # Analyze query structure to explain success
        query_analysis = self._analyze_query_structure(generated_queries, original_query)
        if query_analysis["query_diversity"] > 0.5:
            feedback += f"Good query diversity ({query_analysis['query_diversity']:.0%}). "
        if query_analysis["original_term_coverage"] > 0.7:
            feedback += "Strong coverage of original query terms. "

        feedback += "Continue this approach for similar queries."

        return feedback

    def _partial_feedback_rich(
        self,
        score: float,
        metrics: dict[str, float],
        generated_queries: list[str],
        original_query: str,
        mode: str,
    ) -> str:
        """Generate feedback for partial success with strategy-focused improvements."""
        precision = metrics.get("context_precision", 0)
        recall = metrics.get("context_recall", 0)
        num_hits = metrics.get("num_hits", 0)
        num_relevant = metrics.get("num_relevant", 0)

        feedback = f"PARTIAL: Score={score:.2f}. "

        # Query structure analysis
        query_analysis = self._analyze_query_structure(generated_queries, original_query)
        feedback += f"Analysis: {query_analysis['num_queries']} queries, "
        feedback += f"{query_analysis['query_diversity']:.0%} diversity. "

        # Use retrieval behavior analysis for diagnosis
        behavior_diagnosis = self._analyze_retrieval_behavior(metrics)
        feedback += behavior_diagnosis + " "

        # Mode-specific strategy suggestions
        if mode == "reformulation" and num_relevant > 3 and recall < 0.4:
            feedback += "STRATEGY: Consider decomposing into sub-queries to capture multiple aspects. "
        elif mode == "decomposition" and len(generated_queries) > 3 and precision < 0.3:
            feedback += "STRATEGY: Consolidate into fewer, more focused sub-queries. "
        elif query_analysis["query_diversity"] < 0.3:
            feedback += "STRATEGY: Increase query diversity - target different aspects of the question. "
        elif query_analysis["original_term_coverage"] < 0.5:
            feedback += "STRATEGY: Ensure generated queries cover key terms from the original question. "

        return feedback

    def _negative_feedback_rich(
        self,
        score: float,
        metrics: dict[str, float],
        generated_queries: list[str],
        original_query: str,
        mode: str,
    ) -> str:
        """Generate detailed negative feedback with strategy-focused suggestions."""
        precision = metrics.get("context_precision", 0)
        recall = metrics.get("context_recall", 0)
        num_hits = metrics.get("num_hits", 0)
        num_relevant = metrics.get("num_relevant", 0)

        feedback = f"POOR: Score={score:.2f}. Only {num_hits}/{num_relevant} relevant docs found. "

        # Query structure analysis
        query_analysis = self._analyze_query_structure(generated_queries, original_query)

        # Diagnose primary failure mode using metrics
        if recall < 0.1:
            feedback += "CRITICAL: Almost no relevant documents retrieved. "
            feedback += "The queries may be completely off-topic or missing key terms. "
        elif precision < 0.1:
            feedback += "CRITICAL: Retrieved documents mostly irrelevant. "
            feedback += "The queries may be too broad or generic. "

        # Use retrieval behavior analysis
        behavior_diagnosis = self._analyze_retrieval_behavior(metrics)
        feedback += behavior_diagnosis + " "

        # Structure-based suggestions (no ground truth needed)
        if len(generated_queries) == 1:
            feedback += "STRATEGY: Break into 2-3 sub-queries targeting specific aspects of the question. "
        elif len(generated_queries) > 3:
            feedback += "STRATEGY: Too fragmented. Consolidate into 2-3 comprehensive queries. "
        elif query_analysis["query_diversity"] < 0.2:
            feedback += "STRATEGY: Queries are too similar. Diversify to cover different facets. "
        elif query_analysis["original_term_coverage"] < 0.3:
            feedback += "STRATEGY: Queries miss key terms from the original question. Include them explicitly. "
        else:
            feedback += "STRATEGY: Add specific domain terms, named entities, or technical vocabulary. "

        # General recovery strategies
        feedback += "RECOVERY: (1) Re-read the question for key information needs, "
        feedback += "(2) Generate queries for each distinct aspect, "
        feedback += "(3) Include synonyms and related terms."

        return feedback

    # -------------------------------------------------------------------------
    # Legacy feedback methods (kept for compatibility, delegate to rich versions)
    # -------------------------------------------------------------------------

    def _positive_feedback(self, score: float, metrics: dict[str, float]) -> str:
        """Generate positive feedback for high-scoring query planning."""
        return self._positive_feedback_rich(score, metrics, [], "", "reformulation")

    def _negative_feedback(self, score: float, metrics: dict[str, float]) -> str:
        """Generate improvement suggestions for low-scoring query planning."""
        return self._negative_feedback_rich(score, metrics, [], "", "reformulation")


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
