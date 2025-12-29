"""
Query Planner GEPA Adapter (Module 1)

Optimizes query decomposition/reformulation prompts to improve retrieval quality.

Module I/O:
- Input: QueryPlannerInput(query: str, feedback: Optional[str])
- Output: QueryPlannerOutput(status, mode, queries: List[str], original_query, metadata)

Target Metrics: context_precision, context_recall (via retrieval results)
Optimization Goal: Generate queries that retrieve more relevant documents
"""

from typing import Any, Dict, List, Optional, Tuple

from gepa_adapters.base import (
    RAGModuleAdapter,
    RAGDataInst,
    RAGTrajectory,
    RAGRolloutOutput,
)

# Import module types
import sys
from pathlib import Path
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
    """

    def __init__(
        self,
        query_planner_module,
        retriever_module,  # HybridRetriever for downstream evaluation
        evaluator,
        failure_score: float = 0.0,
    ):
        """
        Initialize Query Planner Adapter.

        Args:
            query_planner_module: QueryPlannerModule instance
            retriever_module: HybridRetriever for evaluating query quality
            evaluator: RAGASEvaluator instance
            failure_score: Score on execution failure
        """
        super().__init__(
            module=query_planner_module,
            evaluator=evaluator,
            component_name="query_planner_prompt",
            failure_score=failure_score,
        )
        self.retriever = retriever_module

    async def _run_single_async(
        self,
        data: RAGDataInst,
    ) -> Tuple[RAGRolloutOutput, Dict[str, Any], Dict[str, Any]]:
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
            top_k=20  # Standard retrieval depth
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
        module_output: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute retrieval quality score using chunk-level metrics.

        Metrics:
        - context_precision: Relevant chunks in retrieved set
        - context_recall: Coverage of relevant chunks
        - F1: Harmonic mean of precision and recall

        Args:
            data: Original input with relevant_chunk_indices
            module_output: Module output with retrieved_chunk_indices

        Returns:
            Tuple of (f1_score, detailed_metrics)
        """
        retrieved_indices = module_output.get("retrieved_chunk_indices", [])
        relevant_indices = data.get("relevant_chunk_indices", [])

        if not relevant_indices:
            # No ground truth available, return neutral score
            return 0.5, {"context_precision": 0.5, "context_recall": 0.5, "f1": 0.5}

        # Use RetrievalMetrics for chunk-level evaluation (clean, no mock objects)
        eval_result = RetrievalMetrics.evaluate(
            retrieved_indices=retrieved_indices,
            relevant_indices=relevant_indices,
        )

        precision = eval_result.get("precision", 0.0)
        recall = eval_result.get("recall", 0.0)
        f1 = eval_result.get("f1", 0.0)

        metrics = {
            "context_precision": precision,
            "context_recall": recall,
            "f1": f1,
            "mrr": eval_result.get("mrr", 0.0),
            "ndcg": eval_result.get("ndcg", 0.0),
            "num_queries": module_output.get("num_queries", 1),
            "num_retrieved": module_output.get("num_retrieved", 0),
            "num_relevant": len(relevant_indices),
            "num_hits": len(set(retrieved_indices) & set(relevant_indices)),
        }

        return f1, metrics

    def _format_trace_for_reflection(
        self,
        trajectory: RAGTrajectory,
    ) -> Dict[str, Any]:
        """
        Format trajectory into GEPA reflection record.

        Returns:
            Dict with Inputs, Generated Outputs, Feedback keys
        """
        data = trajectory["data"]
        module_input = trajectory["module_input"]
        module_output = trajectory["module_output"]
        score = trajectory["score"]
        metrics = trajectory["metrics"]

        # Format inputs
        inputs_text = f"Query: {module_input.get('query', data['query'])}"

        # Format outputs
        mode = module_output.get("mode", "unknown")
        queries = module_output.get("queries", [])
        outputs_text = f"Mode: {mode}\nGenerated Queries:\n"
        for i, q in enumerate(queries, 1):
            outputs_text += f"  {i}. {q}\n"

        # Generate feedback
        feedback = self._generate_feedback(score, metrics)

        return {
            "Inputs": inputs_text,
            "Generated Outputs": outputs_text,
            "Feedback": feedback,
            "Score": score,
            "Metrics": metrics,
        }

    def _positive_feedback(self, score: float, metrics: Dict[str, float]) -> str:
        """Generate positive feedback for high-scoring query planning"""
        precision = metrics.get("context_precision", 0)
        recall = metrics.get("context_recall", 0)
        num_queries = metrics.get("num_queries", 1)

        feedback = f"Excellent query planning (F1={score:.2f}). "

        if num_queries > 1:
            feedback += f"Decomposition into {num_queries} sub-queries was effective. "
        else:
            feedback += "Query reformulation captured intent well. "

        feedback += f"Retrieved documents had {precision:.0%} precision and {recall:.0%} recall. "
        feedback += "The queries successfully targeted relevant documents."

        return feedback

    def _negative_feedback(self, score: float, metrics: Dict[str, float]) -> str:
        """Generate improvement suggestions for low-scoring query planning"""
        precision = metrics.get("context_precision", 0)
        recall = metrics.get("context_recall", 0)
        num_queries = metrics.get("num_queries", 1)

        feedback = f"Query planning needs improvement (F1={score:.2f}). "

        if precision < 0.5:
            feedback += "Low precision: generated queries retrieved too many irrelevant documents. "
            feedback += "Try adding more specific terms and constraints. "

        if recall < 0.5:
            feedback += "Low recall: missed relevant documents. "
            if num_queries == 1:
                feedback += "Consider decomposing the query into sub-queries to capture multiple aspects. "
            else:
                feedback += "Ensure sub-queries cover all key concepts and entities. "

        if num_queries > 3:
            feedback += f"Generated {num_queries} queries may be too fragmented. "
            feedback += "Focus on fewer, more comprehensive queries. "

        feedback += "Improve alignment between query intent and retrieval targets."

        return feedback


# =============================================================================
# Utility: Create adapter with standard configuration
# =============================================================================

def create_query_planner_adapter(
    query_planner_module,
    retriever_module,
    evaluator,
) -> QueryPlannerAdapter:
    """
    Factory function to create QueryPlannerAdapter.

    Args:
        query_planner_module: QueryPlannerModule instance
        retriever_module: HybridRetriever instance
        evaluator: RAGASEvaluator instance

    Returns:
        Configured QueryPlannerAdapter
    """
    return QueryPlannerAdapter(
        query_planner_module=query_planner_module,
        retriever_module=retriever_module,
        evaluator=evaluator,
        failure_score=0.0,
    )
