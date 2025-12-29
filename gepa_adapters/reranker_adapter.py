"""
Reranker GEPA Adapter (Module 2)

Optimizes listwise reranking prompts to improve document ordering.

Module I/O:
- Input: RerankerInput(query: str, documents: List[str], feedback: Optional[str])
- Output: RerankerOutput(status, ranked_documents: List[str], scores: List[float], metadata)

Target Metrics: context_precision, context_recall (post-rerank improvement)
Optimization Goal: Surface relevant documents to top positions
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

from modules.base import RerankerInput


class RerankerAdapter(RAGModuleAdapter):
    """
    GEPA Adapter for Reranker Module.

    This adapter optimizes the reranking prompt to:
    - Move relevant documents to top positions
    - Improve precision in the top-k results
    - Filter out irrelevant documents through reordering

    The optimization evaluates improvement over pre-rerank ordering.
    """

    def __init__(
        self,
        reranker_module,
        evaluator,
        top_k: int = 10,  # Number of documents to consider after reranking
        failure_score: float = 0.0,
    ):
        """
        Initialize Reranker Adapter.

        Args:
            reranker_module: RerankerModule instance
            evaluator: RAGASEvaluator instance
            top_k: Number of top documents to evaluate
            failure_score: Score on execution failure
        """
        super().__init__(
            module=reranker_module,
            evaluator=evaluator,
            component_name="reranker_prompt",
            failure_score=failure_score,
        )
        self.top_k = top_k

    async def _run_single_async(
        self,
        data: RAGDataInst,
    ) -> Tuple[RAGRolloutOutput, Dict[str, Any], Dict[str, Any]]:
        """
        Execute reranker on a single example.

        Process:
        1. Take pre-retrieved documents from data["contexts"]
        2. Run reranker to reorder documents
        3. Return reranked documents for evaluation

        Args:
            data: RAGDataInst with query, contexts (pre-retrieved docs)

        Returns:
            Tuple of (output, module_input, module_output)
        """
        query = data["query"]
        documents = data.get("contexts", [])

        if not documents:
            return {
                "result": None,
                "success": False,
                "error": "No documents provided for reranking",
            }, {"query": query, "documents": []}, {}

        # Run reranker
        reranker_input = RerankerInput(
            query=query,
            documents=documents,
        )
        reranker_output = await self.module.run(reranker_input)

        # Package output
        ranked_docs = reranker_output.ranked_documents[:self.top_k]

        output: RAGRolloutOutput = {
            "result": {
                "ranked_documents": ranked_docs,
                "scores": reranker_output.scores[:self.top_k],
                "num_input": len(documents),
                "num_output": len(ranked_docs),
            },
            "success": reranker_output.status == "success",
            "error": reranker_output.error_message,
        }

        module_input = {
            "query": query,
            "documents": documents,
            "num_documents": len(documents),
        }

        module_output = {
            "ranked_documents": ranked_docs,
            "scores": reranker_output.scores[:self.top_k],
            "original_order": documents,
        }

        return output, module_input, module_output

    async def _compute_score_async(
        self,
        data: RAGDataInst,
        module_output: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reranking quality score.

        Evaluates improvement in ranking quality:
        - How many relevant docs moved to top positions
        - Improvement over original ordering

        Args:
            data: Original input with ground_truth
            module_output: Reranker output with ranked_documents

        Returns:
            Tuple of (score, detailed_metrics)
        """
        ranked_docs = module_output.get("ranked_documents", [])
        original_order = module_output.get("original_order", [])
        ground_truth = data.get("ground_truth", "")
        query = data["query"]

        # Use LLM-based evaluation for reranking quality
        reranker_eval = await self.evaluator._evaluate_reranker(
            input_data=RerankerInput(query=query, documents=original_order),
            output_data=type('obj', (object,), {'ranked_documents': ranked_docs})(),
            ground_truth={"query": query, "reference": ground_truth} if ground_truth else None
        )

        precision = reranker_eval.get("context_precision", 0.0)
        recall = reranker_eval.get("context_recall", 0.0)
        f1 = reranker_eval.get("context_f1", 0.0)

        # Calculate if F1 not provided
        if f1 == 0.0 and (precision > 0 or recall > 0):
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calculate position improvement (how much did relevant docs move up)
        position_improvement = self._calculate_position_improvement(
            original_order, ranked_docs, ground_truth
        )

        metrics = {
            "context_precision": precision,
            "context_recall": recall,
            "f1": f1,
            "position_improvement": position_improvement,
            "num_input": len(original_order),
            "num_output": len(ranked_docs),
        }

        # Combined score: F1 weighted with position improvement bonus
        score = f1 * 0.8 + position_improvement * 0.2

        return score, metrics

    def _calculate_position_improvement(
        self,
        original: List[str],
        reranked: List[str],
        ground_truth: str,
    ) -> float:
        """
        Calculate how much relevant documents improved in position.

        Returns a score from 0-1 where:
        - 1.0 = all relevant docs moved to top
        - 0.0 = no improvement or degradation
        """
        if not ground_truth or not original or not reranked:
            return 0.5

        # Find documents containing ground truth content
        gt_lower = ground_truth.lower()
        relevant_original_positions = []
        relevant_reranked_positions = []

        for i, doc in enumerate(original):
            if any(term in doc.lower() for term in gt_lower.split()[:5]):
                relevant_original_positions.append(i)

        for i, doc in enumerate(reranked):
            if any(term in doc.lower() for term in gt_lower.split()[:5]):
                relevant_reranked_positions.append(i)

        if not relevant_original_positions or not relevant_reranked_positions:
            return 0.5

        # Calculate average position improvement
        avg_original = sum(relevant_original_positions) / len(relevant_original_positions)
        avg_reranked = sum(relevant_reranked_positions) / len(relevant_reranked_positions)

        # Normalize: improvement from moving to top (position 0)
        max_improvement = avg_original
        actual_improvement = avg_original - avg_reranked

        if max_improvement <= 0:
            return 0.5

        improvement_ratio = max(0, min(1, actual_improvement / max_improvement))
        return improvement_ratio

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
        query = module_input.get("query", data["query"])
        docs = module_input.get("documents", [])

        inputs_text = f"Query: {query}\n\nDocuments to rerank ({len(docs)} total):\n"
        for i, doc in enumerate(docs[:5], 1):  # Show first 5
            doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
            inputs_text += f"  [{i}] {doc_preview}\n"
        if len(docs) > 5:
            inputs_text += f"  ... and {len(docs) - 5} more\n"

        # Format outputs
        ranked = module_output.get("ranked_documents", [])
        outputs_text = f"Reranked order (top {len(ranked)}):\n"
        for i, doc in enumerate(ranked[:5], 1):
            doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
            outputs_text += f"  [{i}] {doc_preview}\n"

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
        """Generate positive feedback for effective reranking"""
        precision = metrics.get("context_precision", 0)
        position_improvement = metrics.get("position_improvement", 0)

        feedback = f"Effective reranking (score={score:.2f}). "
        feedback += f"Achieved {precision:.0%} precision in top documents. "

        if position_improvement > 0.5:
            feedback += "Successfully moved relevant documents to top positions. "
        elif position_improvement > 0:
            feedback += "Moderate improvement in document ordering. "

        feedback += "The ranking criteria correctly prioritized relevant content."

        return feedback

    def _negative_feedback(self, score: float, metrics: Dict[str, float]) -> str:
        """Generate improvement suggestions for ineffective reranking"""
        precision = metrics.get("context_precision", 0)
        recall = metrics.get("context_recall", 0)
        position_improvement = metrics.get("position_improvement", 0)

        feedback = f"Reranking needs improvement (score={score:.2f}). "

        if precision < 0.5:
            feedback += "Low precision: irrelevant documents ranked too high. "
            feedback += "Focus on content that directly answers the query. "

        if recall < 0.5:
            feedback += "Low recall: relevant documents pushed too low. "
            feedback += "Ensure comprehensive explanations are prioritized. "

        if position_improvement < 0.3:
            feedback += "Document positions did not improve significantly. "
            feedback += "Better identify which documents contain the answer. "

        feedback += "Ranking criteria should prioritize: "
        feedback += "(1) direct answers, (2) supporting details, (3) related context."

        return feedback


# =============================================================================
# Utility: Create adapter with standard configuration
# =============================================================================

def create_reranker_adapter(
    reranker_module,
    evaluator,
    top_k: int = 10,
) -> RerankerAdapter:
    """
    Factory function to create RerankerAdapter.

    Args:
        reranker_module: RerankerModule instance
        evaluator: RAGASEvaluator instance
        top_k: Number of top documents to evaluate

    Returns:
        Configured RerankerAdapter
    """
    return RerankerAdapter(
        reranker_module=reranker_module,
        evaluator=evaluator,
        top_k=top_k,
        failure_score=0.0,
    )
