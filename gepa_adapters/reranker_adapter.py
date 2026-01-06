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
        3. Return reranked documents with chunk indices for evaluation

        Args:
            data: RAGDataInst with query, contexts (pre-retrieved docs),
                  and optionally context_chunk_indices (for deterministic eval)

        Returns:
            Tuple of (output, module_input, module_output)
        """
        query = data["query"]
        documents = data.get("contexts", [])
        # Get chunk indices for deterministic evaluation
        pre_rerank_chunk_indices = data.get("context_chunk_indices", [])

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

        # Map reranked documents to their chunk indices
        # Build doc -> index mapping from original order
        reranked_chunk_indices = []
        if pre_rerank_chunk_indices:
            doc_to_idx = {}
            for i, (doc, idx) in enumerate(zip(documents, pre_rerank_chunk_indices)):
                # Use first 200 chars as key to handle duplicates
                doc_key = doc[:200] if doc else ""
                if doc_key not in doc_to_idx:
                    doc_to_idx[doc_key] = idx

            for doc in ranked_docs:
                doc_key = doc[:200] if doc else ""
                if doc_key in doc_to_idx:
                    reranked_chunk_indices.append(doc_to_idx[doc_key])
                else:
                    reranked_chunk_indices.append(-1)  # Unknown index

        output: RAGRolloutOutput = {
            "result": {
                "ranked_documents": ranked_docs,
                "scores": reranker_output.scores[:self.top_k],
                "num_input": len(documents),
                "num_output": len(ranked_docs),
                "reranked_chunk_indices": reranked_chunk_indices,
            },
            "success": reranker_output.status == "success",
            "error": reranker_output.error_message,
        }

        module_input = {
            "query": query,
            "documents": documents,
            "num_documents": len(documents),
            "pre_rerank_chunk_indices": pre_rerank_chunk_indices,
        }

        module_output = {
            "ranked_documents": ranked_docs,
            "scores": reranker_output.scores[:self.top_k],
            "original_order": documents,
            "reranked_chunk_indices": reranked_chunk_indices,
            "pre_rerank_chunk_indices": pre_rerank_chunk_indices,
        }

        return output, module_input, module_output

    async def _compute_score_async(
        self,
        data: RAGDataInst,
        module_output: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reranking quality score using DETERMINISTIC ID-based metrics.

        Uses order-aware metrics (NDCG, MRR, Precision@K) when chunk indices available.
        Falls back to LLM-based evaluation only when IDs not available.

        Args:
            data: Original input with ground_truth and relevant_chunk_indices
            module_output: Reranker output with ranked_documents and chunk_indices

        Returns:
            Tuple of (score, detailed_metrics)
        """
        ranked_docs = module_output.get("ranked_documents", [])
        original_order = module_output.get("original_order", [])
        ground_truth = data.get("ground_truth", "")
        query = data["query"]

        # Get chunk indices for deterministic evaluation
        reranked_chunk_indices = module_output.get("reranked_chunk_indices", [])
        pre_rerank_chunk_indices = module_output.get("pre_rerank_chunk_indices", [])
        relevant_chunk_indices = data.get("relevant_chunk_indices", [])

        # Build ground truth dict for evaluator
        gt_dict = {
            "query": query,
            "reference": ground_truth,
            "relevant_chunk_indices": relevant_chunk_indices,
            "pre_rerank_chunk_indices": pre_rerank_chunk_indices,
        }

        # Create output object with chunk_indices for deterministic eval
        class RerankerOutputWithIndices:
            def __init__(self, docs, indices):
                self.ranked_documents = docs
                self.chunk_indices = indices

        reranker_eval = await self.evaluator._evaluate_reranker(
            input_data=RerankerInput(query=query, documents=original_order),
            output_data=RerankerOutputWithIndices(ranked_docs, reranked_chunk_indices),
            ground_truth=gt_dict if (relevant_chunk_indices or ground_truth) else None
        )

        # Use NDCG as primary score (order-aware, deterministic)
        # Falls back to F1 if NDCG not available (LLM-based fallback was used)
        ndcg = reranker_eval.get("rerank_ndcg", 0.0)
        mrr = reranker_eval.get("rerank_mrr", 0.0)
        precision_at_5 = reranker_eval.get("rerank_precision_at_5", 0.0)
        precision_at_10 = reranker_eval.get("rerank_precision_at_10", 0.0)

        # Fallback to LLM-based metrics if deterministic not available
        if ndcg == 0.0 and "rerank_precision" in reranker_eval:
            precision = reranker_eval.get("rerank_precision", 0.0)
            recall = reranker_eval.get("rerank_recall", 0.0)
            f1 = reranker_eval.get("rerank_f1", 0.0)
            if f1 == 0.0 and (precision > 0 or recall > 0):
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            score = f1
        else:
            # Use deterministic metrics (preferred)
            # Combined score: NDCG (60%) + MRR (20%) + Precision@5 (20%)
            score = ndcg * 0.6 + mrr * 0.2 + precision_at_5 * 0.2

        # Build metrics dict
        metrics = {
            "ndcg": ndcg,
            "mrr": mrr,
            "precision_at_5": precision_at_5,
            "precision_at_10": precision_at_10,
            "ndcg_improvement": reranker_eval.get("ndcg_improvement", 0.0),
            "mrr_improvement": reranker_eval.get("mrr_improvement", 0.0),
            "num_input": len(original_order),
            "num_output": len(ranked_docs),
            "has_chunk_indices": bool(reranked_chunk_indices),
        }

        # Add fallback LLM metrics if used
        if "rerank_precision" in reranker_eval:
            metrics["llm_precision"] = reranker_eval.get("rerank_precision", 0.0)
            metrics["llm_recall"] = reranker_eval.get("rerank_recall", 0.0)
            metrics["llm_f1"] = reranker_eval.get("rerank_f1", 0.0)

        return score, metrics

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
