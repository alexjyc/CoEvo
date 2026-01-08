"""
Reranker GEPA Adapter (Module 2)

Optimizes listwise reranking prompts to improve document ordering.

Module I/O:
- Input: RerankerInput(query: str, documents: List[str], feedback: Optional[str])
- Output: RerankerOutput(status, ranked_documents: List[str], scores: List[float], metadata)

Target Metrics: context_precision, context_recall (post-rerank improvement)
Optimization Goal: Surface relevant documents to top positions

GEPA Optimization Notes:
- Reflective feedback includes ground truth context for better prompt evolution
- Shows what documents should have been ranked higher
- Provides contrastive examples for ranking criteria
"""

# Import module types
import hashlib
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

    def _get_doc_key(self, doc: str) -> str:
        """
        Generate a unique key for a document for reliable matching.

        Uses MD5 hash of normalized content for collision-resistant matching.
        This is more reliable than truncation-based keys which can fail on
        documents with similar prefixes.

        Args:
            doc: Document content string

        Returns:
            Hash-based key string
        """
        if not doc:
            return ""
        # Normalize whitespace and compute hash
        normalized = " ".join(doc.split())
        return hashlib.md5(normalized.encode()).hexdigest()

    async def _run_single_async(
        self,
        data: RAGDataInst,
    ) -> tuple[RAGRolloutOutput, dict[str, Any], dict[str, Any]]:
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
            return (
                {
                    "result": None,
                    "success": False,
                    "error": "No documents provided for reranking",
                },
                {"query": query, "documents": []},
                {},
            )

        # Run reranker
        reranker_input = RerankerInput(
            query=query,
            documents=documents,
        )
        reranker_output = await self.module.run(reranker_input)

        # Package output
        ranked_docs = reranker_output.ranked_documents[: self.top_k]

        # Map reranked documents to their chunk indices
        # Uses content hash for more reliable matching (handles near-duplicates better)
        reranked_chunk_indices = []
        if pre_rerank_chunk_indices:
            doc_to_idx = {}
            for i, (doc, idx) in enumerate(zip(documents, pre_rerank_chunk_indices)):
                # Use content hash as key - more reliable than truncation
                doc_key = self._get_doc_key(doc)
                if doc_key not in doc_to_idx:
                    doc_to_idx[doc_key] = idx

            for doc in ranked_docs:
                doc_key = self._get_doc_key(doc)
                if doc_key in doc_to_idx:
                    reranked_chunk_indices.append(doc_to_idx[doc_key])
                else:
                    reranked_chunk_indices.append(-1)  # Unknown index

        output: RAGRolloutOutput = {
            "result": {
                "ranked_documents": ranked_docs,
                "scores": reranker_output.scores[: self.top_k],
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
            "scores": reranker_output.scores[: self.top_k],
            "original_order": documents,
            "reranked_chunk_indices": reranked_chunk_indices,
            "pre_rerank_chunk_indices": pre_rerank_chunk_indices,
        }

        return output, module_input, module_output

    async def _compute_score_async(
        self,
        data: RAGDataInst,
        module_output: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
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
            ground_truth=gt_dict if (relevant_chunk_indices or ground_truth) else None,
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
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
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
    ) -> dict[str, Any]:
        """
        Format trajectory into GEPA reflection record with rich feedback.

        GEPA Best Practices Applied:
        1. Include ground truth answer so reflection LLM knows what to prioritize
        2. Show which documents should have been ranked higher
        3. Provide specific, actionable ranking criteria

        Returns:
            Dict with Inputs, Generated Outputs, Expected Answer, Feedback keys
        """
        data = trajectory["data"]
        module_input = trajectory["module_input"]
        module_output = trajectory["module_output"]
        score = trajectory["score"]
        metrics = trajectory["metrics"]

        # Extract key information
        query = module_input.get("query", data["query"])
        docs = module_input.get("documents", [])
        ground_truth = data.get("ground_truth", "")

        # Format inputs
        inputs_text = f"Query: {query}\n\nDocuments to rerank ({len(docs)} total):\n"
        for i, doc in enumerate(docs[:5], 1):  # Show first 5
            doc_preview = doc[:150] + "..." if len(doc) > 150 else doc
            inputs_text += f"  [{i}] {doc_preview}\n"
        if len(docs) > 5:
            inputs_text += f"  ... and {len(docs) - 5} more\n"

        # Format outputs
        ranked = module_output.get("ranked_documents", [])
        outputs_text = f"Reranked order (top {len(ranked)}):\n"
        for i, doc in enumerate(ranked[:5], 1):
            doc_preview = doc[:150] + "..." if len(doc) > 150 else doc
            outputs_text += f"  [{i}] {doc_preview}\n"

        # Format expected answer (for context)
        if ground_truth:
            gt_preview = ground_truth[:300] + "..." if len(ground_truth) > 300 else ground_truth
            expected_text = f"Expected Answer: {gt_preview}"
        else:
            expected_text = "Expected Answer: Not available"

        # Generate rich feedback with ground truth context
        feedback = self._generate_rich_feedback(score, metrics, ground_truth, docs, ranked)

        return {
            "Inputs": inputs_text,
            "Generated Outputs": outputs_text,
            "Expected Answer": expected_text,
            "Feedback": feedback,
            "Score": f"{score:.3f}",
        }

    def _generate_rich_feedback(
        self,
        score: float,
        metrics: dict[str, float],
        ground_truth: str,
        original_docs: list[str],
        ranked_docs: list[str],
    ) -> str:
        """Generate rich feedback with contrastive examples for reranking."""
        ndcg = metrics.get("ndcg", 0)
        mrr = metrics.get("mrr", 0)
        precision_at_5 = metrics.get("precision_at_5", 0)

        # Determine performance tier (adaptive thresholds)
        if score >= 0.5:
            feedback = f"GOOD: Score={score:.2f}. "
            feedback += f"NDCG={ndcg:.2f}, MRR={mrr:.2f}. "
            feedback += "Ranking effectively prioritized relevant documents. "
            if ground_truth:
                feedback += f"Documents containing '{ground_truth[:100]}...' ranked appropriately."
        elif score >= 0.2:
            feedback = f"PARTIAL: Score={score:.2f}. "
            feedback += f"NDCG={ndcg:.2f}, Precision@5={precision_at_5:.1%}. "
            if mrr < 0.5:
                feedback += "First relevant document not in top positions. "
            if ground_truth:
                feedback += f"PRIORITIZE documents mentioning: '{ground_truth[:150]}...'. "
            feedback += "TRY: Weight documents with direct answers higher than context."
        else:
            feedback = f"POOR: Score={score:.2f}. "
            feedback += "Ranking failed to surface relevant documents. "
            if ground_truth:
                feedback += f"MUST PRIORITIZE content about: '{ground_truth[:150]}...'. "
            feedback += "CRITERIA: (1) Contains answer, (2) Addresses query directly, (3) Has supporting facts."

        return feedback

    def _positive_feedback(self, score: float, metrics: dict[str, float]) -> str:
        """Generate positive feedback for effective reranking"""
        return self._generate_rich_feedback(score, metrics, "", [], [])

    def _negative_feedback(self, score: float, metrics: dict[str, float]) -> str:
        """Generate improvement suggestions for ineffective reranking"""
        return self._generate_rich_feedback(score, metrics, "", [], [])


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
