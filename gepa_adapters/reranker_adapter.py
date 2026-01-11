"""
Reranker GEPA Adapter (Module 2)

Optimizes listwise reranking prompts to improve document ordering.

Module I/O:
- Input: RerankerInput(query: str, documents: List[str], feedback: Optional[str])
- Output: RerankerOutput(status, ranked_documents: List[str], scores: List[float], metadata)

Target Metrics: context_precision, context_recall (post-rerank improvement)
Optimization Goal: Surface relevant documents to top positions

GEPA Optimization Notes:
- Uses metric-based diagnostic feedback to avoid overfitting
- Analyzes ranking behavior (position changes, length bias, query overlap)
- Provides strategy-focused suggestions based on NDCG/MRR patterns
- NO ground truth in reflection records
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

    Key Features for GEPA Optimization:
    - Metric-based diagnostic feedback (no ground truth in reflection)
    - Ranking behavior analysis (position changes, length bias)
    - NDCG/MRR pattern diagnosis for strategy suggestions
    - Query-document term overlap analysis

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

    # -------------------------------------------------------------------------
    # Ranking Behavior Analysis (for diagnostic feedback without ground truth)
    # -------------------------------------------------------------------------

    def _analyze_ranking_behavior(
        self,
        original_docs: list[str],
        ranked_docs: list[str],
        query: str,
        metrics: dict[str, float],
    ) -> dict[str, Any]:
        """
        Analyze ranking patterns without ground truth.

        Args:
            original_docs: Documents in original order
            ranked_docs: Documents in reranked order
            query: The query used for ranking
            metrics: Evaluation metrics

        Returns:
            Dict with ranking behavior analysis
        """
        # Calculate position changes
        position_changes = self._calculate_position_changes(original_docs, ranked_docs)

        # Detect length bias
        length_bias = self._detect_length_bias(ranked_docs)

        # Analyze query term overlap in top-ranked docs
        query_overlap = self._analyze_query_overlap(query, ranked_docs[:5])

        # Stability analysis
        top_k_changed = sum(1 for i, doc in enumerate(ranked_docs[:5])
                          if i < len(original_docs) and original_docs[i] != doc)

        return {
            "avg_position_change": sum(abs(c) for c in position_changes) / max(len(position_changes), 1),
            "max_position_change": max(abs(c) for c in position_changes) if position_changes else 0,
            "length_bias": length_bias,
            "top_5_query_overlap": query_overlap,
            "top_5_docs_changed": top_k_changed,
            "total_docs": len(original_docs),
        }

    def _calculate_position_changes(
        self,
        original: list[str],
        ranked: list[str],
    ) -> list[int]:
        """
        Calculate how much each document moved in ranking.

        Args:
            original: Original document order
            ranked: Reranked document order

        Returns:
            List of position changes (positive = moved up, negative = moved down)
        """
        # Create hash-based lookup for original positions
        original_positions = {}
        for i, doc in enumerate(original):
            doc_key = self._get_doc_key(doc)
            if doc_key not in original_positions:
                original_positions[doc_key] = i

        position_changes = []
        for new_pos, doc in enumerate(ranked):
            doc_key = self._get_doc_key(doc)
            if doc_key in original_positions:
                old_pos = original_positions[doc_key]
                change = old_pos - new_pos  # Positive = moved up
                position_changes.append(change)

        return position_changes

    def _detect_length_bias(self, ranked_docs: list[str]) -> float:
        """
        Detect if ranking favors longer documents.

        Args:
            ranked_docs: Documents in reranked order

        Returns:
            Bias score (>0.5 means longer docs favored, <0.5 means shorter favored)
        """
        if len(ranked_docs) < 2:
            return 0.5

        # Compare average length of top half vs bottom half
        mid = len(ranked_docs) // 2
        top_half_lengths = [len(doc.split()) for doc in ranked_docs[:mid]]
        bottom_half_lengths = [len(doc.split()) for doc in ranked_docs[mid:]]

        avg_top = sum(top_half_lengths) / len(top_half_lengths) if top_half_lengths else 0
        avg_bottom = sum(bottom_half_lengths) / len(bottom_half_lengths) if bottom_half_lengths else 0

        if avg_top + avg_bottom == 0:
            return 0.5

        # Return ratio (>0.5 means top-ranked docs are longer on average)
        return avg_top / (avg_top + avg_bottom)

    def _analyze_query_overlap(self, query: str, top_docs: list[str]) -> float:
        """
        Analyze how well top-ranked documents overlap with query terms.

        Args:
            query: The search query
            top_docs: Top-ranked documents

        Returns:
            Average query term overlap score (0-1)
        """
        if not query or not top_docs:
            return 0.0

        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0

        overlaps = []
        for doc in top_docs:
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms) / len(query_terms)
            overlaps.append(overlap)

        return sum(overlaps) / len(overlaps)

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
        2. Ranking behavior analysis (position changes, length bias)
        3. Metric-based NDCG/MRR diagnosis
        4. Strategy-focused improvement suggestions

        Returns:
            Dict with Inputs, Generated Outputs, Ranking Analysis, Feedback, Score
        """
        data = trajectory["data"]
        module_input = trajectory["module_input"]
        module_output = trajectory["module_output"]
        score = trajectory["score"]
        metrics = trajectory["metrics"]

        # Extract key information
        query = module_input.get("query", data["query"])
        docs = module_input.get("documents", [])
        ranked = module_output.get("ranked_documents", [])

        # Analyze ranking behavior
        ranking_analysis = self._analyze_ranking_behavior(docs, ranked, query, metrics)

        # Format inputs
        inputs_text = f"Query: {query}\n\nDocuments to rerank ({len(docs)} total):\n"
        for i, doc in enumerate(docs[:5], 1):
            doc_preview = doc[:150] + "..." if len(doc) > 150 else doc
            inputs_text += f"  [{i}] {doc_preview}\n"
        if len(docs) > 5:
            inputs_text += f"  ... and {len(docs) - 5} more\n"

        # Format outputs with ranking analysis
        outputs_text = f"Reranked order (top {len(ranked)}):\n"
        for i, doc in enumerate(ranked[:5], 1):
            doc_preview = doc[:150] + "..." if len(doc) > 150 else doc
            outputs_text += f"  [{i}] {doc_preview}\n"

        outputs_text += f"\nRanking Analysis: "
        outputs_text += f"Avg position change={ranking_analysis['avg_position_change']:.1f}, "
        outputs_text += f"Length bias={ranking_analysis['length_bias']:.2f}, "
        outputs_text += f"Query overlap={ranking_analysis['top_5_query_overlap']:.0%}"

        # Generate diagnostic feedback (no ground truth)
        feedback = self._generate_rich_feedback(
            score=score,
            metrics=metrics,
            original_docs=docs,
            ranked_docs=ranked,
            query=query,
            ranking_analysis=ranking_analysis,
        )

        return {
            "Inputs": inputs_text,
            "Generated Outputs": outputs_text,
            "Feedback": feedback,
            "Score": f"{score:.3f}",
        }

    def _generate_rich_feedback(
        self,
        score: float,
        metrics: dict[str, float],
        original_docs: list[str],
        ranked_docs: list[str],
        query: str,
        ranking_analysis: dict[str, Any],
    ) -> str:
        """
        Generate diagnostic feedback for GEPA reflection (no ground truth).

        Uses metric patterns and ranking behavior analysis to provide
        strategy-focused improvement suggestions.

        Args:
            score: The primary metric score
            metrics: All evaluation metrics
            original_docs: Documents in original order
            ranked_docs: Documents in reranked order
            query: The search query
            ranking_analysis: Analysis of ranking behavior

        Returns:
            Diagnostic feedback string with strategy suggestions
        """
        ndcg = metrics.get("ndcg", 0)
        mrr = metrics.get("mrr", 0)
        precision_at_5 = metrics.get("precision_at_5", 0)

        # Determine performance tier and generate appropriate feedback
        if score >= 0.5:
            return self._positive_feedback_diagnostic(score, metrics, ranking_analysis)
        elif score >= 0.2:
            return self._partial_feedback_diagnostic(score, metrics, ranking_analysis)
        else:
            return self._negative_feedback_diagnostic(score, metrics, ranking_analysis)

    def _positive_feedback_diagnostic(
        self,
        score: float,
        metrics: dict[str, float],
        ranking_analysis: dict[str, Any],
    ) -> str:
        """Generate positive feedback with reinforcement of what worked."""
        ndcg = metrics.get("ndcg", 0)
        mrr = metrics.get("mrr", 0)

        feedback = f"GOOD: Score={score:.2f}. "
        feedback += f"NDCG={ndcg:.2f}, MRR={mrr:.2f}. "
        feedback += "Ranking effectively prioritized relevant documents. "

        # Explain what worked based on ranking analysis
        if ranking_analysis["top_5_query_overlap"] > 0.6:
            feedback += "Strong query-document term alignment in top results. "
        if ranking_analysis["avg_position_change"] > 2:
            feedback += "Significant reordering improved document placement. "

        feedback += "Continue this ranking approach for similar queries."

        return feedback

    def _partial_feedback_diagnostic(
        self,
        score: float,
        metrics: dict[str, float],
        ranking_analysis: dict[str, Any],
    ) -> str:
        """Generate feedback for partial success with strategy-focused improvements."""
        ndcg = metrics.get("ndcg", 0)
        mrr = metrics.get("mrr", 0)
        precision_at_5 = metrics.get("precision_at_5", 0)

        feedback = f"PARTIAL: Score={score:.2f}. "
        feedback += f"NDCG={ndcg:.2f}, MRR={mrr:.2f}, Precision@5={precision_at_5:.1%}. "

        # Diagnose based on metric patterns
        if mrr < 0.5:
            feedback += "PATTERN: First relevant document not in top positions. "
            feedback += "TRY: Prioritize documents that directly answer the question. "

        if ndcg < mrr * 0.8:
            feedback += "PATTERN: Good first result but poor overall ranking. "
            feedback += "TRY: Ensure diverse relevant documents are surfaced. "

        # Ranking behavior-based diagnosis
        if ranking_analysis["length_bias"] > 0.65:
            feedback += f"Note: Ranking favors longer documents (bias={ranking_analysis['length_bias']:.2f}). "
            feedback += "STRATEGY: Weight content relevance over document length. "
        elif ranking_analysis["length_bias"] < 0.35:
            feedback += f"Note: Ranking favors shorter documents (bias={ranking_analysis['length_bias']:.2f}). "
            feedback += "STRATEGY: Consider comprehensive documents even if longer. "

        if ranking_analysis["top_5_query_overlap"] < 0.4:
            feedback += f"ISSUE: Low query-document overlap ({ranking_analysis['top_5_query_overlap']:.0%}) in top results. "
            feedback += "STRATEGY: Prioritize documents with query term matches. "

        return feedback

    def _negative_feedback_diagnostic(
        self,
        score: float,
        metrics: dict[str, float],
        ranking_analysis: dict[str, Any],
    ) -> str:
        """Generate detailed negative feedback with strategy-focused suggestions."""
        ndcg = metrics.get("ndcg", 0)
        mrr = metrics.get("mrr", 0)
        precision_at_5 = metrics.get("precision_at_5", 0)

        feedback = f"POOR: Score={score:.2f}. "
        feedback += f"NDCG={ndcg:.2f}, MRR={mrr:.2f}, Precision@5={precision_at_5:.1%}. "
        feedback += "Ranking failed to surface relevant documents. "

        # Diagnose primary failure mode
        if mrr < 0.2:
            feedback += "CRITICAL: First relevant document ranked very low. "
        if precision_at_5 < 0.2:
            feedback += "CRITICAL: Very few relevant documents in top 5. "

        # Ranking behavior-based diagnosis
        if ranking_analysis["avg_position_change"] < 1:
            feedback += "ISSUE: Minimal reordering occurred - ranking may be too conservative. "
            feedback += "STRATEGY: Be more aggressive in promoting relevant-looking documents. "

        if ranking_analysis["length_bias"] > 0.7:
            feedback += "ISSUE: Strong length bias may cause ranking errors. "
            feedback += "STRATEGY: Focus on content relevance, not document length. "

        if ranking_analysis["top_5_query_overlap"] < 0.3:
            feedback += f"ISSUE: Top documents have poor query alignment ({ranking_analysis['top_5_query_overlap']:.0%}). "

        # General recovery strategies
        feedback += "STRATEGIES: "
        feedback += "(1) Prioritize documents with direct query term matches, "
        feedback += "(2) Look for documents that answer the question type (factoid, list, comparison), "
        feedback += "(3) Consider semantic relevance beyond exact term matching."

        return feedback

    def _positive_feedback(self, score: float, metrics: dict[str, float]) -> str:
        """Generate positive feedback for effective reranking."""
        dummy_analysis = {
            "avg_position_change": 0,
            "length_bias": 0.5,
            "top_5_query_overlap": 0,
        }
        return self._generate_rich_feedback(score, metrics, [], [], "", dummy_analysis)

    def _negative_feedback(self, score: float, metrics: dict[str, float]) -> str:
        """Generate improvement suggestions for ineffective reranking."""
        dummy_analysis = {
            "avg_position_change": 0,
            "length_bias": 0.5,
            "top_5_query_overlap": 0,
        }
        return self._generate_rich_feedback(score, metrics, [], [], "", dummy_analysis)


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
