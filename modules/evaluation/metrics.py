"""
Custom Metrics for RAG Pipeline Evaluation

Provides module-specific metrics that complement RAGAS evaluation.
"""

from typing import List, Dict, Any, Set


def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall"""
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0.0


def calculate_precision(retrieved: Set[int], relevant: Set[int]) -> float:
    """Calculate precision: relevant retrieved / total retrieved"""
    if not retrieved:
        return 0.0
    return len(retrieved & relevant) / len(retrieved)


def calculate_recall(retrieved: Set[int], relevant: Set[int]) -> float:
    """Calculate recall: relevant retrieved / total relevant"""
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)


def calculate_mrr(ranked_list: List[int], relevant: Set[int]) -> float:
    """
    Calculate Mean Reciprocal Rank.

    MRR = 1/rank of first relevant item (0 if none found)
    """
    for i, item in enumerate(ranked_list):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg(ranked_list: List[int], relevant: Set[int], k: int = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain.

    NDCG = DCG / IDCG where DCG = sum(rel_i / log2(i+2))
    """
    import math

    if k is None:
        k = len(ranked_list)

    ranked_list = ranked_list[:k]

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(ranked_list):
        rel = 1.0 if item in relevant else 0.0
        dcg += rel / math.log2(i + 2)

    # Calculate IDCG (ideal DCG with all relevant items at top)
    ideal_rels = [1.0] * min(len(relevant), k) + [0.0] * (k - min(len(relevant), k))
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += rel / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_map(ranked_list: List[int], relevant: Set[int]) -> float:
    """
    Calculate Mean Average Precision.

    MAP = average of precision at each relevant item position
    """
    if not relevant:
        return 0.0

    precisions = []
    relevant_found = 0

    for i, item in enumerate(ranked_list):
        if item in relevant:
            relevant_found += 1
            precision_at_i = relevant_found / (i + 1)
            precisions.append(precision_at_i)

    if not precisions:
        return 0.0

    return sum(precisions) / len(relevant)


class RetrievalMetrics:
    """Collection of retrieval quality metrics"""

    @staticmethod
    def evaluate(
        retrieved_indices: List[int],
        relevant_indices: List[int],
        k: int = None
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality with multiple metrics.

        Args:
            retrieved_indices: Ranked list of retrieved chunk indices
            relevant_indices: List of ground truth relevant indices
            k: Cutoff for metrics (default: len(retrieved_indices))

        Returns:
            Dict with precision, recall, f1, mrr, ndcg, map
        """
        if k is None:
            k = len(retrieved_indices)

        retrieved_k = retrieved_indices[:k]
        retrieved_set = set(retrieved_k)
        relevant_set = set(relevant_indices)

        precision = calculate_precision(retrieved_set, relevant_set)
        recall = calculate_recall(retrieved_set, relevant_set)
        f1 = calculate_f1(precision, recall)
        mrr = calculate_mrr(retrieved_k, relevant_set)
        ndcg = calculate_ndcg(retrieved_k, relevant_set, k)
        map_score = calculate_map(retrieved_k, relevant_set)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mrr': mrr,
            'ndcg': ndcg,
            'map': map_score,
        }


class RerankerMetrics:
    """Metrics for evaluating reranking quality"""

    @staticmethod
    def evaluate_improvement(
        pre_rerank_indices: List[int],
        post_rerank_indices: List[int],
        relevant_indices: List[int],
        k: int = None
    ) -> Dict[str, float]:
        """
        Evaluate reranking improvement over original retrieval order.

        Returns:
            Dict with pre/post metrics and improvement scores
        """
        pre_metrics = RetrievalMetrics.evaluate(pre_rerank_indices, relevant_indices, k)
        post_metrics = RetrievalMetrics.evaluate(post_rerank_indices, relevant_indices, k)

        improvement = {
            'pre_precision': pre_metrics['precision'],
            'post_precision': post_metrics['precision'],
            'precision_delta': post_metrics['precision'] - pre_metrics['precision'],

            'pre_recall': pre_metrics['recall'],
            'post_recall': post_metrics['recall'],
            'recall_delta': post_metrics['recall'] - pre_metrics['recall'],

            'pre_ndcg': pre_metrics['ndcg'],
            'post_ndcg': post_metrics['ndcg'],
            'ndcg_delta': post_metrics['ndcg'] - pre_metrics['ndcg'],

            'pre_mrr': pre_metrics['mrr'],
            'post_mrr': post_metrics['mrr'],
            'mrr_delta': post_metrics['mrr'] - pre_metrics['mrr'],
        }

        # Overall improvement score
        improvement['overall_improvement'] = (
            improvement['precision_delta'] +
            improvement['ndcg_delta'] +
            improvement['mrr_delta']
        ) / 3

        return improvement


class GenerationMetrics:
    """Metrics for evaluating generation quality (non-LLM based)"""

    @staticmethod
    def evaluate_basic(
        answer: str,
        ground_truth: str,
        context: str
    ) -> Dict[str, float]:
        """
        Basic generation metrics (without LLM evaluation).

        Returns:
            Dict with answer_length, context_usage, etc.
        """
        # Answer length relative to context
        answer_length_ratio = len(answer) / len(context) if context else 0

        # Check if answer uses terms from context
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        gt_words = set(ground_truth.lower().split()) if ground_truth else set()

        context_overlap = len(answer_words & context_words) / len(answer_words) if answer_words else 0
        gt_overlap = len(answer_words & gt_words) / len(gt_words) if gt_words else 0

        return {
            'answer_length': len(answer),
            'answer_length_ratio': answer_length_ratio,
            'context_term_overlap': context_overlap,
            'ground_truth_term_overlap': gt_overlap,
        }
