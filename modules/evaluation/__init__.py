"""
Evaluation Module for RAG Pipeline

Provides RAGAS-based and custom metrics for evaluating each module.

Module 1 Metrics: context_precision, context_recall, retrieval_f1
Module 2 Metrics: context_relevancy, answer_relevancy (post-rerank)
Module 3 Metrics: faithfulness, answer_relevancy, answer_correctness
"""

from modules.evaluation.metrics import (
    GenerationMetrics,
    RerankerMetrics,
    RetrievalMetrics,
    calculate_f1,
    calculate_map,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision,
    calculate_recall,
)
from modules.evaluation.ragas_eval import RAGASEvaluator

__all__ = [
    "GenerationMetrics",
    "RAGASEvaluator",
    "RerankerMetrics",
    "RetrievalMetrics",
    "calculate_f1",
    "calculate_map",
    "calculate_mrr",
    "calculate_ndcg",
    "calculate_precision",
    "calculate_recall",
]
