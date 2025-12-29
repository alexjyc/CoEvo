"""
Evaluation Module for RAG Pipeline

Provides RAGAS-based and custom metrics for evaluating each module.

Module 1 Metrics: context_precision, context_recall, retrieval_f1
Module 2 Metrics: context_relevancy, answer_relevancy (post-rerank)
Module 3 Metrics: faithfulness, answer_relevancy, answer_correctness
"""

from modules.evaluation.ragas_eval import RAGASEvaluator
from modules.evaluation.metrics import (
    calculate_f1,
    calculate_precision,
    calculate_recall,
    calculate_mrr,
    calculate_ndcg,
    calculate_map,
    RetrievalMetrics,
    RerankerMetrics,
    GenerationMetrics,
)

__all__ = [
    "RAGASEvaluator",
    "calculate_f1",
    "calculate_precision",
    "calculate_recall",
    "calculate_mrr",
    "calculate_ndcg",
    "calculate_map",
    "RetrievalMetrics",
    "RerankerMetrics",
    "GenerationMetrics",
]
