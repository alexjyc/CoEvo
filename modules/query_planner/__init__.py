"""
Module 1: Query Planner + Retrieval

Components:
- QueryPlannerModule: LLM-based query decomposition/reformulation (optimizable prompt)
- HybridRetriever: Dense + BM25 with wRRF fusion (algorithmic, no prompt)

Optimization Target: Query planning prompt
Evaluation Metrics: context_precision, context_recall
"""

from modules.query_planner.planner import QueryPlannerModule, QueryPlannerResponse
from modules.query_planner.retrieval import HybridRetriever

__all__ = [
    "QueryPlannerModule",
    "QueryPlannerResponse",
    "HybridRetriever",
]
