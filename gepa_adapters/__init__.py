"""
GEPA Adapters for Modular RAG Pipeline

Custom adapters implementing the GEPAAdapter protocol for each RAG module:
- QueryPlannerAdapter: Optimizes query decomposition/reformulation prompts
- RerankerAdapter: Optimizes listwise reranking prompts
- GeneratorAdapter: Optimizes answer generation prompts

Each adapter:
1. Inherits from GEPA's GEPAAdapter protocol
2. Works directly with gepa.optimize() function
3. Uses RAGAS metrics for scoring
4. Generates reflective datasets for prompt evolution

Usage:
    from gepa_adapters import QueryPlannerAdapter, optimize_prompt

    # Create adapter
    adapter = QueryPlannerAdapter(planner_module, retriever, evaluator)

    # Run optimization (uses gepa.optimize internally)
    result = optimize_prompt(
        adapter=adapter,
        trainset=training_data,
        valset=validation_data,
        max_metric_calls=50,
    )

    # Apply optimized prompt
    planner_module.prompt = result['best_prompt']

    # Or use gepa.optimize directly
    from gepa import optimize
    result = optimize(
        seed_candidate={"query_planner_prompt": seed_prompt},
        trainset=data,
        adapter=adapter,
        reflection_lm="openai/gpt-4o-mini",
    )
"""

from gepa_adapters.base import (
    RAGDataInst,
    RAGTrajectory,
    RAGRolloutOutput,
    EvaluationBatch,
    RAGModuleAdapter,
    optimize_prompt,
    GEPA_AVAILABLE,
)
from gepa_adapters.query_planner_adapter import QueryPlannerAdapter
from gepa_adapters.reranker_adapter import RerankerAdapter
from gepa_adapters.generator_adapter import GeneratorAdapter

__all__ = [
    # Base types
    "RAGDataInst",
    "RAGTrajectory",
    "RAGRolloutOutput",
    "EvaluationBatch",
    "RAGModuleAdapter",
    # Optimization function
    "optimize_prompt",
    "GEPA_AVAILABLE",
    # Module adapters
    "QueryPlannerAdapter",
    "RerankerAdapter",
    "GeneratorAdapter",
]
