"""
Modular RAG Pipeline Components

Each module is independently testable and optimizable using GEPA.

Architecture:
    Query → [Module 1: Query Planner + Retrieval] → Documents
         ↓
         → [Module 2: Reranker] → Refined Documents
         ↓
         → [Module 3: Generator] → Answer

Each module:
- Is independently testable
- Has an optimizable prompt (via GEPA)
- Is evaluated with RAGAS metrics
"""

# Base types
from modules.base import (
    Module,
    ModuleType,
    ModuleInput,
    ModuleOutput,
    ModuleEvaluator,
    EvaluationResult,
    QueryPlannerInput,
    QueryPlannerOutput,
    RetrievalInput,
    RetrievalOutput,
    RerankerInput,
    RerankerOutput,
    GeneratorInput,
    GeneratorOutput,
    PipelineConfig,
)

# Module implementations
from modules.query_planner import QueryPlannerModule, HybridRetriever
from modules.reranker import RerankerModule
from modules.generator import GeneratorModule
from modules.preprocessor import DocumentPreprocessor

# Evaluation
from modules.evaluation import RAGASEvaluator

# Pipeline
from modules.pipeline import ModularRAGPipeline, PipelineResult

__all__ = [
    # Base types
    "Module",
    "ModuleType",
    "ModuleInput",
    "ModuleOutput",
    "ModuleEvaluator",
    "EvaluationResult",
    "QueryPlannerInput",
    "QueryPlannerOutput",
    "RetrievalInput",
    "RetrievalOutput",
    "RerankerInput",
    "RerankerOutput",
    "GeneratorInput",
    "GeneratorOutput",
    "PipelineConfig",
    # Modules
    "QueryPlannerModule",
    "HybridRetriever",
    "RerankerModule",
    "GeneratorModule",
    "DocumentPreprocessor",
    # Evaluation
    "RAGASEvaluator",
    # Pipeline
    "ModularRAGPipeline",
    "PipelineResult",
]
