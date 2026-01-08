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
    EvaluationResult,
    GeneratorInput,
    GeneratorOutput,
    Module,
    ModuleEvaluator,
    ModuleInput,
    ModuleOutput,
    ModuleType,
    PipelineConfig,
    QueryPlannerInput,
    QueryPlannerOutput,
    RerankerInput,
    RerankerOutput,
    RetrievalInput,
    RetrievalOutput,
)
from modules.chunk_optimizer import ChunkSizeOptimizer

# Evaluation
from modules.evaluation import RAGASEvaluator
from modules.generator import GeneratorModule

# Pipeline
from modules.pipeline import ModularRAGPipeline, PipelineResult
from modules.preprocessor import DocumentPreprocessor

# Module implementations
from modules.query_planner import HybridRetriever, QueryPlannerModule
from modules.reranker import RerankerModule

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
    "ChunkSizeOptimizer",
    # Evaluation
    "RAGASEvaluator",
    # Pipeline
    "ModularRAGPipeline",
    "PipelineResult",
]
