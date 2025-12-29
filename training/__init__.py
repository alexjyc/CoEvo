"""
Training Module for RAG Pipeline Optimization

Implements CRAG (Corrective RAG) methodology for training data generation
and GEPA-based prompt optimization.

Components:
- CRAGRetrievalEvaluator: Assess retrieval quality (Correct/Ambiguous/Incorrect)
- CRAGTrainingGenerator: Generate training data from pipeline execution
- GEPAOptimizationRunner: Run GEPA optimization for each module
"""

from training.crag_benchmark import (
    RetrievalQuality,
    CRAGDataInst,
    ModuleTrainingExample,
    TrainingDataset,
    CRAGRetrievalEvaluator,
    CRAGTrainingGenerator,
    GEPAOptimizationRunner,
)

__all__ = [
    "RetrievalQuality",
    "CRAGDataInst",
    "ModuleTrainingExample",
    "TrainingDataset",
    "CRAGRetrievalEvaluator",
    "CRAGTrainingGenerator",
    "GEPAOptimizationRunner",
]
