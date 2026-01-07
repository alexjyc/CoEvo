"""
Module 3: Generator

Components:
- GeneratorModule: LLM-based answer generation (optimizable prompt)

Optimization Target: Generation prompt
Evaluation Metrics: faithfulness, answer_relevancy, answer_correctness
"""

from modules.generator.generator import AnswerGenerationResponse, GeneratorModule

__all__ = [
    "AnswerGenerationResponse",
    "GeneratorModule",
]
