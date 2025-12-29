"""
CRAG Benchmark Training Loop

Implements Corrective RAG (CRAG) methodology for training data generation
and integrates with GEPA for per-module prompt optimization.

CRAG Components:
1. Retrieval Evaluator: Assess quality of retrieved documents
2. Corrective Actions: Correct / Ambiguous / Incorrect routing
3. Decompose-Recompose: Filter irrelevant information
4. Training Data Generation: Create examples for GEPA optimization

Reference: https://github.com/HuskyInSalt/CRAG
Paper: https://arxiv.org/abs/2401.15884
"""

import asyncio
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


# =============================================================================
# CRAG Types and Enums
# =============================================================================

class RetrievalQuality(Enum):
    """CRAG retrieval quality assessment"""
    CORRECT = "correct"        # High confidence, use retrieved docs
    AMBIGUOUS = "ambiguous"    # Medium confidence, augment with search
    INCORRECT = "incorrect"    # Low confidence, discard and search


@dataclass
class CRAGDataInst:
    """
    CRAG training instance with quality labels.

    Fields:
        query: User query
        ground_truth: Expected answer
        retrieved_docs: Documents from retrieval
        relevant_chunk_indices: Ground truth relevant chunks
        retrieval_quality: CRAG quality assessment
        corrective_action: What correction was needed
        final_docs: Documents after correction
        final_answer: Answer after correction
        metrics: Evaluation metrics
    """
    query: str
    ground_truth: Optional[str] = None
    retrieved_docs: List[str] = field(default_factory=list)
    relevant_chunk_indices: List[int] = field(default_factory=list)
    retrieval_quality: RetrievalQuality = RetrievalQuality.AMBIGUOUS
    corrective_action: str = ""
    final_docs: List[str] = field(default_factory=list)
    final_answer: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModuleTrainingExample:
    """Training example for a specific module"""
    module_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    score: float
    quality_label: str  # "good" / "bad" / "needs_improvement"
    feedback: str
    correction_applied: str = ""


@dataclass
class TrainingDataset:
    """Collection of training examples per module"""
    query_planner_examples: List[ModuleTrainingExample] = field(default_factory=list)
    reranker_examples: List[ModuleTrainingExample] = field(default_factory=list)
    generator_examples: List[ModuleTrainingExample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_planner": [asdict(e) for e in self.query_planner_examples],
            "reranker": [asdict(e) for e in self.reranker_examples],
            "generator": [asdict(e) for e in self.generator_examples],
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        """Save dataset to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# =============================================================================
# CRAG Retrieval Evaluator
# =============================================================================

class CRAGRetrievalEvaluator:
    """
    Evaluates retrieval quality and assigns CRAG labels.

    Confidence thresholds:
    - CORRECT: confidence > 0.8
    - AMBIGUOUS: 0.3 <= confidence <= 0.8
    - INCORRECT: confidence < 0.3
    """

    def __init__(
        self,
        correct_threshold: float = 0.8,
        incorrect_threshold: float = 0.3,
    ):
        self.correct_threshold = correct_threshold
        self.incorrect_threshold = incorrect_threshold

    def evaluate(
        self,
        retrieved_indices: List[int],
        relevant_indices: List[int],
    ) -> Tuple[RetrievalQuality, float]:
        """
        Evaluate retrieval quality.

        Args:
            retrieved_indices: Indices of retrieved chunks
            relevant_indices: Ground truth relevant chunk indices

        Returns:
            Tuple of (quality_label, confidence_score)
        """
        if not relevant_indices:
            return RetrievalQuality.AMBIGUOUS, 0.5

        if not retrieved_indices:
            return RetrievalQuality.INCORRECT, 0.0

        # Calculate recall (how many relevant docs were retrieved)
        retrieved_set = set(retrieved_indices)
        relevant_set = set(relevant_indices)

        hits = len(retrieved_set & relevant_set)
        recall = hits / len(relevant_set)

        # Calculate precision
        precision = hits / len(retrieved_set) if retrieved_set else 0

        # F1 as confidence
        if precision + recall > 0:
            confidence = 2 * precision * recall / (precision + recall)
        else:
            confidence = 0.0

        # Assign quality label
        if confidence >= self.correct_threshold:
            return RetrievalQuality.CORRECT, confidence
        elif confidence < self.incorrect_threshold:
            return RetrievalQuality.INCORRECT, confidence
        else:
            return RetrievalQuality.AMBIGUOUS, confidence

    def evaluate_docs(
        self,
        retrieved_doc_ids: List[str],
        reference_doc_ids: List[str],
    ) -> Tuple[RetrievalQuality, float]:
        """
        Evaluate retrieval quality at document level (preferred method).

        Args:
            retrieved_doc_ids: IDs of retrieved documents
            reference_doc_ids: Ground truth relevant document IDs

        Returns:
            Tuple of (quality_label, confidence_score)
        """
        if not reference_doc_ids:
            return RetrievalQuality.AMBIGUOUS, 0.5

        if not retrieved_doc_ids:
            return RetrievalQuality.INCORRECT, 0.0

        # Calculate recall and precision at document level
        retrieved_set = set(retrieved_doc_ids)
        reference_set = set(reference_doc_ids)

        hits = len(retrieved_set & reference_set)
        recall = hits / len(reference_set)
        precision = hits / len(retrieved_set) if retrieved_set else 0

        # F1 as confidence
        if precision + recall > 0:
            confidence = 2 * precision * recall / (precision + recall)
        else:
            confidence = 0.0

        # Assign quality label
        if confidence >= self.correct_threshold:
            return RetrievalQuality.CORRECT, confidence
        elif confidence < self.incorrect_threshold:
            return RetrievalQuality.INCORRECT, confidence
        else:
            return RetrievalQuality.AMBIGUOUS, confidence


# =============================================================================
# CRAG Training Data Generator
# =============================================================================

class CRAGTrainingGenerator:
    """
    Generates training data using CRAG methodology.

    Process:
    1. Run pipeline on evaluation queries
    2. Assess retrieval quality using CRAG evaluator
    3. Apply corrective actions based on quality
    4. Generate training examples with quality labels
    5. Create module-specific datasets for GEPA
    """

    def __init__(
        self,
        pipeline,  # ModularRAGPipeline instance
        evaluator,  # RAGASEvaluator instance
        retrieval_evaluator: Optional[CRAGRetrievalEvaluator] = None,
    ):
        self.pipeline = pipeline
        self.evaluator = evaluator
        self.retrieval_evaluator = retrieval_evaluator or CRAGRetrievalEvaluator()

    async def generate_training_data(
        self,
        evaluation_queries: List[Dict[str, Any]],
        relevance_labels: Dict[int, List[int]],
        output_dir: Path,
    ) -> TrainingDataset:
        """
        Generate training dataset from evaluation queries.

        Args:
            evaluation_queries: List of query dicts with ground_truth
            relevance_labels: Mapping of query_id to relevant chunk indices
            output_dir: Directory to save training data

        Returns:
            TrainingDataset with examples for each module
        """
        dataset = TrainingDataset(
            metadata={
                "generated_at": datetime.now().isoformat(),
                "num_queries": len(evaluation_queries),
            }
        )

        print(f"Generating CRAG training data from {len(evaluation_queries)} queries...")

        for query_data in evaluation_queries:
            query_id = query_data.get("query_id", 0)
            query = query_data["query"]
            ground_truth = query_data.get("ground_truth", "")
            relevant_indices = relevance_labels.get(query_id, [])

            try:
                # Run full pipeline
                result = await self.pipeline.run(
                    query=query,
                    ground_truth=ground_truth,
                    relevant_chunk_indices=relevant_indices,
                )

                # Assess retrieval quality using chunk-level metrics
                retrieved_indices = result.metadata.get("retrieved_chunk_indices", [])
                quality, confidence = self.retrieval_evaluator.evaluate(
                    retrieved_indices, relevant_indices
                )

                # Create CRAG instance
                crag_instance = CRAGDataInst(
                    query=query,
                    ground_truth=ground_truth,
                    retrieved_docs=result.contexts,
                    relevant_chunk_indices=relevant_indices,
                    retrieval_quality=quality,
                    final_docs=result.ranked_documents,
                    final_answer=result.answer,
                    metrics=result.metrics,
                )

                # Generate module-specific examples
                self._add_query_planner_example(dataset, crag_instance, result)
                self._add_reranker_example(dataset, crag_instance, result)
                self._add_generator_example(dataset, crag_instance, result)

            except Exception as e:
                print(f"Error processing query {query_id}: {e}")
                continue

        # Add statistics to metadata
        dataset.metadata.update({
            "num_query_planner_examples": len(dataset.query_planner_examples),
            "num_reranker_examples": len(dataset.reranker_examples),
            "num_generator_examples": len(dataset.generator_examples),
        })

        # Save dataset
        dataset.save(output_dir / "crag_training_data.json")
        print(f"Saved training data to {output_dir}")

        return dataset

    def _add_query_planner_example(
        self,
        dataset: TrainingDataset,
        crag_inst: CRAGDataInst,
        result,
    ) -> None:
        """Create query planner training example based on retrieval quality"""
        retrieval_f1 = result.metrics.get("retrieval_f1", 0)

        # Determine quality label
        if crag_inst.retrieval_quality == RetrievalQuality.CORRECT:
            quality_label = "good"
            feedback = "Query decomposition/reformulation led to high-quality retrieval."
        elif crag_inst.retrieval_quality == RetrievalQuality.INCORRECT:
            quality_label = "bad"
            feedback = (
                "Query decomposition/reformulation failed to retrieve relevant documents. "
                f"Ground truth required chunks: {crag_inst.relevant_chunk_indices}. "
                "Need better query expansion or decomposition strategy."
            )
        else:
            quality_label = "needs_improvement"
            feedback = (
                "Query achieved partial retrieval success. "
                "Consider adding more specific terms or decomposing into sub-queries."
            )

        example = ModuleTrainingExample(
            module_name="query_planner",
            input_data={"query": crag_inst.query},
            output_data={
                "mode": result.metadata.get("query_mode", "unknown"),
                "num_queries": result.metadata.get("num_queries", 1),
                "num_retrieved": result.metadata.get("num_retrieved", 0),
            },
            score=retrieval_f1,
            quality_label=quality_label,
            feedback=feedback,
            correction_applied=crag_inst.corrective_action,
        )

        dataset.query_planner_examples.append(example)

    def _add_reranker_example(
        self,
        dataset: TrainingDataset,
        crag_inst: CRAGDataInst,
        result,
    ) -> None:
        """Create reranker training example based on context quality"""
        context_f1 = result.metrics.get("context_f1", 0)

        # Determine quality based on reranking improvement
        if context_f1 >= 0.7:
            quality_label = "good"
            feedback = "Reranking effectively surfaced relevant documents."
        elif context_f1 < 0.4:
            quality_label = "bad"
            feedback = (
                "Reranking failed to prioritize relevant documents. "
                f"Expected to find: {crag_inst.ground_truth[:100]}... "
                "Focus on documents with direct answers."
            )
        else:
            quality_label = "needs_improvement"
            feedback = (
                "Reranking partially effective. "
                "Consider prioritizing documents with explicit answers over tangential context."
            )

        example = ModuleTrainingExample(
            module_name="reranker",
            input_data={
                "query": crag_inst.query,
                "documents": crag_inst.retrieved_docs[:5],  # First 5 for brevity
                "num_documents": len(crag_inst.retrieved_docs),
            },
            output_data={
                "ranked_documents": crag_inst.final_docs[:5],
                "num_ranked": len(crag_inst.final_docs),
            },
            score=context_f1,
            quality_label=quality_label,
            feedback=feedback,
        )

        dataset.reranker_examples.append(example)

    def _add_generator_example(
        self,
        dataset: TrainingDataset,
        crag_inst: CRAGDataInst,
        result,
    ) -> None:
        """Create generator training example based on answer quality"""
        faithfulness = result.metrics.get("faithfulness", 0)
        correctness = result.metrics.get("answer_correctness", 0)
        generation_quality = result.metrics.get("generation_quality", 0)

        # Determine quality label
        if faithfulness >= 0.7 and correctness >= 0.7:
            quality_label = "good"
            feedback = "Answer is faithful to context and correct."
        elif faithfulness < 0.5:
            quality_label = "bad"
            feedback = (
                "Answer contains hallucinated content not in context. "
                "Only include claims supported by the provided documents."
            )
        elif correctness < 0.5:
            quality_label = "bad"
            feedback = (
                f"Answer incorrect. Expected: {crag_inst.ground_truth[:200]}... "
                "Focus on directly answering the question asked."
            )
        else:
            quality_label = "needs_improvement"
            feedback = "Answer partially correct. Improve grounding and accuracy."

        example = ModuleTrainingExample(
            module_name="generator",
            input_data={
                "query": crag_inst.query,
                "context": "\n".join(crag_inst.final_docs[:3]),  # Top 3 for brevity
            },
            output_data={
                "answer": crag_inst.final_answer,
                "ground_truth": crag_inst.ground_truth,
            },
            score=generation_quality,
            quality_label=quality_label,
            feedback=feedback,
        )

        dataset.generator_examples.append(example)


# =============================================================================
# GEPA Optimization Runner
# =============================================================================

# Import GEPA for optimization
try:
    import gepa
    GEPA_AVAILABLE = True
except ImportError:
    print("⚠️  GEPA not installed. Install with: pip install gepa-ai")
    GEPA_AVAILABLE = False


class GEPAOptimizationRunner:
    """
    Runs GEPA optimization for each module using CRAG training data.

    Uses gepa.optimize() directly for prompt evolution. This class:
    1. Converts CRAG training examples to GEPA DataInst format
    2. Calls gepa.optimize() with the appropriate adapter
    3. Saves optimized prompts and results

    Workflow per module:
    1. Load CRAG training examples
    2. Convert to GEPA DataInst format
    3. Run gepa.optimize() for prompt evolution
    4. Save optimized prompts
    """

    def __init__(
        self,
        query_planner_adapter,
        reranker_adapter,
        generator_adapter,
        output_dir: Path,
        budget: int = 100,  # Max optimization iterations
        reflection_lm: str = "openai/gpt-4o-mini",
    ):
        self.adapters = {
            "query_planner": query_planner_adapter,
            "reranker": reranker_adapter,
            "generator": generator_adapter,
        }
        self.output_dir = output_dir
        self.budget = budget
        self.reflection_lm = reflection_lm

    def optimize_module(
        self,
        module_name: str,
        training_examples: List[ModuleTrainingExample],
        validation_examples: Optional[List[ModuleTrainingExample]] = None,
    ) -> Dict[str, Any]:
        """
        Run GEPA optimization for a single module.

        Uses gepa.optimize() to evolve prompts through reflective mutation.

        Args:
            module_name: Name of module to optimize
            training_examples: CRAG training examples
            validation_examples: Optional validation set

        Returns:
            Dict with optimization results including best_prompt and best_score
        """
        if not GEPA_AVAILABLE:
            raise ImportError(
                "GEPA not installed. Install with: pip install gepa-ai\n"
                "See https://github.com/gepa-ai/gepa for documentation."
            )

        adapter = self.adapters.get(module_name)
        if not adapter:
            raise ValueError(f"Unknown module: {module_name}")

        print(f"\n{'='*60}")
        print(f"GEPA Optimization: {module_name}")
        print(f"{'='*60}")
        print(f"Training examples: {len(training_examples)}")
        if validation_examples:
            print(f"Validation examples: {len(validation_examples)}")
        print(f"Budget: {self.budget} metric calls")
        print(f"Reflection LM: {self.reflection_lm}")

        # Convert to GEPA DataInst format
        trainset = self._convert_to_gepa_batch(training_examples, module_name)
        valset = self._convert_to_gepa_batch(validation_examples, module_name) if validation_examples else None

        # Get seed candidate (current prompt)
        seed_candidate = adapter.get_candidate()
        component_name = adapter.component_name

        print(f"\nSeed prompt ({component_name}):")
        print(f"  Length: {len(seed_candidate[component_name])} chars")

        # Run GEPA optimization
        print("\nRunning GEPA optimization...")
        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=self.reflection_lm,
            max_metric_calls=self.budget,
        )

        # Extract results
        best_prompt = result.best_candidate.get(component_name, "")
        best_score = result.best_score

        # Save optimized prompt
        prompt_path = self.output_dir / module_name / "prompts" / "v1_optimized.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(best_prompt)

        # Prepare results dict
        results = {
            "module_name": module_name,
            "component_name": component_name,
            "baseline_score": result.seed_score if hasattr(result, 'seed_score') else None,
            "best_score": best_score,
            "improvement": best_score - result.seed_score if hasattr(result, 'seed_score') else None,
            "optimized_prompt": best_prompt,
            "prompt_path": str(prompt_path),
            "num_candidates_evaluated": result.num_candidates if hasattr(result, 'num_candidates') else None,
        }

        print(f"\nOptimization complete for {module_name}")
        print(f"  Best score: {best_score:.4f}")
        if hasattr(result, 'seed_score'):
            print(f"  Baseline: {result.seed_score:.4f}")
            print(f"  Improvement: {best_score - result.seed_score:+.4f}")
        print(f"  Saved to: {prompt_path}")

        return results

    def _convert_to_gepa_batch(
        self,
        examples: List[ModuleTrainingExample],
        module_name: str,
    ) -> List[Dict[str, Any]]:
        """Convert CRAG examples to GEPA DataInst format"""
        if not examples:
            return []

        batch = []

        for ex in examples:
            data_inst = {
                "query": ex.input_data.get("query", ""),
                "ground_truth": ex.output_data.get("ground_truth", ""),
                "relevant_chunk_indices": None,
                "contexts": ex.input_data.get("documents", []),
                "metadata": {
                    "quality_label": ex.quality_label,
                    "feedback": ex.feedback,
                    "score": ex.score,
                },
            }
            batch.append(data_inst)

        return batch

    def optimize_all_modules(
        self,
        training_dataset: TrainingDataset,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run GEPA optimization for all modules.

        Args:
            training_dataset: CRAG training dataset

        Returns:
            Dict with results per module
        """
        results = {}

        # Module 1: Query Planner
        if training_dataset.query_planner_examples:
            results["query_planner"] = self.optimize_module(
                "query_planner",
                training_dataset.query_planner_examples,
            )

        # Module 2: Reranker
        if training_dataset.reranker_examples:
            results["reranker"] = self.optimize_module(
                "reranker",
                training_dataset.reranker_examples,
            )

        # Module 3: Generator
        if training_dataset.generator_examples:
            results["generator"] = self.optimize_module(
                "generator",
                training_dataset.generator_examples,
            )

        # Save summary
        summary_path = self.output_dir / "optimization_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nOptimization summary saved to: {summary_path}")

        return results