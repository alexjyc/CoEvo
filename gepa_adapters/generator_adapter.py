"""
Generator GEPA Adapter (Module 3)

Optimizes answer generation prompts to improve faithfulness and correctness.

Module I/O:
- Input: GeneratorInput(query: str, context: str, feedback: Optional[str])
- Output: GeneratorOutput(status, answer: str, reference: str, rationale: str, metadata)

Target Metrics: faithfulness, answer_correctness
Optimization Goal: Generate accurate answers grounded in context
"""

from typing import Any, Dict, List, Optional, Tuple

from gepa_adapters.base import (
    RAGModuleAdapter,
    RAGDataInst,
    RAGTrajectory,
    RAGRolloutOutput,
)

# Import module types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.base import GeneratorInput


class GeneratorAdapter(RAGModuleAdapter):
    """
    GEPA Adapter for Generator Module.

    This adapter optimizes the generation prompt to:
    - Produce answers that are faithful to the context
    - Generate correct answers matching ground truth
    - Provide proper references and rationale

    The optimization maximizes (faithfulness + answer_correctness) / 2.
    """

    def __init__(
        self,
        generator_module,
        evaluator,
        failure_score: float = 0.0,
    ):
        """
        Initialize Generator Adapter.

        Args:
            generator_module: GeneratorModule instance
            evaluator: RAGASEvaluator instance
            failure_score: Score on execution failure
        """
        super().__init__(
            module=generator_module,
            evaluator=evaluator,
            component_name="generator_prompt",
            failure_score=failure_score,
        )

    async def _run_single_async(
        self,
        data: RAGDataInst,
    ) -> Tuple[RAGRolloutOutput, Dict[str, Any], Dict[str, Any]]:
        """
        Execute generator on a single example.

        Process:
        1. Take query and context from data
        2. Generate answer using the module
        3. Return answer for evaluation

        Args:
            data: RAGDataInst with query, contexts (as list or concatenated)

        Returns:
            Tuple of (output, module_input, module_output)
        """
        query = data["query"]
        contexts = data.get("contexts", [])

        # Convert contexts list to single string if needed
        if isinstance(contexts, list):
            context_text = "\n\n".join(contexts)
        else:
            context_text = contexts or ""

        if not context_text:
            return {
                "result": None,
                "success": False,
                "error": "No context provided for generation",
            }, {"query": query, "context": ""}, {}

        # Run generator
        generator_input = GeneratorInput(
            query=query,
            context=context_text,
        )
        generator_output = await self.module.run(generator_input)

        # Package output
        output: RAGRolloutOutput = {
            "result": {
                "answer": generator_output.answer,
                "reference": generator_output.reference,
                "rationale": generator_output.rationale,
            },
            "success": generator_output.status == "success",
            "error": generator_output.error_message,
        }

        module_input = {
            "query": query,
            "context": context_text,
            "context_length": len(context_text),
        }

        module_output = {
            "answer": generator_output.answer,
            "reference": generator_output.reference,
            "rationale": generator_output.rationale,
            "answer_length": len(generator_output.answer),
        }

        return output, module_input, module_output

    async def _compute_score_async(
        self,
        data: RAGDataInst,
        module_output: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute generation quality score.

        Metrics:
        - faithfulness: Is the answer grounded in context?
        - answer_correctness: Does the answer match ground truth?

        Args:
            data: Original input with ground_truth
            module_output: Generator output with answer

        Returns:
            Tuple of (generation_quality, detailed_metrics)
        """
        answer = module_output.get("answer", "")
        contexts = data.get("contexts", [])
        ground_truth = data.get("ground_truth", "")
        query = data["query"]

        # Convert contexts to list if needed
        if isinstance(contexts, str):
            contexts = [contexts] if contexts else []

        # Use evaluator for generation metrics
        generator_eval = await self.evaluator._evaluate_generator(
            input_data=GeneratorInput(query=query, context="\n\n".join(contexts)),
            output_data=type('obj', (object,), {'answer': answer})(),
            ground_truth={
                "query": query,
                "contexts": contexts,
                "reference": ground_truth
            } if ground_truth else None
        )

        faithfulness = generator_eval.get("faithfulness", 0.0)
        answer_correctness = generator_eval.get("answer_correctness", 0.0)
        generation_quality = generator_eval.get("generation_quality", 0.0)

        # If quality not computed, calculate it
        if generation_quality == 0.0 and (faithfulness > 0 or answer_correctness > 0):
            generation_quality = (faithfulness + answer_correctness) / 2

        metrics = {
            "faithfulness": faithfulness,
            "answer_correctness": answer_correctness,
            "generation_quality": generation_quality,
            "answer_length": module_output.get("answer_length", len(answer)),
            "has_reference": len(module_output.get("reference", "")) > 0,
            "has_rationale": len(module_output.get("rationale", "")) > 0,
        }

        return generation_quality, metrics

    def _format_trace_for_reflection(
        self,
        trajectory: RAGTrajectory,
    ) -> Dict[str, Any]:
        """
        Format trajectory into GEPA reflection record.

        Returns:
            Dict with Inputs, Generated Outputs, Feedback keys
        """
        data = trajectory["data"]
        module_input = trajectory["module_input"]
        module_output = trajectory["module_output"]
        score = trajectory["score"]
        metrics = trajectory["metrics"]

        # Format inputs
        query = module_input.get("query", data["query"])
        context = module_input.get("context", "")
        context_preview = context[:500] + "..." if len(context) > 500 else context

        inputs_text = f"Query: {query}\n\nContext:\n{context_preview}"

        # Format outputs
        answer = module_output.get("answer", "")
        reference = module_output.get("reference", "")
        rationale = module_output.get("rationale", "")

        outputs_text = f"Answer: {answer}\n\nReference: {reference}\n\nRationale: {rationale}"

        # Add ground truth for comparison
        ground_truth = data.get("ground_truth", "")
        if ground_truth:
            outputs_text += f"\n\n[Ground Truth: {ground_truth}]"

        # Generate feedback
        feedback = self._generate_feedback(score, metrics)

        return {
            "Inputs": inputs_text,
            "Generated Outputs": outputs_text,
            "Feedback": feedback,
            "Score": score,
            "Metrics": metrics,
        }

    def _positive_feedback(self, score: float, metrics: Dict[str, float]) -> str:
        """Generate positive feedback for high-quality generation"""
        faithfulness = metrics.get("faithfulness", 0)
        correctness = metrics.get("answer_correctness", 0)
        has_reference = metrics.get("has_reference", False)

        feedback = f"High-quality generation (score={score:.2f}). "

        if faithfulness >= 0.8:
            feedback += "Answer is well-grounded in the provided context. "

        if correctness >= 0.8:
            feedback += "Answer accurately addresses the query. "

        if has_reference:
            feedback += "Good use of references to support claims. "

        feedback += "Continue focusing on accuracy and context-grounding."

        return feedback

    def _negative_feedback(self, score: float, metrics: Dict[str, float]) -> str:
        """Generate improvement suggestions for low-quality generation"""
        faithfulness = metrics.get("faithfulness", 0)
        correctness = metrics.get("answer_correctness", 0)
        has_reference = metrics.get("has_reference", False)
        has_rationale = metrics.get("has_rationale", False)

        feedback = f"Generation needs improvement (score={score:.2f}). "

        if faithfulness < 0.5:
            feedback += "LOW FAITHFULNESS: Answer contains claims not supported by context. "
            feedback += "Only include information explicitly stated in the context. "

        if correctness < 0.5:
            feedback += "LOW CORRECTNESS: Answer does not match expected response. "
            feedback += "Focus on directly answering the question asked. "

        if not has_reference:
            feedback += "Missing references: cite specific parts of context. "

        if not has_rationale:
            feedback += "Missing rationale: explain reasoning. "

        if faithfulness >= 0.5 and correctness >= 0.5:
            feedback += "Improve clarity and completeness of the answer. "

        feedback += "Key improvements: (1) ground all claims in context, "
        feedback += "(2) directly answer the question, (3) provide evidence."

        return feedback


# =============================================================================
# Utility: Create adapter with standard configuration
# =============================================================================

def create_generator_adapter(
    generator_module,
    evaluator,
) -> GeneratorAdapter:
    """
    Factory function to create GeneratorAdapter.

    Args:
        generator_module: GeneratorModule instance
        evaluator: RAGASEvaluator instance

    Returns:
        Configured GeneratorAdapter
    """
    return GeneratorAdapter(
        generator_module=generator_module,
        evaluator=evaluator,
        failure_score=0.0,
    )
