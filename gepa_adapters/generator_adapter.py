"""
Generator GEPA Adapter (Module 3)

Optimizes answer generation prompts to improve faithfulness and correctness.

Module I/O:
- Input: GeneratorInput(query: str, context: str, feedback: Optional[str])
- Output: GeneratorOutput(status, answer: str, reference: str, rationale: str, metadata)

Target Metrics: faithfulness, answer_correctness
Optimization Goal: Generate accurate answers grounded in context
"""

# Import module types
import sys
from pathlib import Path
from typing import Any

from gepa_adapters.base import (
    RAGDataInst,
    RAGModuleAdapter,
    RAGRolloutOutput,
    RAGTrajectory,
)

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
    ) -> tuple[RAGRolloutOutput, dict[str, Any], dict[str, Any]]:
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
            return (
                {
                    "result": None,
                    "success": False,
                    "error": "No context provided for generation",
                },
                {"query": query, "context": ""},
                {},
            )

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
        module_output: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
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
            output_data=type("obj", (object,), {"answer": answer})(),
            ground_truth={"query": query, "contexts": contexts, "reference": ground_truth}
            if ground_truth
            else None,
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
    ) -> dict[str, Any]:
        """
        Format trajectory into GEPA reflection record with rich feedback.

        GEPA Best Practices Applied:
        1. Include ground truth answer as a separate field
        2. Show comparison between generated and expected
        3. Provide specific, actionable improvement suggestions

        Returns:
            Dict with Inputs, Generated Outputs, Expected Answer, Feedback keys
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

        # Format generated outputs
        answer = module_output.get("answer", "")
        reference = module_output.get("reference", "")
        rationale = module_output.get("rationale", "")

        outputs_text = f"Generated Answer: {answer}"
        if reference:
            outputs_text += f"\n\nReference: {reference}"
        if rationale:
            outputs_text += f"\n\nRationale: {rationale}"

        # Format expected answer separately (for GEPA's reflection LLM)
        ground_truth = data.get("ground_truth", "")
        if ground_truth:
            expected_text = f"Expected Answer: {ground_truth}"
        else:
            expected_text = "Expected Answer: Not available"

        # Generate rich feedback with ground truth comparison
        feedback = self._generate_rich_feedback(score, metrics, ground_truth, answer)

        return {
            "Inputs": inputs_text,
            "Generated Outputs": outputs_text,
            "Expected Answer": expected_text,
            "Feedback": feedback,
            "Score": f"{score:.3f}",
        }

    def _generate_rich_feedback(
        self,
        score: float,
        metrics: dict[str, float],
        ground_truth: str,
        generated_answer: str,
    ) -> str:
        """Generate rich feedback with contrastive comparison to ground truth."""
        faithfulness = metrics.get("faithfulness", 0)
        correctness = metrics.get("answer_correctness", 0)

        # Determine performance tier (adaptive thresholds)
        if score >= 0.6:
            feedback = f"GOOD: Score={score:.2f}. "
            feedback += f"Faithfulness={faithfulness:.2f}, Correctness={correctness:.2f}. "
            feedback += "Answer is well-grounded and accurate. "
            if ground_truth:
                feedback += "Matches expected answer closely."
        elif score >= 0.3:
            feedback = f"PARTIAL: Score={score:.2f}. "
            if faithfulness < correctness:
                feedback += "ISSUE: Some claims not supported by context. "
                feedback += "TRY: Only include information explicitly in context. "
            elif correctness < faithfulness:
                feedback += "ISSUE: Answer doesn't fully match expected response. "
                if ground_truth:
                    gt_preview = ground_truth[:200]
                    feedback += f"EXPECTED: '{gt_preview}...'. "
                feedback += "TRY: Focus on directly answering the question. "
            else:
                feedback += "Moderate quality. Improve both grounding and accuracy. "
        else:
            feedback = f"POOR: Score={score:.2f}. "
            if faithfulness < 0.3:
                feedback += "CRITICAL: Answer contains hallucinations. "
            if correctness < 0.3:
                feedback += "CRITICAL: Answer is incorrect. "
            if ground_truth:
                gt_preview = ground_truth[:250]
                feedback += f"MUST PRODUCE: '{gt_preview}...'. "
            feedback += "REQUIREMENTS: (1) Ground every claim in context, (2) Answer the question directly, (3) No made-up information."

        return feedback

    def _positive_feedback(self, score: float, metrics: dict[str, float]) -> str:
        """Generate positive feedback for high-quality generation"""
        return self._generate_rich_feedback(score, metrics, "", "")

    def _negative_feedback(self, score: float, metrics: dict[str, float]) -> str:
        """Generate improvement suggestions for low-quality generation"""
        return self._generate_rich_feedback(score, metrics, "", "")


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
