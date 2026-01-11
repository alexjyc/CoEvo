"""
Generator GEPA Adapter (Module 3)

Optimizes answer generation prompts to improve faithfulness and correctness.

Module I/O:
- Input: GeneratorInput(query: str, context: str, feedback: Optional[str])
- Output: GeneratorOutput(status, answer: str, reference: str, rationale: str, metadata)

Target Metrics: faithfulness, answer_correctness
Optimization Goal: Generate accurate answers grounded in context

GEPA Optimization Notes:
- Uses metric-based diagnostic feedback to avoid overfitting
- Analyzes answer quality (context usage, query coverage, hedging)
- Provides strategy-focused suggestions based on faithfulness/correctness patterns
- NO ground truth in reflection records
"""

# Import module types
import re
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
    - Generate accurate and complete answers
    - Provide proper references and rationale

    Key Features for GEPA Optimization:
    - Metric-based diagnostic feedback (no ground truth in reflection)
    - Answer quality analysis (context usage, query coverage, hedging)
    - Faithfulness vs correctness pattern diagnosis
    - Strategy-focused improvement suggestions

    The optimization maximizes (faithfulness + answer_correctness) / 2.
    """

    # Hedging words that indicate uncertainty
    HEDGING_WORDS = {
        "might", "could", "may", "possibly", "perhaps", "probably",
        "likely", "unlikely", "seems", "appears", "suggest", "suggests",
        "indicate", "indicates", "potential", "potentially", "uncertain",
        "unclear", "approximately", "roughly", "about", "around",
    }

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

    # -------------------------------------------------------------------------
    # Answer Quality Analysis (for diagnostic feedback without ground truth)
    # -------------------------------------------------------------------------

    def _analyze_answer_quality(
        self,
        answer: str,
        context: str,
        query: str,
    ) -> dict[str, Any]:
        """
        Analyze answer characteristics without ground truth comparison.

        Args:
            answer: Generated answer text
            context: Context provided to the generator
            query: Original user query

        Returns:
            Dict with quality analysis metrics
        """
        if not answer:
            return {
                "answer_length": 0,
                "context_coverage": 0.0,
                "query_term_coverage": 0.0,
                "hedging_level": 0.0,
                "has_specific_details": False,
            }

        answer_words = answer.lower().split()
        answer_length = len(answer_words)

        # Measure context usage
        context_coverage = self._measure_context_usage(answer, context)

        # Measure query addressing
        query_coverage = self._measure_query_addressing(answer, query)

        # Detect hedging
        hedging_level = self._detect_hedging(answer)

        # Check for specific details (numbers, named entities, etc.)
        has_specifics = bool(re.search(r'\d+|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer))

        return {
            "answer_length": answer_length,
            "context_coverage": context_coverage,
            "query_term_coverage": query_coverage,
            "hedging_level": hedging_level,
            "has_specific_details": has_specifics,
        }

    def _measure_context_usage(self, answer: str, context: str) -> float:
        """
        Measure how much of the answer content comes from the context.

        Args:
            answer: Generated answer text
            context: Context provided to the generator

        Returns:
            Score from 0-1 indicating context usage
        """
        if not answer or not context:
            return 0.0

        # Extract significant words (longer than 4 chars, not common)
        answer_words = set(w.lower() for w in answer.split() if len(w) > 4)
        context_words = set(w.lower() for w in context.split() if len(w) > 4)

        if not answer_words:
            return 0.0

        # Measure overlap
        overlap = len(answer_words & context_words)
        return overlap / len(answer_words)

    def _measure_query_addressing(self, answer: str, query: str) -> float:
        """
        Measure how well the answer addresses the query terms.

        Args:
            answer: Generated answer text
            query: Original user query

        Returns:
            Score from 0-1 indicating query coverage
        """
        if not answer or not query:
            return 0.0

        # Extract query terms (significant words)
        query_terms = set(w.lower() for w in query.split() if len(w) > 3)
        answer_lower = answer.lower()

        if not query_terms:
            return 1.0  # No significant query terms to cover

        # Count query terms present in answer
        covered = sum(1 for term in query_terms if term in answer_lower)
        return covered / len(query_terms)

    def _detect_hedging(self, answer: str) -> float:
        """
        Detect hedging language that indicates uncertainty.

        Args:
            answer: Generated answer text

        Returns:
            Score from 0-1 indicating hedging level
        """
        if not answer:
            return 0.0

        words = answer.lower().split()
        if not words:
            return 0.0

        hedge_count = sum(1 for w in words if w in self.HEDGING_WORDS)
        return min(hedge_count / len(words) * 10, 1.0)  # Scale up, cap at 1.0

    # -------------------------------------------------------------------------
    # Reflective Dataset Generation (Enhanced for GEPA)
    # -------------------------------------------------------------------------

    def _format_trace_for_reflection(
        self,
        trajectory: RAGTrajectory,
    ) -> dict[str, Any]:
        """
        Format trajectory into GEPA reflection record with diagnostic feedback.

        GEPA Best Practices Applied:
        1. NO ground truth in reflection (prevents overfitting)
        2. Answer quality analysis (context usage, query coverage, hedging)
        3. Metric-based faithfulness vs correctness diagnosis
        4. Strategy-focused improvement suggestions

        Returns:
            Dict with Inputs, Generated Outputs, Quality Analysis, Feedback, Score
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

        # Analyze answer quality
        quality_analysis = self._analyze_answer_quality(answer, context, query)

        # Add quality analysis to outputs
        outputs_text += f"\n\nQuality Analysis: "
        outputs_text += f"Context coverage={quality_analysis['context_coverage']:.0%}, "
        outputs_text += f"Query term coverage={quality_analysis['query_term_coverage']:.0%}, "
        outputs_text += f"Hedging level={quality_analysis['hedging_level']:.0%}"

        # Generate diagnostic feedback (no ground truth)
        feedback = self._generate_rich_feedback(
            score=score,
            metrics=metrics,
            answer=answer,
            context=context,
            query=query,
        )

        return {
            "Inputs": inputs_text,
            "Generated Outputs": outputs_text,
            "Feedback": feedback,
            "Score": f"{score:.3f}",
        }

    def _generate_rich_feedback(
        self,
        score: float,
        metrics: dict[str, float],
        answer: str,
        context: str,
        query: str,
    ) -> str:
        """
        Generate diagnostic feedback for GEPA reflection (no ground truth).

        Uses metric patterns and answer quality analysis to provide
        strategy-focused improvement suggestions.

        Args:
            score: The primary metric score
            metrics: All evaluation metrics
            answer: Generated answer text
            context: Context provided to the generator
            query: Original user query

        Returns:
            Diagnostic feedback string with strategy suggestions
        """
        faithfulness = metrics.get("faithfulness", 0)
        correctness = metrics.get("answer_correctness", 0)

        # Analyze answer quality
        quality = self._analyze_answer_quality(answer, context, query)

        # Determine performance tier and generate appropriate feedback
        if score >= 0.6:
            return self._positive_feedback_diagnostic(score, metrics, quality)
        elif score >= 0.3:
            return self._partial_feedback_diagnostic(score, metrics, quality)
        else:
            return self._negative_feedback_diagnostic(score, metrics, quality)

    def _positive_feedback_diagnostic(
        self,
        score: float,
        metrics: dict[str, float],
        quality: dict[str, Any],
    ) -> str:
        """Generate positive feedback with reinforcement of what worked."""
        faithfulness = metrics.get("faithfulness", 0)
        correctness = metrics.get("answer_correctness", 0)

        feedback = f"GOOD: Score={score:.2f}. "
        feedback += f"Faithfulness={faithfulness:.2f}, Correctness={correctness:.2f}. "
        feedback += "Answer is well-grounded and addresses the question. "

        # Explain what worked based on quality analysis
        if quality["context_coverage"] > 0.6:
            feedback += "Strong context usage. "
        if quality["query_term_coverage"] > 0.7:
            feedback += "Good query term coverage. "
        if quality["hedging_level"] < 0.2:
            feedback += "Appropriately confident tone. "

        feedback += "Continue this approach for similar questions."

        return feedback

    def _partial_feedback_diagnostic(
        self,
        score: float,
        metrics: dict[str, float],
        quality: dict[str, Any],
    ) -> str:
        """Generate feedback for partial success with strategy-focused improvements."""
        faithfulness = metrics.get("faithfulness", 0)
        correctness = metrics.get("answer_correctness", 0)

        feedback = f"PARTIAL: Score={score:.2f}. "
        feedback += f"Faithfulness={faithfulness:.2f}, Correctness={correctness:.2f}. "

        # Diagnose based on metric patterns (no ground truth needed)
        if faithfulness < correctness:
            feedback += "PATTERN: Some claims may not be supported by context. "
            feedback += "TRY: Ground every statement in explicit context quotes. "
            if quality["context_coverage"] < 0.5:
                feedback += f"Note: Only {quality['context_coverage']:.0%} context coverage. "
        elif correctness < faithfulness:
            feedback += "PATTERN: Answer is grounded but may not fully address the question. "
            feedback += "TRY: Ensure response directly answers what was asked. "
            if quality["query_term_coverage"] < 0.5:
                feedback += f"Note: Only {quality['query_term_coverage']:.0%} query term coverage. "
        else:
            feedback += "PATTERN: Moderate quality on both grounding and accuracy. "
            feedback += "TRY: Improve both context usage and question addressing. "

        # Additional quality-based suggestions
        if quality["hedging_level"] > 0.3:
            feedback += "STRATEGY: Reduce hedging language - be more definitive when context supports it. "

        return feedback

    def _negative_feedback_diagnostic(
        self,
        score: float,
        metrics: dict[str, float],
        quality: dict[str, Any],
    ) -> str:
        """Generate detailed negative feedback with strategy-focused suggestions."""
        faithfulness = metrics.get("faithfulness", 0)
        correctness = metrics.get("answer_correctness", 0)

        feedback = f"POOR: Score={score:.2f}. "
        feedback += f"Faithfulness={faithfulness:.2f}, Correctness={correctness:.2f}. "

        # Diagnose primary failure mode using metrics
        if faithfulness < 0.3:
            feedback += "CRITICAL: Low faithfulness - answer may contain unsupported claims. "
            feedback += "The generated content may not be grounded in the provided context. "
        if correctness < 0.3:
            feedback += "CRITICAL: Low correctness - answer may not address the question properly. "

        # Quality-based diagnosis
        if quality["context_coverage"] < 0.3:
            feedback += f"ISSUE: Only {quality['context_coverage']:.0%} of answer terms from context. "
        if quality["query_term_coverage"] < 0.3:
            feedback += f"ISSUE: Only {quality['query_term_coverage']:.0%} query terms addressed. "

        # Strategy suggestions
        feedback += "STRATEGIES: "
        feedback += "(1) Quote directly from context to support claims, "
        feedback += "(2) Address the question stem explicitly, "
        feedback += "(3) Avoid adding information not in context. "

        # Recovery guidance
        if quality["hedging_level"] > 0.4:
            feedback += "RECOVERY: Excessive hedging detected - be more definitive when evidence is clear. "
        if not quality["has_specific_details"]:
            feedback += "RECOVERY: Include specific details from context (names, numbers, facts). "

        return feedback

    def _positive_feedback(self, score: float, metrics: dict[str, float]) -> str:
        """Generate positive feedback for high-quality generation."""
        return self._generate_rich_feedback(score, metrics, "", "", "")

    def _negative_feedback(self, score: float, metrics: dict[str, float]) -> str:
        """Generate improvement suggestions for low-quality generation."""
        return self._generate_rich_feedback(score, metrics, "", "", "")


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
