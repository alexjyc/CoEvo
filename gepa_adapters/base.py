"""
GEPA Adapter Base Types and Abstract Class

Implements the GEPAAdapter protocol for RAG module optimization.
Uses the actual GEPA library for optimization.

GEPA Protocol Requirements:
1. evaluate(batch, candidate, capture_traces) -> EvaluationBatch
2. make_reflective_dataset(candidate, eval_batch, components_to_update) -> Mapping

Usage with gepa.optimize():
    from gepa import optimize
    from gepa_adapters import QueryPlannerAdapter

    adapter = QueryPlannerAdapter(module, retriever, evaluator)
    result = optimize(
        seed_candidate={"query_planner_prompt": seed_prompt},
        trainset=training_data,
        valset=validation_data,
        adapter=adapter,
        reflection_lm="openai/gpt-4o-mini",
        max_metric_calls=100,
    )
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Mapping, Optional, Sequence, TypeVar, TypedDict

# Import GEPA types - use actual library when available
try:
    from gepa.core.adapter import GEPAAdapter, EvaluationBatch as GEPAEvaluationBatch
    GEPA_AVAILABLE = True
except ImportError:
    print("⚠️  GEPA not installed. Install with: pip install gepa-ai")
    GEPA_AVAILABLE = False
    GEPAAdapter = ABC  # Fallback to ABC if GEPA not installed
    GEPAEvaluationBatch = None


# =============================================================================
# GEPA Type Definitions (matching gepa-ai/gepa protocol)
# =============================================================================

class RAGDataInst(TypedDict):
    """
    Data instance for RAG module evaluation.

    Fields:
        query: The user query
        ground_truth: Expected answer (for evaluation)
        relevant_chunk_indices: Ground truth relevant chunk IDs
        contexts: Pre-retrieved documents (for reranker/generator)
        metadata: Additional task-specific data
    """
    query: str
    ground_truth: Optional[str]
    relevant_chunk_indices: Optional[List[int]]
    contexts: Optional[List[str]]
    metadata: Dict[str, Any]


class RAGTrajectory(TypedDict):
    """
    Execution trace for reflection.

    Fields:
        data: Original input data instance
        module_input: Formatted input to the module
        module_output: Raw output from the module
        intermediate_steps: Any intermediate execution state
        score: Evaluation score for this example
        metrics: Detailed metric breakdown
    """
    data: RAGDataInst
    module_input: Dict[str, Any]
    module_output: Dict[str, Any]
    intermediate_steps: List[Dict[str, Any]]
    score: float
    metrics: Dict[str, float]


class RAGRolloutOutput(TypedDict):
    """
    Output from a single rollout/evaluation.

    Fields:
        result: The module's output (type varies by module)
        success: Whether execution succeeded
        error: Error message if failed
    """
    result: Any
    success: bool
    error: Optional[str]


Trajectory = TypeVar("Trajectory")
RolloutOutput = TypeVar("RolloutOutput")


@dataclass
class EvaluationBatch(Generic[Trajectory, RolloutOutput]):
    """
    Container for evaluation results on a batch of examples.

    GEPA Protocol Constraints:
    - len(outputs) == len(scores) == len(batch)
    - If capture_traces=True: len(trajectories) == len(batch)

    Attributes:
        outputs: Per-example raw outputs
        scores: Per-example scores (higher is better)
        trajectories: Optional execution traces for reflection
    """
    outputs: List[RolloutOutput]
    scores: List[float]
    trajectories: Optional[List[Trajectory]] = None

    @property
    def aggregate_score(self) -> float:
        """Mean score across all examples"""
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of successful executions"""
        if not self.outputs:
            return 0.0
        successes = sum(1 for o in self.outputs if o.get("success", False))
        return successes / len(self.outputs)


# =============================================================================
# Abstract Base Adapter - Implements GEPAAdapter Protocol
# =============================================================================

# Choose base class based on GEPA availability
_BaseClass = GEPAAdapter if GEPA_AVAILABLE else ABC


class RAGModuleAdapter(_BaseClass):
    """
    Abstract base class implementing the GEPAAdapter protocol for RAG modules.

    IMPORTANT: This class is designed to work directly with gepa.optimize().
    The evaluate() method is SYNCHRONOUS as required by GEPA protocol.

    Each adapter must implement:
    1. _run_single_async(): Execute module on single example (async)
    2. _compute_score_async(): Calculate score from module output (async)
    3. _format_trace_for_reflection(): Format trajectory for GEPA reflection

    The base class handles:
    - Prompt injection/restoration
    - Batch execution with error handling (wraps async in sync)
    - EvaluationBatch construction
    - Reflective dataset generation

    Usage with gepa.optimize():
        adapter = QueryPlannerAdapter(module, retriever, evaluator)
        result = gepa.optimize(
            seed_candidate={"query_planner_prompt": seed_prompt},
            trainset=data,
            adapter=adapter,
            reflection_lm="openai/gpt-4o-mini",
        )
    """

    def __init__(
        self,
        module,
        evaluator,
        component_name: str,
        failure_score: float = 0.0,
    ):
        """
        Initialize the adapter.

        Args:
            module: The RAG module to optimize (QueryPlannerModule, etc.)
            evaluator: RAGASEvaluator instance for scoring
            component_name: Name of the prompt component (e.g., "query_planner_prompt")
            failure_score: Score to assign on execution failure
        """
        self.module = module
        self.evaluator = evaluator
        self.component_name = component_name
        self.failure_score = failure_score
        self._original_prompt: Optional[str] = None

    # -------------------------------------------------------------------------
    # Prompt Management
    # -------------------------------------------------------------------------

    @property
    def current_prompt(self) -> str:
        """Get the current prompt from the module"""
        return self.module.prompt or ""

    def inject_prompt(self, candidate: Dict[str, str]) -> None:
        """Inject candidate prompt into the module"""
        if self.component_name in candidate:
            if self._original_prompt is None:
                self._original_prompt = self.module.prompt
            self.module.prompt = candidate[self.component_name]

    def restore_prompt(self) -> None:
        """Restore the original prompt after optimization"""
        if self._original_prompt is not None:
            self.module.prompt = self._original_prompt
            self._original_prompt = None

    def get_candidate(self) -> Dict[str, str]:
        """Get current prompt as a candidate dict"""
        return {self.component_name: self.current_prompt}

    # -------------------------------------------------------------------------
    # Abstract Methods (implement in subclasses) - ASYNC internal methods
    # -------------------------------------------------------------------------

    @abstractmethod
    async def _run_single_async(
        self,
        data: RAGDataInst,
    ) -> tuple[RAGRolloutOutput, Dict[str, Any], Dict[str, Any]]:
        """
        Execute the module on a single data instance (async).

        Args:
            data: Input data instance

        Returns:
            Tuple of (output, module_input_dict, module_output_dict)
        """
        pass

    @abstractmethod
    async def _compute_score_async(
        self,
        data: RAGDataInst,
        module_output: Dict[str, Any],
    ) -> tuple[float, Dict[str, float]]:
        """
        Compute evaluation score for module output (async).

        Args:
            data: Original input data
            module_output: Module's output dictionary

        Returns:
            Tuple of (aggregate_score, detailed_metrics)
        """
        pass

    @abstractmethod
    def _format_trace_for_reflection(
        self,
        trajectory: RAGTrajectory,
    ) -> Dict[str, Any]:
        """
        Format a trajectory into a reflection record for GEPA.

        Must return a dict with:
        - "Inputs": What was given to the module
        - "Generated Outputs": What the module produced
        - "Feedback": What went well/wrong

        Args:
            trajectory: Execution trace

        Returns:
            Reflection record dict
        """
        pass

    # -------------------------------------------------------------------------
    # GEPA Protocol Method #1: evaluate() - SYNCHRONOUS for GEPA compatibility
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        batch: List[RAGDataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[RAGTrajectory, RAGRolloutOutput]:
        """
        Evaluate the module on a batch of examples using the candidate prompt.

        GEPA Protocol Method #1 - SYNCHRONOUS (required by GEPA)

        This method wraps async operations using asyncio.run() to provide
        a synchronous interface as required by the GEPA protocol.

        Args:
            batch: List of data instances
            candidate: Dict mapping component names to prompt strings
            capture_traces: Whether to capture execution traces

        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories
        """
        # Handle both standalone and nested event loop contexts (e.g., GEPA)
        try:
            loop = asyncio.get_running_loop()
            # We're in an existing event loop - use loop.run_until_complete or thread
            try:
                import nest_asyncio
                nest_asyncio.apply()
                # Use the existing loop instead of creating a new one
                future = asyncio.ensure_future(
                    self._evaluate_async(batch, candidate, capture_traces),
                    loop=loop
                )
                return loop.run_until_complete(future)
            except ImportError:
                # nest_asyncio not available - run in separate thread with new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        lambda: asyncio.run(
                            self._evaluate_async(batch, candidate, capture_traces)
                        )
                    )
                    return future.result()
        except RuntimeError:
            # No running event loop - safe to use asyncio.run()
            return asyncio.run(self._evaluate_async(batch, candidate, capture_traces))

    async def _evaluate_async(
        self,
        batch: List[RAGDataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[RAGTrajectory, RAGRolloutOutput]:
        """
        Async implementation of evaluate - called by sync evaluate().

        Args:
            batch: List of data instances
            candidate: Dict mapping component names to prompt strings
            capture_traces: Whether to capture execution traces

        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories
        """
        # Inject the candidate prompt
        self.inject_prompt(candidate)

        outputs: List[RAGRolloutOutput] = []
        scores: List[float] = []
        trajectories: Optional[List[RAGTrajectory]] = [] if capture_traces else None

        try:
            for data in batch:
                try:
                    # Run the module (async)
                    output, module_input, module_output = await self._run_single_async(data)

                    # Compute score (async)
                    score, metrics = await self._compute_score_async(data, module_output)

                    outputs.append(output)
                    scores.append(score)

                    # Capture trajectory if requested
                    if capture_traces:
                        trajectory: RAGTrajectory = {
                            "data": data,
                            "module_input": module_input,
                            "module_output": module_output,
                            "intermediate_steps": [],
                            "score": score,
                            "metrics": metrics,
                        }
                        trajectories.append(trajectory)

                except Exception as e:
                    # Handle individual example failures
                    outputs.append({
                        "result": None,
                        "success": False,
                        "error": str(e),
                    })
                    scores.append(self.failure_score)

                    if capture_traces:
                        trajectories.append({
                            "data": data,
                            "module_input": {},
                            "module_output": {},
                            "intermediate_steps": [],
                            "score": self.failure_score,
                            "metrics": {},
                        })

            return EvaluationBatch(
                outputs=outputs,
                scores=scores,
                trajectories=trajectories,
            )

        finally:
            # Always restore the original prompt
            self.restore_prompt()

    # -------------------------------------------------------------------------
    # GEPA Protocol Method #2: make_reflective_dataset()
    # -------------------------------------------------------------------------

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch[RAGTrajectory, RAGRolloutOutput],
        components_to_update: List[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Generate a reflection dataset for prompt evolution.

        GEPA Protocol Method #2

        Args:
            candidate: The candidate prompts that were used
            eval_batch: Evaluation results with trajectories
            components_to_update: Which components need new prompts

        Returns:
            Dict mapping component names to lists of reflection records
        """
        if self.component_name not in components_to_update:
            return {}

        if not eval_batch.trajectories:
            return {self.component_name: []}

        # Create reflection records from trajectories
        records: List[Dict[str, Any]] = []

        for trajectory in eval_batch.trajectories:
            record = self._format_trace_for_reflection(trajectory)
            records.append(record)

        return {self.component_name: records}

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _generate_feedback(
        self,
        score: float,
        metrics: Dict[str, float],
        threshold: float = 0.7,
    ) -> str:
        """
        Generate feedback string based on score and metrics.

        Args:
            score: Aggregate score
            metrics: Detailed metric breakdown
            threshold: Score threshold for positive/negative feedback

        Returns:
            Feedback string for reflection
        """
        if score >= threshold:
            return self._positive_feedback(score, metrics)
        else:
            return self._negative_feedback(score, metrics)

    @abstractmethod
    def _positive_feedback(self, score: float, metrics: Dict[str, float]) -> str:
        """Generate positive feedback for high-scoring examples"""
        pass

    @abstractmethod
    def _negative_feedback(self, score: float, metrics: Dict[str, float]) -> str:
        """Generate negative feedback with improvement suggestions"""
        pass


# =============================================================================
# Convenience Function: Run GEPA Optimization Directly
# =============================================================================

def optimize_prompt(
    adapter: RAGModuleAdapter,
    trainset: List[RAGDataInst],
    valset: Optional[List[RAGDataInst]] = None,
    max_metric_calls: int = 100,
    reflection_lm: str = "openai/gpt-4o-mini",
    seed_prompt: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run GEPA optimization to find the best prompt for a module.

    This is a convenience wrapper around gepa.optimize() that handles
    adapter setup and result extraction.

    Args:
        adapter: RAGModuleAdapter instance (QueryPlannerAdapter, etc.)
        trainset: Training data instances
        valset: Optional validation data instances
        max_metric_calls: Maximum number of evaluations (optimization budget)
        reflection_lm: Model for GEPA's reflective mutation
        seed_prompt: Optional seed prompt (uses module's current prompt if not provided)
        **kwargs: Additional arguments passed to gepa.optimize()

    Returns:
        Dict with:
            - best_prompt: The optimized prompt string
            - best_score: Score achieved by best prompt
            - all_results: Full GEPA results object

    Example:
        from gepa_adapters import QueryPlannerAdapter, optimize_prompt

        adapter = QueryPlannerAdapter(planner_module, retriever, evaluator)
        result = optimize_prompt(
            adapter=adapter,
            trainset=training_data,
            valset=validation_data,
            max_metric_calls=50,
        )
        print(f"Best prompt score: {result['best_score']}")
        planner_module.prompt = result['best_prompt']
    """
    if not GEPA_AVAILABLE:
        raise ImportError(
            "GEPA not installed. Install with: pip install gepa-ai\n"
            "See https://github.com/gepa-ai/gepa for documentation."
        )

    import gepa

    # Build seed candidate
    component_name = adapter.component_name
    if seed_prompt is not None:
        seed_candidate = {component_name: seed_prompt}
    else:
        seed_candidate = adapter.get_candidate()

    print(f"\n{'='*60}")
    print(f"GEPA Prompt Optimization: {component_name}")
    print(f"{'='*60}")
    print(f"Training examples: {len(trainset)}")
    print(f"Validation examples: {len(valset) if valset else 'Using trainset'}")
    print(f"Budget: {max_metric_calls} metric calls")
    print(f"Seed prompt length: {len(seed_candidate[component_name])} chars\n")

    # Run GEPA optimization
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        **kwargs,
    )

    # Extract best prompt and score from GEPAResult
    # GEPAResult has: candidates, val_aggregate_scores, best_idx, best_candidate
    best_idx = getattr(result, 'best_idx', 0)
    candidates = getattr(result, 'candidates', [])
    val_scores = getattr(result, 'val_aggregate_scores', [])
    
    # Get best candidate
    best_candidate = getattr(result, 'best_candidate', None)
    if best_candidate is None and candidates:
        best_candidate = candidates[best_idx] if best_idx < len(candidates) else candidates[0]
    
    best_prompt = best_candidate.get(component_name, "") if isinstance(best_candidate, dict) else ""
    
    # Get best score
    best_score = val_scores[best_idx] if val_scores and best_idx < len(val_scores) else 0.0

    print(f"\n{'='*60}")
    print(f"Optimization Complete")
    print(f"{'='*60}")
    print(f"Best score (valset): {best_score:.4f}")
    print(f"Total candidates: {len(candidates)}")
    print(f"Optimized prompt length: {len(best_prompt)} chars")

    return {
        "best_prompt": best_prompt,
        "best_score": best_score or 0.0,
        "component_name": component_name,
        "all_results": result,
    }
