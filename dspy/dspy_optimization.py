"""
DSPy-based prompt orchestration and lightweight reward utilities.

This module wraps DSPy signatures & modules so LangGraph stages can
compile/optimize prompts automatically using representative examples.
"""

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal
import inspect
import textwrap


import dspy
from dspy.teleprompt import BootstrapFewShot

dspy.configure(
    lm=dspy.LM(
        "openai/gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=float(os.environ.get("REFORM_TEMPERATURE", 0.3)),
        top_p=0.9,
    )
)

try:  # Import lazily to avoid forcing LangChain deps when unavailable.
    from LLMBase import LLMBase  # type: ignore
except Exception:  # pragma: no cover - fallback when optional deps missing
    LLMBase = None  # type: ignore


@dataclass
class PromptCallResult:
    stage: str
    output: Dict[str, Any]
    prompt_record: Dict[str, Any]
    raw: Any

StageName = Literal["reformulation", "reranking", "generation"]


_STAGE_PROMPT_KEYS: Dict[StageName, str] = {
    "reformulation": "prompt_reformation",
    "reranking": "document_reranking",
    "generation": "answer_generation",
}

_DEFAULT_STAGE_INSTRUCTIONS: Dict[StageName, str] = {
    "reformulation": "Rewrite the user query to improve retrieval quality.",
    "reranking": "Rank retrieved contexts by usefulness for the query.",
    "generation": "Answer grounded strictly in the provided contexts.",
}


def _load_stage_instructions() -> Dict[StageName, str]:
    """Pull system prompts from LLMBase without initializing remote LLM clients."""
    base = object.__new__(LLMBase) if LLMBase else None
    instructions: Dict[StageName, str] = {}

    for stage, prompt_key in _STAGE_PROMPT_KEYS.items():
        raw_prompt = ""
        try:
            if base is not None:
                raw_prompt = LLMBase.get_model_prompt(base, prompt_key) or ""  # type: ignore[arg-type]
        except Exception:
            raw_prompt = ""
        clean_prompt = textwrap.dedent(raw_prompt).strip()
        instructions[stage] = clean_prompt or _DEFAULT_STAGE_INSTRUCTIONS[stage]

    return instructions


STAGE_INSTRUCTIONS: Dict[StageName, str] = _load_stage_instructions()


class ReformulateSignature(dspy.Signature):
    """placeholder overwritten at runtime"""

    query: str = dspy.InputField(desc="Original user query.")
    feedback: str = dspy.InputField(desc="Brief guidance from prior retrieval performance.")
    rewritten_query: str = dspy.OutputField(desc="Improved query for retrieval.")


class RerankSignature(dspy.Signature):
    """placeholder overwritten at runtime"""

    query: str = dspy.InputField(desc="User question.")
    contexts: List[str] = dspy.InputField(desc="Retrieved context snippets.")
    ranked_contexts: List[str] = dspy.OutputField(desc="Contexts ordered most-to-least helpful.")


class GenerateSignature(dspy.Signature):
    """placeholder overwritten at runtime"""

    query: str = dspy.InputField(desc="Reformulated question.")
    context: str = dspy.InputField(desc="Top contexts joined as a single blob.")
    answer: str = dspy.OutputField(desc="Grounded final answer.")
    rationale: str = dspy.OutputField(desc="Short explanation or citations.")


ReformulateSignature.__doc__ = STAGE_INSTRUCTIONS["reformulation"]
RerankSignature.__doc__ = STAGE_INSTRUCTIONS["reranking"]
GenerateSignature.__doc__ = STAGE_INSTRUCTIONS["generation"]


class ReformulationJudgeSignature(dspy.Signature):
    """Score whether the rewritten query preserves + clarifies intent."""

    original_query: str = dspy.InputField()
    rewritten_query: str = dspy.InputField()
    score: float = dspy.OutputField(desc="Score in [0, 1].")
    verdict: str = dspy.OutputField()
    rationale: str = dspy.OutputField()


class RerankJudgeSignature(dspy.Signature):
    """Score whether contexts cover complementary evidence."""

    query: str = dspy.InputField()
    contexts: List[str] = dspy.InputField()
    score: float = dspy.OutputField(desc="Score in [0, 1].")
    verdict: str = dspy.OutputField()
    rationale: str = dspy.OutputField()


class GenerationJudgeSignature(dspy.Signature):
    """Score whether the answer is faithful and addresses the query."""

    query: str = dspy.InputField()
    answer: str = dspy.InputField()
    contexts: List[str] = dspy.InputField()
    reference: str = dspy.InputField(desc="Optional ground-truth answer.")
    score: float = dspy.OutputField(desc="Score in [0, 1].")
    verdict: str = dspy.OutputField()
    rationale: str = dspy.OutputField()


class DSPYPromptJudge:
    """Optional judge that scores stage outputs via DSPy modules."""

    def __init__(self, enable: bool = True) -> None:
        self.enabled = enable
        if not enable:
            self.modules: Dict[str, Any] = {}
            return
        self.modules = {
            "reformulation": dspy.ChainOfThought(ReformulationJudgeSignature),
            "reranking": dspy.ChainOfThought(RerankJudgeSignature),
            "generation": dspy.ChainOfThought(GenerationJudgeSignature),
        }

    def assess_reformulation(self, *, query: str, rewritten: str) -> Dict[str, Any]:
        return self._invoke("reformulation", original_query=query, rewritten_query=rewritten)

    def assess_reranking(self, *, query: str, contexts: List[str]) -> Dict[str, Any]:
        return self._invoke("reranking", query=query, contexts=contexts)

    def assess_generation(
        self,
        *,
        query: str,
        answer: str,
        contexts: List[str],
        reference: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._invoke("generation", query=query, answer=answer, contexts=contexts, reference=reference or "")

    def _invoke(self, stage: str, **kwargs: Any) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        module = self.modules.get(stage)
        if module is None:
            return {}
        prediction = module(**kwargs)
        return {
            "score": _safe_float(getattr(prediction, "score", None)),
            "verdict": getattr(prediction, "verdict", "") or "",
            "rationale": getattr(prediction, "rationale", "") or "",
        }


class DSPYPromptOrchestrator:
    """Construct DSPy modules per stage with optional judge feedback."""

    def __init__(
        self,
        *,
        enable_judge: bool = True,
        optimizer_factory: Optional[Callable[[str], Any]] = None,
        stage_instructions: Optional[Dict[StageName, str]] = None,
    ) -> None:
        self.stage_instructions = dict(stage_instructions or STAGE_INSTRUCTIONS)
        self.modules = {
            "reformulation": dspy.Predict(ReformulateSignature),
            "reranking": dspy.ChainOfThought(RerankSignature),
            "generation": dspy.ChainOfThought(GenerateSignature),
        }
        self.optimizer_factory = optimizer_factory or (lambda stage: BootstrapFewShot(max_bootstrapped_demos=4))
        self.judge = DSPYPromptJudge(enable=enable_judge)
        self.version = 0
        self.previous_modules: Optional[Dict[str, Any]] = None
        self.last_trainsets: Optional[Dict[str, List[dspy.Example]]] = None

    def _prompt_text(self, stage: StageName) -> str:
        mod = self.modules.get(stage)
        sig = getattr(mod, "signature", None)
        doc = getattr(sig, "__doc__", None)
        if doc:
            return inspect.cleandoc(doc)
        # fallback to stored defaults
        instructions = getattr(self, "stage_instructions", STAGE_INSTRUCTIONS)
        return instructions.get(stage, _DEFAULT_STAGE_INSTRUCTIONS[stage])

    def compile(
        self,
        trainsets: Dict[str, List[dspy.Example]],
        metric_fns: Optional[Dict[str, Callable[[Any, Any], float]]] = None,
    ) -> None:
        prev_modules = dict(self.modules)
        new_modules = dict(self.modules)
        for stage, examples in trainsets.items():
            if not examples:
                continue
            optimizer = self.optimizer_factory(stage)
            if metric_fns and stage in metric_fns and hasattr(optimizer, "metric"):
                optimizer.metric = metric_fns[stage]
            compiled = optimizer.compile(self.modules[stage], trainset=examples)
            new_modules[stage] = compiled
        self.previous_modules = prev_modules
        self.modules = new_modules
        self.last_trainsets = trainsets
        self.version += 1

    def rollback(self) -> None:
        """Revert to the previously compiled modules if available."""
        if self.previous_modules is None:
            return
        self.modules, self.previous_modules = self.previous_modules, None
        self.version = max(self.version - 1, 0)

    def reformulate(
        self,
        query: str,
        *,
        feedback: Optional[str] = None,
    ) -> PromptCallResult:
        payload = {
            "query": query,
            "feedback": feedback or "",
        }
        prediction = self.modules["reformulation"](**payload)
        rewritten = getattr(prediction, "rewritten_query", None) or query
        prompt_record = {
            "stage": "reformulation",
            "prompt_name": "dspy_reformulation",
            "prompt_text": self._prompt_text("reformulation"),
        }
        return PromptCallResult(
            stage="reformulation",
            output={"reformulated_query": rewritten},
            prompt_record=prompt_record,
            raw=prediction,
        )

    def rerank(self, query: str, contexts: List[str]) -> PromptCallResult:
        prediction = self.modules["reranking"](query=query, contexts=contexts)
        ranked = getattr(prediction, "ranked_contexts", None)
        if ranked is None:
            ranked_list = contexts
        elif isinstance(ranked, list):
            ranked_list = ranked
        else:
            ranked_list = [line.strip() for line in str(ranked).splitlines() if line.strip()]
        prompt_record = {
            "stage": "reranking",
            "prompt_name": "dspy_reranking",
            "prompt_text": self._prompt_text("reranking"),
        }
        return PromptCallResult(
            stage="reranking",
            output={"ranked_contexts": ranked_list},
            prompt_record=prompt_record,
            raw=prediction,
        )

    def generate(
        self,
        query: str,
        context: List[str],
        reference: Optional[str] = None,
    ) -> PromptCallResult:
        context_blob = "\n\n".join(context)
        prediction = self.modules["generation"](query=query, context=context_blob)
        answer = getattr(prediction, "answer", "") or ""
        rationale = getattr(prediction, "rationale", "") or ""
        prompt_record = {
            "stage": "generation",
            "prompt_name": "dspy_generation",
            "prompt_text": self._prompt_text("generation"),
        }
        return PromptCallResult(
            stage="generation",
            output={"answer": answer, "rationale": rationale},
            prompt_record=prompt_record,
            raw=prediction,
        )


def build_trainsets_from_representatives(
    representatives: List[Dict[str, Any]],
    *,
    context_provider: Callable[[Dict[str, Any]], List[str]],
    rewrite_provider: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> Dict[str, List[dspy.Example]]:
    trainsets: Dict[str, List[dspy.Example]] = {
        "reformulation": [],
        "reranking": [],
        "generation": [],
    }

    for rep in representatives:
        query = rep.get("query", "")
        if not query:
            continue
        contexts = context_provider(rep) or []
        ground_truth = rep.get("ground_truth") or ""

        target_rewrite = rewrite_provider(rep) if rewrite_provider else query
        trainsets["reformulation"].append(
            dspy.Example(
                query=query,
                rewritten_query=target_rewrite,
                feedback="",
            ).with_inputs("query", "feedback")
        )

        if contexts:
            trainsets["reranking"].append(
                dspy.Example(
                    query=query,
                    contexts=contexts,
                    ranked_contexts=contexts,
                ).with_inputs("query", "contexts")
            )

        if ground_truth and contexts:
            trainsets["generation"].append(
                dspy.Example(
                    query=query,
                    context="\n\n".join(contexts),
                    answer=ground_truth,
                    rationale="",
                ).with_inputs("query", "context")
            )

    return trainsets


def build_trainsets_from_results(results: List[Dict[str, Any]]) -> Dict[str, List[dspy.Example]]:
    """Create DSPy trainsets from finished RAG runs."""
    trainsets: Dict[str, List[dspy.Example]] = {
        "reformulation": [],
        "reranking": [],
        "generation": [],
    }

    for res in results:
        if not isinstance(res, dict):
            continue
        query = res.get("query") or ""
        if not query:
            continue
        rewritten = res.get("reformulated_query") or query
        contexts = res.get("final_contexts") or res.get("contexts") or []
        ground_truth = res.get("ground_truth") or res.get("answer_reference") or ""
        answer = res.get("answer") or ""
        rationale = res.get("generation_rationale") or ""

        trainsets["reformulation"].append(
            dspy.Example(
                query=query,
                rewritten_query=rewritten,
                feedback="",
            ).with_inputs("query", "feedback")
        )

        if contexts:
            trainsets["reranking"].append(
                dspy.Example(
                    query=query,
                    contexts=contexts,
                    ranked_contexts=contexts,
                ).with_inputs("query", "contexts")
            )

        if contexts:
            if ground_truth:
                target_answer = ground_truth
            elif answer:
                target_answer = answer
            else:
                target_answer = ""
            if target_answer:
                trainsets["generation"].append(
                    dspy.Example(
                        query=rewritten,
                        context="\n\n".join(contexts),
                        answer=target_answer,
                        rationale=rationale,
                    ).with_inputs("query", "context")
                )

    return trainsets


def optimize_prompts_with_representatives(
    orchestrator: DSPYPromptOrchestrator,
    representatives: List[Dict[str, Any]],
    *,
    context_provider: Callable[[Dict[str, Any]], List[str]],
    rewrite_provider: Optional[Callable[[Dict[str, Any]], str]] = None,
    metric_fns: Optional[Dict[str, Callable[[Any, Any], float]]] = None,
) -> Dict[str, List[dspy.Example]]:
    """
    Convenience helper to (1) build DSPy trainsets from representative queries and
    (2) compile the orchestrator so each stage instruction is optimized.
    """
    trainsets = build_trainsets_from_representatives(
        representatives,
        context_provider=context_provider,
        rewrite_provider=rewrite_provider,
    )
    orchestrator.compile(trainsets, metric_fns=metric_fns)
    return trainsets


class RewardManager:
    """Metric-driven rewards with optional judge blending."""

    def __init__(
        self,
        memory_weight: float = 0.15,
        window: int = 3,
        judge_weight: float = 0.0,
    ) -> None:
        self.memory_weight = memory_weight
        self.window = window
        self.judge_weight = judge_weight

    def compute(
        self,
        stage: str,
        state_snapshot: Dict[str, Any],
        memory: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        metrics = self._metrics_for_stage(stage, state_snapshot)
        base_reward = metrics.get("base_reward", 0.0)
        memory_bonus = self._memory_bonus(memory or [])
        total = max(min(base_reward + memory_bonus, 1.0), -1.0)
        breakdown = {
            "base": base_reward,
            "memory": memory_bonus,
            "total": total,
        }
        return total, breakdown

    def _metrics_for_stage(self, stage: str, state_snapshot: Dict[str, Any]) -> Dict[str, float]:
        if stage == "reformulation":
            metrics = state_snapshot.get("retrieval_eval") or {}
            precision = metrics.get("context_precision", 0.0)
            recall = metrics.get("context_recall", 0.0)
            base = self._safe_avg([precision, recall])
        elif stage == "reranking":
            metrics = state_snapshot.get("reranking_eval") or {}
            base = metrics.get("context_precision", 0.0)
        else:  # generation
            metrics = state_snapshot.get("generation_eval") or {}
            faithfulness = metrics.get("faithfulness", state_snapshot.get("faithfulness_score", 0.0))
            relevancy = metrics.get("answer_relevancy", state_snapshot.get("answer_relevancy_score", 0.0))
            base = self._safe_avg([faithfulness, relevancy])

        judge_score = None
        judge_bucket = state_snapshot.get("judge_eval") or {}
        if isinstance(judge_bucket, dict):
            judge_stage = judge_bucket.get(stage)
            if isinstance(judge_stage, dict):
                judge_score = judge_stage.get("score")

        if judge_score is not None and self.judge_weight > 0:
            base = (1 - self.judge_weight) * base + self.judge_weight * judge_score

        return {"base_reward": base}

    def _memory_bonus(self, memory: List[Dict[str, Any]]) -> float:
        if not memory:
            return 0.0
        recent = memory[-self.window :]
        rewards = [entry.get("reward", 0.0) for entry in recent if entry is not None]
        if not rewards:
            return 0.0
        return sum(rewards) / len(rewards) * self.memory_weight

    @staticmethod
    def _safe_avg(values: List[float]) -> float:
        non_null = [v for v in values if v is not None]
        if not non_null:
            return 0.0
        return sum(non_null) / len(non_null)


def _safe_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
