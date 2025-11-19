from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
import math
import random
from collections import deque

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


@dataclass
class PromptOption:
    """Container for a discrete prompt template."""

    name: str
    template: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def _softmax(logits: List[float], temperature: float) -> List[float]:
    if not logits:
        return []
    scaled = [logit / max(temperature, 1e-6) for logit in logits]
    max_val = max(scaled)
    exp_values = [math.exp(val - max_val) for val in scaled]
    total = sum(exp_values)
    if total == 0:
        return [1.0 / len(logits) for _ in logits]
    return [val / total for val in exp_values]


def _normalize_score(score: Optional[float], fallback: float = 0.0) -> float:
    if score is None:
        return fallback
    # Map [0, 1] → [-1, 1] and clamp to safety bounds.
    return max(min((score - 0.5) * 2.0, 1.0), -1.0)


class PromptRLPolicy:
    """
    Minimal categorical policy with REINFORCE-style updates over discrete prompts.
    """

    def __init__(
        self,
        options: List[PromptOption],
        lr: float = 0.15,
        temperature: float = 1.0,
        baseline_momentum: float = 0.9,
        reward_window: int = 25,
    ) -> None:
        if not options:
            raise ValueError("PromptRLPolicy requires at least one prompt option.")
        self.options = options
        self.lr = lr
        self.temperature = temperature
        self.baseline_momentum = min(max(baseline_momentum, 0.0), 0.999)
        self.logits: List[float] = [0.0 for _ in options]
        self.baseline: float = 0.0
        self.reward_history: Deque[float] = deque(maxlen=reward_window)
        self.last_probs: Optional[List[float]] = None

    def sample(self, deterministic: bool = False) -> Tuple[int, PromptOption, List[float]]:
        probs = _softmax(self.logits, self.temperature)
        self.last_probs = probs
        if deterministic:
            index = max(range(len(probs)), key=lambda i: probs[i])
        else:
            draw = random.random()
            cumulative = 0.0
            index = len(probs) - 1
            for i, prob in enumerate(probs):
                cumulative += prob
                if draw <= cumulative:
                    index = i
                    break
        return index, self.options[index], probs

    def normalize_reward(self, reward: float) -> float:
        self.reward_history.append(reward)
        if len(self.reward_history) < 2:
            return reward
        mean = sum(self.reward_history) / len(self.reward_history)
        variance = sum((r - mean) ** 2 for r in self.reward_history) / len(self.reward_history)
        std = math.sqrt(max(variance, 1e-6))
        return (reward - mean) / std

    def update(self, action_index: int, normalized_reward: float, probs: Optional[List[float]] = None) -> None:
        probs = probs or self.last_probs or _softmax(self.logits, self.temperature)
        if not probs:
            return
        advantage = normalized_reward - self.baseline
        self.baseline = (
            self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * normalized_reward
        )
        one_hot = [1.0 if i == action_index else 0.0 for i in range(len(self.logits))]
        for i, logit in enumerate(self.logits):
            grad = advantage * (one_hot[i] - probs[i])
            updated = logit + self.lr * grad
            self.logits[i] = max(min(updated, 10.0), -10.0)

class JudgeAssessment(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Overall quality score in [0,1]")
    rationale: str = Field(description="Explanation of the score")
    verdict: str = Field(description="One-line verdict summarizing the assessment")


class LLMJudge:
    """Lightweight LLM-based judge that scores stage outputs."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        if model_name == "gpt-4o-mini":
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        elif model_name == "gemini-2.5-flash":
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        else:
            raise ValueError(f"Unsupported judge model: {model_name}")

    def assess_reformulation(self, original_query: str, rewritten_query: str) -> Dict[str, Any]:
        """Score how well the reformulated query clarifies and enriches the original."""
        if not (original_query and rewritten_query):
            return {}
        system_prompt = (
            "You are judging a query reformulation module for retrieval-augmented generation. "
            "Reward rewritten queries that preserve intent, add clarifying details, remove ambiguity, "
            "and remain concise. Penalize hallucinated entities or loss of essential constraints."
        )
        user_prompt = (
            f"Original Query:\n{original_query}\n\n"
            f"Rewritten Query:\n{rewritten_query}\n"
        )
        return self._structured_score(system_prompt, user_prompt)

    def assess_reranking(self, query: str, contexts: List[str]) -> Dict[str, Any]:
        """Score whether the reranked contexts are diverse, relevant, and evidence-rich."""
        if not (query and contexts):
            return {}
        context_blob = "\n\n".join(contexts[:8])
        system_prompt = (
            "You evaluate the ordering of retrieved passages for a question. "
            "Reward selections that prioritize passages directly answering the question, "
            "cover complementary aspects, and avoid redundant or speculative text."
        )
        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Top Reranked Passages:\n{context_blob}\n"
        )
        return self._structured_score(system_prompt, user_prompt)

    def assess_generation(self, query: str, contexts: List[str], answer: str) -> Dict[str, Any]:
        """Score the final answer for grounding and completeness."""
        if not (query and contexts and answer):
            return {}
        context_blob = "\n\n".join(contexts[:8])
        system_prompt = (
            "You are an impartial judge for retrieval-augmented generation. "
            "Score how well the answer is supported by the provided context, ensuring factual accuracy "
            "and coverage. Penalize unsupported claims or missing key facts."
        )
        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Context:\n{context_blob}\n\n"
            f"Answer:\n{answer}\n"
        )
        return self._structured_score(system_prompt, user_prompt)

    def _structured_score(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call the judge model and return the structured assessment."""
        structured_llm = self.llm.with_structured_output(JudgeAssessment)
        response = structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return {
            "score": float(response.score),
            "rationale": response.rationale,
            "verdict": response.verdict,
        }


class PromptOptimizationCycle:
    """
    Coordinates RL policies for each promptable stage of the RAG pipeline.
    """

    def __init__(
        self,
        stage_options: Optional[Dict[str, List[PromptOption]]] = None,
        stage_reward_fns: Optional[Dict[str, Callable[[Dict[str, Any]], float]]] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        judge_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.stage_options = stage_options or self._default_stage_options()
        self.stage_reward_fns = stage_reward_fns or self._build_default_reward_fns()
        self.policy_kwargs = policy_kwargs or {}
        self.policies: Dict[str, PromptRLPolicy] = {
            stage: PromptRLPolicy(options, **self.policy_kwargs) for stage, options in self.stage_options.items()
        }
        self.current_iteration: Optional[int] = None
        self.active_records: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.judge = None
        self._judge_cache: Dict[Any, Dict[str, Any]] = {}
        judge_settings = None if judge_config is False else (judge_config or {})
        if judge_settings is not None:
            try:
                self.judge = LLMJudge(**judge_settings)
            except Exception as err:
                print(f"[PromptOptimization] Failed to initialize LLM judge: {err}")
                self.judge = None

    def reset_active_records(self) -> None:
        self.current_iteration = None
        self.active_records = []
        self._judge_cache.clear()

    def select_prompt(
        self,
        stage: str,
        iteration_id: int,
        context: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        if stage not in self.policies:
            raise ValueError(f"Unknown prompt stage '{stage}'. Available: {list(self.policies)}")
        if self.current_iteration != iteration_id:
            self.current_iteration = iteration_id
            self.active_records = []
        policy = self.policies[stage]
        action_index, option, probs = policy.sample(deterministic=deterministic)
        record = {
            "iteration": iteration_id,
            "stage": stage,
            "prompt_name": option.name,
            "prompt_template": option.template,
            "description": option.description,
            "action_index": action_index,
            "selection_prob": probs[action_index],
            "context": context or {},
        }
        self.active_records.append(record)
        return option.template, record

    def finalize_iteration(
        self,
        state_snapshot: Dict[str, Any],
        stages: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.active_records:
            return []
        allowed = set(stages) if stages else None
        to_finalize = (
            [record for record in self.active_records if record["stage"] in allowed]
            if allowed is not None
            else list(self.active_records)
        )
        if not to_finalize:
            return []
        if allowed is None:
            self.active_records = []
        else:
            self.active_records = [record for record in self.active_records if record["stage"] not in allowed]
        finalized: List[Dict[str, Any]] = []
        for record in to_finalize:
            stage = record["stage"]
            policy = self.policies[stage]
            reward, breakdown = self._compute_reward(stage, state_snapshot)
            normalized_reward = policy.normalize_reward(reward)
            policy.update(record["action_index"], normalized_reward)
            enriched = dict(record)
            enriched["reward"] = reward
            enriched["normalized_reward"] = normalized_reward
            if stage == "reformulation":
                retrieval_eval = state_snapshot.get("retrieval_eval") or {}
                enriched["metrics_snapshot"] = {
                    "retrieval_confidence": retrieval_eval.get("retrieval_confidence"),
                    "top_score": retrieval_eval.get("top_score"),
                    "agreement": retrieval_eval.get("bm25_dense_agreement"),
                }
            elif stage == "reranking":
                reranking_eval = state_snapshot.get("reranking_eval") or {}
                enriched["metrics_snapshot"] = {
                    "reranking_confidence": reranking_eval.get("reranking_confidence"),
                    "improvement": reranking_eval.get("score_improvement"),
                    "concentration": reranking_eval.get("score_concentration"),
                }
            else:  # generation
                enriched["metrics_snapshot"] = {
                    "faithfulness": state_snapshot.get("faithfulness_score"),
                    "answer_accuracy": state_snapshot.get("answer_accuracy_score"),
                    "context_precision": state_snapshot.get("context_precision_score"),
                    "context_recall": state_snapshot.get("context_recall_score"),
                    "overall": state_snapshot.get("overall_score"),
                }
            enriched["reward_breakdown"] = breakdown
            finalized.append(enriched)
        self.history.extend(finalized)
        return finalized

    def _compute_reward(self, stage: str, state_snapshot: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Derive the reward for a stage using only the LLM judge (fallbacks optional)."""
        reward_fn = self.stage_reward_fns.get(stage)
        fallback_reward = reward_fn(state_snapshot) if reward_fn else 0.0
        judge_result = self._evaluate_with_judge(stage, state_snapshot)
        if judge_result:
            state_snapshot.setdefault("judge_eval", {})[stage] = judge_result
            judge_score = _normalize_score(judge_result.get("score"))
        else:
            judge_score = 0.0
        total = max(min(judge_score if judge_result else fallback_reward, 1.0), -1.0)
        breakdown = {
            "judge": judge_score,
            "fallback": fallback_reward if not judge_result else 0.0,
            "total": total,
        }
        return total, breakdown

    def _build_default_reward_fns(self) -> Dict[str, Callable[[Dict[str, Any]], float]]:
        """Provide optional fallback reward hooks; defaults return 0 to rely on the judge."""

        def zero_reward(_: Dict[str, Any]) -> float:
            return 0.0

        return {
            "reformulation": zero_reward,
            "reranking": zero_reward,
            "generation": zero_reward,
        }

    def _evaluate_with_judge(self, stage: str, state_snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dispatch stage data to the judge and cache the structured result."""
        if not self.judge:
            return None
        try:
            if stage == "reformulation":
                original = state_snapshot.get("query")
                rewritten = state_snapshot.get("reformulated_query")
                cache_key = ("reformulation", state_snapshot.get("iteration"), hash(rewritten))
                if cache_key in self._judge_cache:
                    return self._judge_cache[cache_key]
                result = self.judge.assess_reformulation(original_query=original, rewritten_query=rewritten)
            elif stage == "reranking":
                contexts = state_snapshot.get("final_contexts") or state_snapshot.get("contexts") or []
                cache_key = ("reranking", state_snapshot.get("iteration"), len(contexts))
                if cache_key in self._judge_cache:
                    return self._judge_cache[cache_key]
                result = self.judge.assess_reranking(query=state_snapshot.get("query"), contexts=contexts)
            else:
                contexts = state_snapshot.get("final_contexts") or []
                cache_key = ("generation", state_snapshot.get("iteration"), hash(state_snapshot.get("answer")), len(contexts))
                if cache_key in self._judge_cache:
                    return self._judge_cache[cache_key]
                result = self.judge.assess_generation(
                    query=state_snapshot.get("query"),
                    contexts=contexts,
                    answer=state_snapshot.get("answer"),
                )
        except Exception as err:
            print(f"[PromptOptimization] Judge evaluation failed: {err}")
            return None
        if result:
            self._judge_cache[cache_key] = result
        return result

    @staticmethod
    def _default_stage_options() -> Dict[str, List[PromptOption]]:
        return {
            "reformulation": [
                PromptOption(
                    name="precision_reformulator",
                    description="Bias toward entity disambiguation and structured constraints.",
                    template=(
                        "You rewrite analyst queries to maximize retrieval precision.\n"
                        "• Highlight explicit entities, dates, metrics.\n"
                        "• Inject synonyms and ticker symbols when available.\n"
                        "• Preserve intent; prefer unambiguous wording.\n"
                        "Return only the rewritten query."
                    ),
                ),
                PromptOption(
                    name="contextual_expander",
                    description="Adds related domain terminology for richer recall.",
                    template=(
                        "Act as a senior researcher expanding research questions for hybrid retrieval.\n"
                        "Include alternate phrasings, domain taxonomies, and contextual qualifiers that "
                        "improve dense + sparse search. Avoid introducing unsupported requirements. "
                        "Output the enhanced query."
                    ),
                ),
                PromptOption(
                    name="minimalist_grounder",
                    description="Keeps query short but enforces grounding signals.",
                    template=(
                        "Rewrite the question with the smallest number of high-signal terms needed for "
                        "grounded lookup. Emphasize cited entities and measurable attributes. "
                        "Do not add speculative language."
                    ),
                ),
            ],
            "reranking": [
                PromptOption(
                    name="faithfulness_guard",
                    description="Promotes chunks with factual alignment and cite-worthiness.",
                    template=(
                        "You evaluate retrieved snippets for a question. Rank chunks highest when they\n"
                        "contain verifiable facts that directly answer the query. Penalize redundant or "
                        "speculative passages. Return the chunk texts ordered most-to-least helpful."
                    ),
                ),
                PromptOption(
                    name="coverage_balancer",
                    description="Prefers complementary evidence to maximize coverage.",
                    template=(
                        "Given multiple retrieved passages, order them so that the top entries collectively "
                        "cover distinct aspects of the query. Prefer passages with unique details or "
                        "figures over near duplicates. Respond with the reordered chunk texts."
                    ),
                ),
                PromptOption(
                    name="risk_averse_filter",
                    description="Demotes chunks likely to trigger hallucinations.",
                    template=(
                        "Re-rank the passages for safe answer generation. Boost chunks that mention the key "
                        "entities and provide explicit evidence. Demote passages that speculate, hedge, or "
                        "lack concrete data. Return the reordered texts."
                    ),
                ),
            ],
            "generation": [
                PromptOption(
                    name="evidence_first_answer",
                    description="Forces citation-style grounding.",
                    template=(
                        "You compose answers from provided context only.\n"
                        "Instruction:\n"
                        "1. Start with the direct answer in one sentence.\n"
                        "2. Support it with bullet points referencing the source snippets.\n"
                        "3. State 'Information unavailable' when context is insufficient.\n"
                        "Context: {context}\n"
                        "Question: {query}"
                    ),
                ),
                PromptOption(
                    name="structured_briefing",
                    description="Encourages structured summaries tying back to context.",
                    template=(
                        "Act as an analyst preparing a briefing strictly from the supplied context.\n"
                        "Produce sections: 'Answer', 'Supporting Evidence', 'Open Questions'.\n"
                        "Answer must cite the specific facts quoted. If data conflicts, explain.\n"
                        "Context: {context}\n"
                        "Question: {query}"
                    ),
                ),
                PromptOption(
                    name="contrastive_reporter",
                    description="Highlights agreements/conflicts in context.",
                    template=(
                        "You compare all context passages to craft an answer.\n"
                        "• Extract overlapping facts first.\n"
                        "• Then list any conflicting statements and how they affect confidence.\n"
                        "Only rely on provided context.\n"
                        "Context: {context}\n"
                        'Question: {query}\nAnswer in the format "Answer / Evidence / Conflicts".'
                    ),
                ),
            ],
        }

    @staticmethod
    def _default_reward_fns() -> Dict[str, Callable[[Dict[str, Any]], float]]:
        return {
            "reformulation": lambda metrics: _safe_average(
                [metrics.get("context_precision_score"), metrics.get("answer_accuracy_score")], fallback=0.0
            ),
            "reranking": lambda metrics: _safe_average(
                [metrics.get("faithfulness_score"), metrics.get("context_precision_score")], fallback=0.0
            ),
            "generation": lambda metrics: _safe_average(
                [metrics.get("faithfulness_score"), metrics.get("answer_accuracy_score")], fallback=0.0
            ),
        }
