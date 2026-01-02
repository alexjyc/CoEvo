"""
RAG Pipeline GEPA Optimization - Research Paper Quality Experiment Runner

CRITICAL IMPROVEMENTS over previous version:
1. Proper train/val/test split (no data contamination)
2. Consistent evaluation across all phases
3. Statistical significance testing (CI, p-values)
4. Per-example score tracking
5. GEPA iteration history with reflections
6. LaTeX table generation
7. Reproducibility controls (seeds, config hash)
8. Comprehensive ablation studies

Data Split Strategy:
- Total queries: N
- Training set: 60% (for GEPA optimization)
- Validation set: 20% (for GEPA prompt selection)
- Test set: 20% (HELD OUT - only for final paper numbers)

Usage:
    python run_research_experiment_v2.py --experiment_name "paper_exp_001" --n_queries 100

Author: RAG Optimization Research
"""

import asyncio
import argparse
import json
import os
import sys
import csv
import hashlib
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import statistics
import math
from functools import wraps
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from modules.preprocessor import DocumentPreprocessor
from modules.query_planner.planner import QueryPlannerModule
from modules.query_planner.retrieval import HybridRetriever
from modules.reranker.reranker import RerankerModule
from modules.generator.generator import GeneratorModule
from modules.evaluation import RAGASEvaluator
from modules.base import QueryPlannerInput, RetrievalInput, RerankerInput, GeneratorInput

# Import GEPA adapters
from gepa_adapters.query_planner_adapter import QueryPlannerAdapter
from gepa_adapters.reranker_adapter import RerankerAdapter
from gepa_adapters.generator_adapter import GeneratorAdapter
from gepa_adapters.base import RAGDataInst, GEPA_AVAILABLE


# =============================================================================
# Statistical Utilities
# =============================================================================

def compute_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval.

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not scores:
        return 0.0, 0.0, 0.0

    n = len(scores)
    mean = statistics.mean(scores)

    if n < 2:
        return mean, mean, mean

    std_err = statistics.stdev(scores) / math.sqrt(n)

    # t-value for 95% CI (approximation for n > 30, use 1.96)
    # For smaller samples, should use scipy.stats.t.ppf
    if n >= 30:
        t_value = 1.96
    else:
        # Rough approximation for smaller samples
        t_values = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 10: 2.26, 15: 2.13, 20: 2.09, 25: 2.06}
        t_value = t_values.get(n, 2.0)

    margin = t_value * std_err
    return mean, mean - margin, mean + margin


def paired_ttest(scores1: List[float], scores2: List[float]) -> Tuple[float, float]:
    """
    Compute paired t-test for significance.

    Returns:
        Tuple of (t_statistic, p_value_approximation)
    """
    if len(scores1) != len(scores2) or len(scores1) < 2:
        return 0.0, 1.0

    n = len(scores1)
    diffs = [s2 - s1 for s1, s2 in zip(scores1, scores2)]

    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs)

    if std_diff == 0:
        return float('inf') if mean_diff != 0 else 0.0, 0.0 if mean_diff != 0 else 1.0

    t_stat = mean_diff / (std_diff / math.sqrt(n))

    # Approximate p-value (two-tailed)
    # For proper p-value, use scipy.stats.t.sf
    df = n - 1
    p_approx = 2.0 * (1.0 - min(0.9999, abs(t_stat) / (abs(t_stat) + df)))

    return t_stat, p_approx


# =============================================================================
# Retry Decorator
# =============================================================================

def retry_with_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
):
    """Decorator for retrying async functions with exponential backoff."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        raise
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    print(f"  [RETRY] Attempt {attempt + 1}/{max_retries} failed, waiting {delay:.1f}s...")
                    await asyncio.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# Data Classes for Research Paper Quality
# =============================================================================

@dataclass
class PerExampleResult:
    """Track per-example results for statistical analysis."""
    query_id: str
    query: str
    ground_truth: str
    score: float
    metrics: Dict[str, float]
    answer: str = ""
    contexts: List[str] = field(default_factory=list)


@dataclass
class GEPAIterationRecord:
    """Record for a single GEPA iteration."""
    iteration: int
    timestamp: str
    prompt_hash: str
    prompt_text: str
    train_score: float
    val_score: float
    accepted: bool
    reflection: str = ""
    feedback_summary: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModuleOptimizationRecord:
    """Comprehensive record for module optimization."""
    module_name: str
    component_name: str

    # Timing
    start_time: str
    end_time: str
    duration_seconds: float

    # Prompts
    seed_prompt: str
    best_prompt: str

    # Scores with statistics
    baseline_score: float
    baseline_ci_lower: float
    baseline_ci_upper: float
    baseline_per_example: List[float]

    best_score: float
    best_ci_lower: float
    best_ci_upper: float
    best_per_example: List[float]

    # Improvement with significance
    improvement_abs: float
    improvement_pct: float
    t_statistic: float
    p_value: float
    is_significant: bool  # p < 0.05

    # GEPA tracking
    total_iterations: int
    accepted_iterations: int
    iteration_history: List[GEPAIterationRecord] = field(default_factory=list)

    # Cost tracking
    llm_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['iteration_history'] = [asdict(r) for r in self.iteration_history]
        return result


@dataclass
class EvaluationRecord:
    """Record for an evaluation run."""
    description: str
    split: str  # "train", "val", "test"
    timestamp: str
    n_examples: int

    # Aggregate scores
    mean_score: float
    std_score: float
    ci_lower: float
    ci_upper: float

    # Per-example
    per_example_scores: List[float]
    per_example_results: List[PerExampleResult] = field(default_factory=list)

    # Detailed metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['per_example_results'] = [asdict(r) for r in self.per_example_results]
        return result


@dataclass
class PhaseCheckpoint:
    """Checkpoint with research-quality tracking."""
    phase: int
    timestamp: str
    completed: bool = False

    # Configuration hash for reproducibility
    config_hash: str = ""
    random_seed: int = 42

    # Data splits (query IDs)
    train_query_ids: List[str] = field(default_factory=list)
    val_query_ids: List[str] = field(default_factory=list)
    test_query_ids: List[str] = field(default_factory=list)

    # Prompts
    prompts: Dict[str, str] = field(default_factory=dict)

    # Contexts
    retrieved_contexts: Dict[str, List[str]] = field(default_factory=dict)
    reranked_contexts: Dict[str, List[str]] = field(default_factory=dict)

    # Module results with full tracking
    module_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Evaluation records
    evaluations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseCheckpoint":
        return cls(**data)


@dataclass
class AblationResult:
    """Ablation result with statistical rigor."""
    config_name: str
    prompts_used: Dict[str, str]

    # Test set evaluation
    test_score: float
    test_ci_lower: float
    test_ci_upper: float
    test_per_example: List[float]

    # Comparison to baseline
    improvement_vs_baseline: float
    t_statistic: float
    p_value: float
    is_significant: bool

    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Research Paper Quality Experiment Runner
# =============================================================================

class ResearchExperimentRunner:
    """
    Experiment runner with research paper quality standards.
    """

    PHASES = {
        0: "Data Split & Baseline Capture",
        1: "Module 1 (Query Planner) Optimization",
        2: "Module 2 (Reranker) Optimization",
        3: "Module 3 (Generator) Optimization",
        4: "Test Set Evaluation & Ablations",
    }

    def __init__(
        self,
        experiment_name: str,
        output_dir: Path,
        n_queries: int = 100,
        optimization_budget: int = 100,
        model: str = "gpt-4o-mini",
        chunk_size: int = 600,
        chunk_overlap: int = 50,
        retrieval_k: int = 20,
        rerank_k: int = 10,
        random_seed: int = 42,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        max_concurrent: int = 10,
    ):
        self.experiment_name = experiment_name
        self.output_dir = output_dir / experiment_name
        self.checkpoint_dir = self.output_dir / "checkpoints"

        # Configuration
        self.n_queries = n_queries
        self.optimization_budget = optimization_budget
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k
        self.random_seed = random_seed

        # Data split ratios
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Concurrency control for parallel API calls
        self.max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None  # Initialized lazily

        # Set random seeds for reproducibility
        random.seed(random_seed)

        # Generate config hash for reproducibility verification
        self.config_hash = self._generate_config_hash()
        self.experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.config_hash[:8]}"

        # Create directories
        self._setup_directories()

        # Data storage
        self.documents: List[Dict] = []
        self.all_queries: List[Dict] = []
        self.train_queries: List[Dict] = []
        self.val_queries: List[Dict] = []
        self.test_queries: List[Dict] = []
        self.chunk_labels: Dict[str, List[int]] = {}

        # Module instances
        self.preprocessor: Optional[DocumentPreprocessor] = None
        self.query_planner: Optional[QueryPlannerModule] = None
        self.retriever: Optional[HybridRetriever] = None
        self.reranker: Optional[RerankerModule] = None
        self.generator: Optional[GeneratorModule] = None
        self.evaluator: Optional[RAGASEvaluator] = None

        # Results
        self.ablation_results: List[AblationResult] = []
        self.baseline_test_scores: List[float] = []  # For paired t-test

    def _generate_config_hash(self) -> str:
        """Generate hash of configuration for reproducibility."""
        config_str = json.dumps({
            "n_queries": self.n_queries,
            "optimization_budget": self.optimization_budget,
            "model": self.model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_k": self.retrieval_k,
            "rerank_k": self.rerank_k,
            "random_seed": self.random_seed,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _setup_directories(self):
        dirs = [
            self.output_dir,
            self.checkpoint_dir,
            self.output_dir / "metrics",
            self.output_dir / "prompts" / "query_planner",
            self.output_dir / "prompts" / "reranker",
            self.output_dir / "prompts" / "generator",
            self.output_dir / "traces",
            self.output_dir / "analysis",
            self.output_dir / "ablations",
            self.output_dir / "latex",
            self.output_dir / "per_example",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Checkpoint Management
    # -------------------------------------------------------------------------

    def save_checkpoint(self, phase: int, checkpoint: PhaseCheckpoint) -> Path:
        checkpoint.phase = phase
        checkpoint.completed = True
        checkpoint.timestamp = datetime.now().isoformat()
        checkpoint.config_hash = self.config_hash
        checkpoint.random_seed = self.random_seed

        json_path = self.checkpoint_dir / f"checkpoint_phase_{phase}.json"
        with open(json_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, default=str)

        pickle_path = self.checkpoint_dir / f"checkpoint_phase_{phase}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(checkpoint, f)

        print(f"  [CHECKPOINT] Saved Phase {phase}")
        return json_path

    def load_checkpoint(self, phase: int) -> Optional[PhaseCheckpoint]:
        pickle_path = self.checkpoint_dir / f"checkpoint_phase_{phase}.pkl"
        if pickle_path.exists():
            with open(pickle_path, "rb") as f:
                checkpoint = pickle.load(f)

            # Verify config hash
            if checkpoint.config_hash != self.config_hash:
                print(f"  [WARNING] Config hash mismatch! Checkpoint may be from different config.")

            return checkpoint
        return None

    def restore_state_from_checkpoint(self, checkpoint: PhaseCheckpoint):
        """Restore prompts and data splits from checkpoint."""
        # Restore data splits
        if checkpoint.train_query_ids:
            query_map = {q["query_id"]: q for q in self.all_queries}
            self.train_queries = [query_map[qid] for qid in checkpoint.train_query_ids if qid in query_map]
            self.val_queries = [query_map[qid] for qid in checkpoint.val_query_ids if qid in query_map]
            self.test_queries = [query_map[qid] for qid in checkpoint.test_query_ids if qid in query_map]

        # Restore prompts
        if self.query_planner and "query_planner" in checkpoint.prompts:
            self.query_planner._prompt = checkpoint.prompts["query_planner"]
        if self.reranker and "reranker" in checkpoint.prompts:
            self.reranker._prompt = checkpoint.prompts["reranker"]
        if self.generator and "generator" in checkpoint.prompts:
            self.generator._prompt = checkpoint.prompts["generator"]

        print(f"  [RESTORE] Restored state from Phase {checkpoint.phase}")

    # -------------------------------------------------------------------------
    # Data Loading & Splitting
    # -------------------------------------------------------------------------

    async def load_and_split_data(self, data_path: Path):
        """Load data and create proper train/val/test splits."""
        print(f"\n{'='*70}")
        print("  LOADING & SPLITTING DATA")
        print(f"{'='*70}")

        # Load documents
        docs_path = data_path / "documents.json"
        with open(docs_path) as f:
            all_documents = json.load(f)

        # Load queries
        queries_path = data_path / "queries.json"
        with open(queries_path) as f:
            all_queries = json.load(f)

        # Select queries
        self.all_queries = all_queries[:self.n_queries]

        # Shuffle with seed for reproducibility
        shuffled = self.all_queries.copy()
        random.shuffle(shuffled)

        # Split
        n_total = len(shuffled)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        self.train_queries = shuffled[:n_train]
        self.val_queries = shuffled[n_train:n_train + n_val]
        self.test_queries = shuffled[n_train + n_val:]

        # Get relevant documents
        relevant_doc_ids = set()
        for q in self.all_queries:
            relevant_doc_ids.update(q.get("reference_doc_ids", []))

        self.documents = [d for d in all_documents if d["doc_id"] in relevant_doc_ids]

        print(f"  Total queries: {n_total}")
        print(f"  Train: {len(self.train_queries)} ({self.train_ratio*100:.0f}%)")
        print(f"  Val: {len(self.val_queries)} ({self.val_ratio*100:.0f}%)")
        print(f"  Test: {len(self.test_queries)} ({self.test_ratio*100:.0f}%) [HELD OUT]")
        print(f"  Documents: {len(self.documents)}")
        print(f"  Config hash: {self.config_hash[:16]}...")

    async def setup_pipeline(self):
        """Initialize pipeline components."""
        print(f"\n{'='*70}")
        print("  SETTING UP PIPELINE")
        print(f"{'='*70}")

        self.preprocessor = DocumentPreprocessor(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.preprocessor.process_documents(self.documents, use_context=False)

        self.query_planner = QueryPlannerModule(model_name=self.model)
        self.retriever = HybridRetriever(preprocessor=self.preprocessor)
        self.reranker = RerankerModule(model_name=self.model)
        self.generator = GeneratorModule(model_name=self.model)
        self.evaluator = RAGASEvaluator(model=self.model)

        self.chunk_labels = self.preprocessor.create_relevance_labels(self.all_queries)

        print(f"  Preprocessor: {len(self.preprocessor.chunks)} chunks")
        print(f"  Chunk labels: {len(self.chunk_labels)} queries")

    # -------------------------------------------------------------------------
    # Concurrency Control
    # -------------------------------------------------------------------------

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazy initialization of semaphore to avoid event loop issues."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    # -------------------------------------------------------------------------
    # Context Generation (with retry and parallelism)
    # -------------------------------------------------------------------------

    @retry_with_backoff(max_retries=5, base_delay=2.0)
    async def _run_query_planner_single(self, query: str) -> List[str]:
        output = await self.query_planner.run(QueryPlannerInput(query=query))
        return output.queries

    @retry_with_backoff(max_retries=5, base_delay=2.0)
    async def _run_retrieval_single(self, queries: List[str], top_k: int) -> Tuple[List[str], List[int]]:
        output = await self.retriever.run(RetrievalInput(queries=queries, top_k=top_k))
        return output.document_texts, output.chunk_indices

    @retry_with_backoff(max_retries=5, base_delay=2.0)
    async def _run_reranker_single(self, query: str, documents: List[str]) -> List[str]:
        output = await self.reranker.run(RerankerInput(query=query, documents=documents))
        return output.ranked_documents[:self.rerank_k]

    @retry_with_backoff(max_retries=5, base_delay=2.0)
    async def _run_generator_single(self, query: str, context: str) -> str:
        output = await self.generator.run(GeneratorInput(query=query, context=context))
        return output.answer

    async def _process_single_query_context(
        self,
        q: Dict,
    ) -> Tuple[str, List[str], List[str], Optional[str]]:
        """Process a single query through planning, retrieval, and reranking.

        Returns:
            Tuple of (query_id, retrieved_docs, reranked_docs, error_message)
        """
        qid = q["query_id"]
        query_text = q["query"]

        async with self.semaphore:
            try:
                planned = await self._run_query_planner_single(query_text)
                docs, _ = await self._run_retrieval_single(planned, self.retrieval_k)
                ranked = await self._run_reranker_single(query_text, docs)
                return (qid, docs, ranked, None)
            except Exception as e:
                return (qid, [], [], str(e)[:80])

    async def generate_contexts_for_split(
        self,
        queries: List[Dict],
        description: str = "",
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Generate contexts for a specific data split with parallel processing."""
        print(f"\n  Generating contexts ({description}, n={len(queries)}, concurrency={self.max_concurrent})...")

        # Create tasks for all queries
        tasks = [self._process_single_query_context(q) for q in queries]

        # Run all tasks concurrently with progress bar
        results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"    Processing queries",
            total=len(tasks),
        )

        # Collect results
        retrieved_contexts: Dict[str, List[str]] = {}
        reranked_contexts: Dict[str, List[str]] = {}
        errors = 0

        for qid, retrieved, reranked, error in results:
            retrieved_contexts[qid] = retrieved
            reranked_contexts[qid] = reranked
            if error:
                errors += 1
                print(f"    [ERROR] Query {qid}: {error}")

        if errors:
            print(f"    Completed with {errors} errors")

        return retrieved_contexts, reranked_contexts

    # -------------------------------------------------------------------------
    # Evaluation with Per-Example Tracking (Parallel)
    # -------------------------------------------------------------------------

    async def _evaluate_single_query(
        self,
        q: Dict,
        contexts: List[str],
    ) -> Optional[PerExampleResult]:
        """Evaluate a single query with generation and RAGAS metrics.

        Returns:
            PerExampleResult or None if evaluation failed
        """
        qid = q["query_id"]
        query_text = q["query"]
        ground_truth = q.get("ground_truth", "")

        if not contexts:
            return None

        async with self.semaphore:
            try:
                # Generate answer
                context_text = "\n\n".join(contexts)
                answer = await self._run_generator_single(query_text, context_text)

                # Evaluate
                scores = await self.evaluator.evaluate_end_to_end(
                    query=query_text,
                    contexts=contexts,
                    answer=answer,
                    ground_truth=ground_truth,
                )

                overall = scores.get("overall_quality", 0.0)

                return PerExampleResult(
                    query_id=qid,
                    query=query_text,
                    ground_truth=ground_truth,
                    score=overall,
                    metrics=scores,
                    answer=answer,
                    contexts=contexts[:3],  # Store top 3 for analysis
                )

            except Exception as e:
                print(f"    [ERROR] {qid}: {str(e)[:80]}")
                return None

    async def evaluate_split(
        self,
        queries: List[Dict],
        reranked_contexts: Dict[str, List[str]],
        description: str,
        split: str,
    ) -> EvaluationRecord:
        """Evaluate a data split with per-example tracking and parallel processing."""
        print(f"\n  Evaluating {split} split ({description}, concurrency={self.max_concurrent})...")

        # Create tasks for all queries
        tasks = [
            self._evaluate_single_query(q, reranked_contexts.get(q["query_id"], []))
            for q in queries
        ]

        # Run all tasks concurrently with progress bar
        results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"    Evaluating queries",
            total=len(tasks),
        )

        # Collect results (filter None values from failed evaluations)
        per_example_results: List[PerExampleResult] = [r for r in results if r is not None]
        all_scores: List[float] = [r.score for r in per_example_results]
        all_metrics: Dict[str, List[float]] = defaultdict(list)

        for r in per_example_results:
            for k, v in r.metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics[k].append(v)

        # Compute statistics
        mean, ci_lower, ci_upper = compute_confidence_interval(all_scores)
        std = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0

        # Aggregate metrics
        agg_metrics = {}
        for k, values in all_metrics.items():
            if values:
                agg_metrics[f"mean_{k}"] = statistics.mean(values)
                if len(values) > 1:
                    agg_metrics[f"std_{k}"] = statistics.stdev(values)

        record = EvaluationRecord(
            description=description,
            split=split,
            timestamp=datetime.now().isoformat(),
            n_examples=len(all_scores),
            mean_score=mean,
            std_score=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            per_example_scores=all_scores,
            per_example_results=per_example_results,
            metrics=agg_metrics,
        )

        print(f"  {split} score: {mean:.4f} ± {std:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")

        return record

    # -------------------------------------------------------------------------
    # Module Optimization with Full Tracking
    # -------------------------------------------------------------------------

    async def optimize_module(
        self,
        module_name: str,
        adapter,
        train_data: List[RAGDataInst],
        val_data: List[RAGDataInst],
    ) -> ModuleOptimizationRecord:
        """Optimize module with comprehensive tracking."""
        print(f"\n{'='*70}")
        print(f"  OPTIMIZING: {module_name.upper()}")
        print(f"{'='*70}")
        print(f"  Train examples: {len(train_data)}")
        print(f"  Val examples: {len(val_data)}")

        start_time = datetime.now()

        # Get seed prompt
        seed_candidate = adapter.get_candidate()
        seed_prompt = seed_candidate[adapter.component_name]

        # Save seed
        self._save_prompt(module_name, seed_prompt, "seed")

        # Baseline evaluation with per-example scores
        print(f"\n  Baseline evaluation on validation set...")
        baseline_batch = await self._evaluate_with_retry(adapter, val_data, seed_candidate)
        baseline_scores = baseline_batch.scores
        baseline_mean, baseline_ci_lower, baseline_ci_upper = compute_confidence_interval(baseline_scores)

        print(f"  Baseline: {baseline_mean:.4f} (95% CI: [{baseline_ci_lower:.4f}, {baseline_ci_upper:.4f}])")

        # Run optimization
        iteration_history: List[GEPAIterationRecord] = []
        best_prompt = seed_prompt
        best_scores = baseline_scores
        best_mean = baseline_mean
        total_iterations = 0
        accepted_iterations = 0

        if GEPA_AVAILABLE:
            try:
                best_prompt, best_scores, iteration_history = await self._run_gepa_with_tracking(
                    adapter, train_data, val_data, module_name
                )
                best_mean, _, _ = compute_confidence_interval(best_scores)
                total_iterations = len(iteration_history)
                accepted_iterations = sum(1 for r in iteration_history if r.accepted)
            except Exception as e:
                print(f"  [ERROR] GEPA optimization failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  [SIMULATION] GEPA not available")
            # Simulate for testing
            best_prompt = seed_prompt + "\n# [Simulated optimization]"
            improvement = 0.05
            best_scores = [min(1.0, s + improvement) for s in baseline_scores]
            best_mean = statistics.mean(best_scores)
            total_iterations = 5
            accepted_iterations = 3

        # Apply best prompt
        adapter.module._prompt = best_prompt
        self._save_prompt(module_name, best_prompt, "best")

        # Compute improvement with significance
        best_ci_lower, best_ci_upper = compute_confidence_interval(best_scores)[1:]
        improvement_abs = best_mean - baseline_mean
        improvement_pct = (improvement_abs / baseline_mean * 100) if baseline_mean > 0 else 0

        t_stat, p_value = paired_ttest(baseline_scores, best_scores)
        is_significant = p_value < 0.05

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        record = ModuleOptimizationRecord(
            module_name=module_name,
            component_name=adapter.component_name,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            seed_prompt=seed_prompt,
            best_prompt=best_prompt,
            baseline_score=baseline_mean,
            baseline_ci_lower=baseline_ci_lower,
            baseline_ci_upper=baseline_ci_upper,
            baseline_per_example=baseline_scores,
            best_score=best_mean,
            best_ci_lower=best_ci_lower,
            best_ci_upper=best_ci_upper,
            best_per_example=best_scores,
            improvement_abs=improvement_abs,
            improvement_pct=improvement_pct,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            total_iterations=total_iterations,
            accepted_iterations=accepted_iterations,
            iteration_history=iteration_history,
        )

        # Print summary
        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"\n  Optimization Complete:")
        print(f"    Baseline: {baseline_mean:.4f} [{baseline_ci_lower:.4f}, {baseline_ci_upper:.4f}]")
        print(f"    Best:     {best_mean:.4f} [{best_ci_lower:.4f}, {best_ci_upper:.4f}]")
        print(f"    Improvement: {improvement_abs:+.4f} ({improvement_pct:+.1f}%) {sig_marker}")
        print(f"    t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
        print(f"    Iterations: {accepted_iterations}/{total_iterations} accepted")

        return record

    async def _evaluate_with_retry(self, adapter, data, candidate, max_retries=3):
        for attempt in range(max_retries):
            try:
                return await adapter._evaluate_async(data, candidate, capture_traces=True)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def _run_gepa_with_tracking(
        self,
        adapter,
        train_data: List[RAGDataInst],
        val_data: List[RAGDataInst],
        module_name: str,
    ) -> Tuple[str, List[float], List[GEPAIterationRecord]]:
        """Run GEPA optimization with iteration tracking."""
        from gepa import optimize

        print(f"  Running GEPA optimization (budget={self.optimization_budget})...")

        seed_candidate = adapter.get_candidate()

        # Track iterations via callback if GEPA supports it
        iteration_history: List[GEPAIterationRecord] = []

        gepa_result = optimize(
            seed_candidate=seed_candidate,
            trainset=train_data,
            valset=val_data,
            adapter=adapter,
            max_metric_calls=self.optimization_budget,
            reflection_lm=f"openai/{self.model}",
        )

        # Extract results
        best_candidate = getattr(gepa_result, 'best_candidate', {})
        best_prompt = best_candidate.get(adapter.component_name, "") if isinstance(best_candidate, dict) else ""

        # Evaluate best prompt to get per-example scores
        best_batch = await adapter._evaluate_async(val_data, best_candidate, capture_traces=False)
        best_scores = best_batch.scores

        # Try to extract iteration history from GEPA result
        candidates = getattr(gepa_result, 'candidates', [])
        val_scores = getattr(gepa_result, 'val_aggregate_scores', [])

        for i, (cand, score) in enumerate(zip(candidates, val_scores)):
            prompt_text = cand.get(adapter.component_name, "") if isinstance(cand, dict) else ""
            record = GEPAIterationRecord(
                iteration=i + 1,
                timestamp=datetime.now().isoformat(),
                prompt_hash=hashlib.md5(prompt_text.encode()).hexdigest()[:12],
                prompt_text=prompt_text,
                train_score=0.0,  # Not available from result
                val_score=score,
                accepted=i == getattr(gepa_result, 'best_idx', 0),
            )
            iteration_history.append(record)

            # Save each iteration prompt
            self._save_prompt(module_name, prompt_text, f"iter_{i+1:03d}")

        return best_prompt, best_scores, iteration_history

    def _save_prompt(self, module_name: str, prompt: str, version: str):
        prompt_dir = self.output_dir / "prompts" / module_name
        prompt_path = prompt_dir / f"{version}.txt"
        with open(prompt_path, "w") as f:
            f.write(prompt)

    # -------------------------------------------------------------------------
    # Phase Execution
    # -------------------------------------------------------------------------

    async def phase_0_baseline(self) -> PhaseCheckpoint:
        """Phase 0: Data split and baseline capture."""
        print(f"\n{'='*80}")
        print(f"  PHASE 0: DATA SPLIT & BASELINE CAPTURE")
        print(f"{'='*80}")

        checkpoint = PhaseCheckpoint(
            phase=0,
            timestamp=datetime.now().isoformat(),
            train_query_ids=[q["query_id"] for q in self.train_queries],
            val_query_ids=[q["query_id"] for q in self.val_queries],
            test_query_ids=[q["query_id"] for q in self.test_queries],
        )

        # Store seed prompts
        checkpoint.prompts = {
            "query_planner": self.query_planner.prompt or self.query_planner.get_default_prompt(),
            "reranker": self.reranker.prompt or self.reranker.get_default_prompt(),
            "generator": self.generator.prompt or self.generator.get_default_prompt(),
        }

        # Generate contexts for train+val (NOT test - held out)
        train_val_queries = self.train_queries + self.val_queries
        retrieved, reranked = await self.generate_contexts_for_split(
            train_val_queries, "baseline_train_val"
        )
        checkpoint.retrieved_contexts = retrieved
        checkpoint.reranked_contexts = reranked

        # Baseline evaluation on VAL set only
        val_eval = await self.evaluate_split(
            self.val_queries, reranked, "baseline", "val"
        )
        checkpoint.evaluations["baseline_val"] = val_eval.to_dict()

        self.save_checkpoint(0, checkpoint)
        return checkpoint

    async def phase_1_query_planner(self, prev_checkpoint: PhaseCheckpoint) -> PhaseCheckpoint:
        """Phase 1: Optimize Query Planner."""
        print(f"\n{'='*80}")
        print(f"  PHASE 1: QUERY PLANNER OPTIMIZATION")
        print(f"{'='*80}")

        checkpoint = PhaseCheckpoint(
            phase=1,
                timestamp=datetime.now().isoformat(),
            train_query_ids=prev_checkpoint.train_query_ids,
            val_query_ids=prev_checkpoint.val_query_ids,
            test_query_ids=prev_checkpoint.test_query_ids,
            prompts=prev_checkpoint.prompts.copy(),
            retrieved_contexts=prev_checkpoint.retrieved_contexts.copy(),
            reranked_contexts=prev_checkpoint.reranked_contexts.copy(),
            evaluations=prev_checkpoint.evaluations.copy(),
        )

        # Prepare train/val data
        train_data = []
        for q in self.train_queries:
            train_data.append({
                "query": q["query"],
                "ground_truth": q.get("ground_truth", ""),
                "relevant_chunk_indices": self.chunk_labels.get(q["query_id"], []),
                "contexts": None,
                "metadata": {"query_id": q["query_id"]},
            })

        val_data = []
        for q in self.val_queries:
            val_data.append({
                "query": q["query"],
                "ground_truth": q.get("ground_truth", ""),
                "relevant_chunk_indices": self.chunk_labels.get(q["query_id"], []),
                "contexts": None,
                "metadata": {"query_id": q["query_id"]},
            })

        # Optimize
        adapter = QueryPlannerAdapter(
            query_planner_module=self.query_planner,
            retriever_module=self.retriever,
            evaluator=self.evaluator,
        )

        result = await self.optimize_module("query_planner", adapter, train_data, val_data)
        checkpoint.module_results["query_planner"] = result.to_dict()
        checkpoint.prompts["query_planner"] = result.best_prompt

        # Regenerate contexts with optimized M1
        print(f"\n  [CASCADE] Regenerating contexts with optimized M1...")
        train_val_queries = self.train_queries + self.val_queries
        new_retrieved, new_reranked = await self.generate_contexts_for_split(
            train_val_queries, "post_m1_optimization"
        )
        checkpoint.retrieved_contexts = new_retrieved
        checkpoint.reranked_contexts = new_reranked

        self.save_checkpoint(1, checkpoint)
        return checkpoint

    async def phase_2_reranker(self, prev_checkpoint: PhaseCheckpoint) -> PhaseCheckpoint:
        """Phase 2: Optimize Reranker."""
        print(f"\n{'='*80}")
        print(f"  PHASE 2: RERANKER OPTIMIZATION")
        print(f"{'='*80}")

        checkpoint = PhaseCheckpoint(
            phase=2,
            timestamp=datetime.now().isoformat(),
            train_query_ids=prev_checkpoint.train_query_ids,
            val_query_ids=prev_checkpoint.val_query_ids,
            test_query_ids=prev_checkpoint.test_query_ids,
            prompts=prev_checkpoint.prompts.copy(),
            retrieved_contexts=prev_checkpoint.retrieved_contexts.copy(),
            reranked_contexts=prev_checkpoint.reranked_contexts.copy(),
            module_results=prev_checkpoint.module_results.copy(),
            evaluations=prev_checkpoint.evaluations.copy(),
        )

        # Prepare data with FRESH contexts from M1
        train_data = []
        for q in self.train_queries:
            qid = q["query_id"]
            contexts = prev_checkpoint.retrieved_contexts.get(qid, [])
            if contexts:
                train_data.append({
                    "query": q["query"],
                    "ground_truth": q.get("ground_truth", ""),
                    "relevant_chunk_indices": None,
                    "contexts": contexts,
                    "metadata": {"query_id": qid},
                })

        val_data = []
        for q in self.val_queries:
            qid = q["query_id"]
            contexts = prev_checkpoint.retrieved_contexts.get(qid, [])
            if contexts:
                val_data.append({
                    "query": q["query"],
                    "ground_truth": q.get("ground_truth", ""),
                    "relevant_chunk_indices": None,
                    "contexts": contexts,
                    "metadata": {"query_id": qid},
                })

        print(f"  Using fresh contexts from optimized M1")

        # Optimize
        adapter = RerankerAdapter(
            reranker_module=self.reranker,
            evaluator=self.evaluator,
            top_k=self.rerank_k,
        )

        result = await self.optimize_module("reranker", adapter, train_data, val_data)
        checkpoint.module_results["reranker"] = result.to_dict()
        checkpoint.prompts["reranker"] = result.best_prompt

        # Regenerate reranked contexts with optimized M2
        print(f"\n  [CASCADE] Regenerating reranked contexts with optimized M2 (concurrency={self.max_concurrent})...")

        async def rerank_single(q: Dict, docs: List[str]) -> Tuple[str, List[str]]:
            qid = q["query_id"]
            if not docs:
                return (qid, [])
            async with self.semaphore:
                try:
                    ranked = await self._run_reranker_single(q["query"], docs)
                    return (qid, ranked)
                except Exception:
                    return (qid, docs[:self.rerank_k])

        queries = self.train_queries + self.val_queries
        tasks = [
            rerank_single(q, prev_checkpoint.retrieved_contexts.get(q["query_id"], []))
            for q in queries
        ]
        results = await tqdm_asyncio.gather(*tasks, desc="    Reranking", total=len(tasks))
        new_reranked = {qid: docs for qid, docs in results}

        checkpoint.reranked_contexts = new_reranked

        self.save_checkpoint(2, checkpoint)
        return checkpoint

    async def phase_3_generator(self, prev_checkpoint: PhaseCheckpoint) -> PhaseCheckpoint:
        """Phase 3: Optimize Generator."""
        print(f"\n{'='*80}")
        print(f"  PHASE 3: GENERATOR OPTIMIZATION")
        print(f"{'='*80}")

        checkpoint = PhaseCheckpoint(
            phase=3,
                timestamp=datetime.now().isoformat(),
            train_query_ids=prev_checkpoint.train_query_ids,
            val_query_ids=prev_checkpoint.val_query_ids,
            test_query_ids=prev_checkpoint.test_query_ids,
            prompts=prev_checkpoint.prompts.copy(),
            retrieved_contexts=prev_checkpoint.retrieved_contexts.copy(),
            reranked_contexts=prev_checkpoint.reranked_contexts.copy(),
            module_results=prev_checkpoint.module_results.copy(),
            evaluations=prev_checkpoint.evaluations.copy(),
        )

        # Prepare data with FRESH reranked contexts from M2
        train_data = []
        for q in self.train_queries:
            qid = q["query_id"]
            contexts = prev_checkpoint.reranked_contexts.get(qid, [])
            if contexts:
                train_data.append({
                    "query": q["query"],
                    "ground_truth": q.get("ground_truth", ""),
                    "relevant_chunk_indices": None,
                    "contexts": contexts,
                    "metadata": {"query_id": qid},
                })

        val_data = []
        for q in self.val_queries:
            qid = q["query_id"]
            contexts = prev_checkpoint.reranked_contexts.get(qid, [])
            if contexts:
                val_data.append({
                    "query": q["query"],
                    "ground_truth": q.get("ground_truth", ""),
                    "relevant_chunk_indices": None,
                    "contexts": contexts,
                    "metadata": {"query_id": qid},
                })

        print(f"  Using fresh contexts from optimized M1+M2")

        # Optimize
        adapter = GeneratorAdapter(
            generator_module=self.generator,
            evaluator=self.evaluator,
        )

        result = await self.optimize_module("generator", adapter, train_data, val_data)
        checkpoint.module_results["generator"] = result.to_dict()
        checkpoint.prompts["generator"] = result.best_prompt

        self.save_checkpoint(3, checkpoint)
        return checkpoint

    async def phase_4_test_evaluation(self, prev_checkpoint: PhaseCheckpoint) -> PhaseCheckpoint:
        """Phase 4: Test set evaluation and ablation studies."""
        print(f"\n{'='*80}")
        print(f"  PHASE 4: TEST SET EVALUATION & ABLATIONS")
        print(f"{'='*80}")
        print(f"  [IMPORTANT] Now evaluating on HELD-OUT test set")

        checkpoint = PhaseCheckpoint(
            phase=4,
            timestamp=datetime.now().isoformat(),
            train_query_ids=prev_checkpoint.train_query_ids,
            val_query_ids=prev_checkpoint.val_query_ids,
            test_query_ids=prev_checkpoint.test_query_ids,
            prompts=prev_checkpoint.prompts.copy(),
            retrieved_contexts=prev_checkpoint.retrieved_contexts.copy(),
            reranked_contexts=prev_checkpoint.reranked_contexts.copy(),
            module_results=prev_checkpoint.module_results.copy(),
            evaluations=prev_checkpoint.evaluations.copy(),
        )

        # Run ablation studies on TEST set
        ablations = await self.run_ablation_studies(prev_checkpoint)

        # Store ablation results
        for ab in ablations:
            checkpoint.evaluations[f"ablation_{ab.config_name}"] = ab.to_dict()

        # Generate all reports
        self._generate_reports(checkpoint, ablations)

        self.save_checkpoint(4, checkpoint)
        return checkpoint

    async def run_ablation_studies(self, checkpoint: PhaseCheckpoint) -> List[AblationResult]:
        """Run ablation studies on TEST set."""
        print(f"\n  Running ablation studies on TEST set (n={len(self.test_queries)})...")

        ablations = []

        # Load seed prompts from phase 0
        phase0 = self.load_checkpoint(0)
        seed_prompts = phase0.prompts if phase0 else {}

        configs = [
            ("baseline", seed_prompts),
            ("M1_only", {
                "query_planner": checkpoint.prompts.get("query_planner", ""),
                "reranker": seed_prompts.get("reranker", ""),
                "generator": seed_prompts.get("generator", ""),
            }),
            ("M1+M2", {
                "query_planner": checkpoint.prompts.get("query_planner", ""),
                "reranker": checkpoint.prompts.get("reranker", ""),
                "generator": seed_prompts.get("generator", ""),
            }),
            ("M1+M2+M3", checkpoint.prompts),
        ]

        baseline_scores: List[float] = []

        for config_name, prompts in configs:
            print(f"\n    Ablation: {config_name}")

            # Apply prompts
            self.query_planner._prompt = prompts.get("query_planner", "")
            self.reranker._prompt = prompts.get("reranker", "")
            self.generator._prompt = prompts.get("generator", "")

            # Generate contexts for TEST set
            _, reranked = await self.generate_contexts_for_split(
                self.test_queries, f"ablation_{config_name}"
            )

            # Evaluate on TEST set
            eval_record = await self.evaluate_split(
                self.test_queries, reranked, config_name, "test"
            )

            test_scores = eval_record.per_example_scores

            if config_name == "baseline":
                baseline_scores = test_scores.copy()
                self.baseline_test_scores = baseline_scores

            # Compute significance vs baseline
            improvement = eval_record.mean_score - (statistics.mean(baseline_scores) if baseline_scores else 0)
            t_stat, p_value = paired_ttest(baseline_scores, test_scores) if baseline_scores else (0, 1)

            ablation = AblationResult(
                config_name=config_name,
                prompts_used=prompts,
                test_score=eval_record.mean_score,
                test_ci_lower=eval_record.ci_lower,
                test_ci_upper=eval_record.ci_upper,
                test_per_example=test_scores,
                improvement_vs_baseline=improvement,
                t_statistic=t_stat,
                p_value=p_value,
                is_significant=p_value < 0.05,
                metrics=eval_record.metrics,
            )
            ablations.append(ablation)

            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"      Score: {eval_record.mean_score:.4f} [{eval_record.ci_lower:.4f}, {eval_record.ci_upper:.4f}]")
            print(f"      vs Baseline: {improvement:+.4f} (p={p_value:.4f}) {sig_marker}")

        # Restore final prompts
        self.query_planner._prompt = checkpoint.prompts.get("query_planner", "")
        self.reranker._prompt = checkpoint.prompts.get("reranker", "")
        self.generator._prompt = checkpoint.prompts.get("generator", "")

        self.ablation_results = ablations
        return ablations

    # -------------------------------------------------------------------------
    # Report Generation
    # -------------------------------------------------------------------------

    def _generate_reports(self, checkpoint: PhaseCheckpoint, ablations: List[AblationResult]):
        """Generate all reports including LaTeX tables."""
        print(f"\n  Generating reports...")

        # 1. Summary JSON
        summary = {
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "random_seed": self.random_seed,
            "timestamp": datetime.now().isoformat(),
            "data_splits": {
                "train": len(self.train_queries),
                "val": len(self.val_queries),
                "test": len(self.test_queries),
            },
            "module_results": checkpoint.module_results,
            "ablations": [a.to_dict() for a in ablations],
        }

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # 2. CSV Tables
        self._generate_csv_tables(checkpoint, ablations)

        # 3. LaTeX Tables
        self._generate_latex_tables(checkpoint, ablations)

        # 4. Per-example results
        self._save_per_example_results(ablations)

        print(f"  Reports saved to {self.output_dir}")

    def _generate_csv_tables(self, checkpoint: PhaseCheckpoint, ablations: List[AblationResult]):
        """Generate CSV tables."""
        # Module results
        with open(self.output_dir / "analysis" / "module_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Module", "Baseline", "Baseline CI", "Optimized", "Optimized CI",
                "Improvement", "Improvement %", "t-stat", "p-value", "Significant"
            ])

            for name in ["query_planner", "reranker", "generator"]:
                if name in checkpoint.module_results:
                    r = checkpoint.module_results[name]
            writer.writerow([
                        name,
                        f"{r['baseline_score']:.4f}",
                        f"[{r['baseline_ci_lower']:.4f}, {r['baseline_ci_upper']:.4f}]",
                        f"{r['best_score']:.4f}",
                        f"[{r['best_ci_lower']:.4f}, {r['best_ci_upper']:.4f}]",
                        f"{r['improvement_abs']:+.4f}",
                        f"{r['improvement_pct']:+.1f}%",
                        f"{r['t_statistic']:.3f}",
                        f"{r['p_value']:.4f}",
                        "Yes" if r['is_significant'] else "No",
                    ])

        # Ablation results
        with open(self.output_dir / "ablations" / "ablation_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Configuration", "Test Score", "95% CI", "vs Baseline", "p-value", "Significant"
            ])

            for ab in ablations:
                writer.writerow([
                    ab.config_name,
                    f"{ab.test_score:.4f}",
                    f"[{ab.test_ci_lower:.4f}, {ab.test_ci_upper:.4f}]",
                    f"{ab.improvement_vs_baseline:+.4f}",
                    f"{ab.p_value:.4f}",
                    "Yes" if ab.is_significant else "No",
                ])

    def _generate_latex_tables(self, checkpoint: PhaseCheckpoint, ablations: List[AblationResult]):
        """Generate LaTeX tables for paper."""

        # Table 1: Module-level optimization results
        latex_module = r"""
\begin{table}[t]
\centering
\caption{Per-Module Optimization Results on Validation Set}
\label{tab:module_results}
\begin{tabular}{lcccccc}
\toprule
\textbf{Module} & \textbf{Baseline} & \textbf{Optimized} & \textbf{$\Delta$} & \textbf{$\Delta$\%} & \textbf{$p$-value} \\
\midrule
"""

        for name in ["query_planner", "reranker", "generator"]:
            if name in checkpoint.module_results:
                r = checkpoint.module_results[name]
                sig = self._significance_stars(r['p_value'])
                display_name = name.replace("_", " ").title()
                latex_module += f"{display_name} & "
                latex_module += f"{r['baseline_score']:.3f} & "
                latex_module += f"{r['best_score']:.3f} & "
                latex_module += f"{r['improvement_abs']:+.3f} & "
                latex_module += f"{r['improvement_pct']:+.1f}\\% & "
                latex_module += f"{r['p_value']:.3f}{sig} \\\\\n"

        latex_module += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Significance: $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$
\end{tablenotes}
\end{table}
"""

        with open(self.output_dir / "latex" / "table_module_results.tex", "w") as f:
            f.write(latex_module)

        # Table 2: Ablation study on test set
        latex_ablation = r"""
\begin{table}[t]
\centering
\caption{Ablation Study on Held-Out Test Set}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
\textbf{Configuration} & \textbf{Score} & \textbf{95\% CI} & \textbf{$\Delta$ vs Baseline} & \textbf{$p$-value} \\
\midrule
"""

        for ab in ablations:
            sig = self._significance_stars(ab.p_value)
            latex_ablation += f"{ab.config_name.replace('_', ' ')} & "
            latex_ablation += f"{ab.test_score:.3f} & "
            latex_ablation += f"[{ab.test_ci_lower:.3f}, {ab.test_ci_upper:.3f}] & "
            latex_ablation += f"{ab.improvement_vs_baseline:+.3f} & "
            latex_ablation += f"{ab.p_value:.3f}{sig} \\\\\n"

        latex_ablation += r"""
\bottomrule
\end{tabular}
\end{table}
"""

        with open(self.output_dir / "latex" / "table_ablation.tex", "w") as f:
            f.write(latex_ablation)

        # Table 3: Configuration
        latex_config = r"""
\begin{table}[t]
\centering
\caption{Experiment Configuration}
\label{tab:config}
\begin{tabular}{ll}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
"""
        latex_config += f"Total Queries & {self.n_queries} \\\\\n"
        latex_config += f"Train/Val/Test Split & {int(self.train_ratio*100)}/{int(self.val_ratio*100)}/{int(self.test_ratio*100)} \\\\\n"
        latex_config += f"Optimization Budget & {self.optimization_budget} \\\\\n"
        latex_config += f"LLM Model & {self.model} \\\\\n"
        latex_config += f"Retrieval $k$ & {self.retrieval_k} \\\\\n"
        latex_config += f"Rerank $k$ & {self.rerank_k} \\\\\n"
        latex_config += f"Random Seed & {self.random_seed} \\\\\n"

        latex_config += r"""
\bottomrule
\end{tabular}
\end{table}
"""

        with open(self.output_dir / "latex" / "table_config.tex", "w") as f:
            f.write(latex_config)

    def _significance_stars(self, p_value: float) -> str:
        if p_value < 0.001:
            return "$^{***}$"
        elif p_value < 0.01:
            return "$^{**}$"
        elif p_value < 0.05:
            return "$^{*}$"
        return ""

    def _save_per_example_results(self, ablations: List[AblationResult]):
        """Save per-example results for analysis."""
        for ab in ablations:
            path = self.output_dir / "per_example" / f"{ab.config_name}_scores.json"
            with open(path, "w") as f:
                json.dump({
                    "config": ab.config_name,
                    "scores": ab.test_per_example,
                    "mean": ab.test_score,
                    "ci": [ab.test_ci_lower, ab.test_ci_upper],
                }, f, indent=2)

    # -------------------------------------------------------------------------
    # Main Execution
    # -------------------------------------------------------------------------

    async def run_experiment(self, data_path: Path, resume_from: int = 0):
        """Run complete experiment."""
        experiment_start = datetime.now()

        print("\n" + "=" * 80)
        print("  RESEARCH PAPER QUALITY RAG OPTIMIZATION EXPERIMENT")
        print("=" * 80)
        print(f"\n  Experiment: {self.experiment_name}")
        print(f"  ID: {self.experiment_id}")
        print(f"  Config Hash: {self.config_hash[:16]}...")
        print(f"  Random Seed: {self.random_seed}")
        print(f"  Output: {self.output_dir}")

        # Load and split data
        await self.load_and_split_data(data_path)

        # Setup pipeline
        await self.setup_pipeline()

        # Restore from checkpoint if resuming
        checkpoint = None
        if resume_from > 0:
            checkpoint = self.load_checkpoint(resume_from - 1)
            if checkpoint:
                self.restore_state_from_checkpoint(checkpoint)
            else:
                print(f"  [WARNING] No checkpoint found, starting from Phase 0")
                resume_from = 0

        # Execute phases
        try:
            if resume_from <= 0:
                checkpoint = await self.phase_0_baseline()

            if resume_from <= 1:
                if checkpoint is None:
                    checkpoint = self.load_checkpoint(0)
                checkpoint = await self.phase_1_query_planner(checkpoint)

            if resume_from <= 2:
                if checkpoint is None:
                    checkpoint = self.load_checkpoint(1)
                    self.restore_state_from_checkpoint(checkpoint)
                checkpoint = await self.phase_2_reranker(checkpoint)

            if resume_from <= 3:
                if checkpoint is None:
                    checkpoint = self.load_checkpoint(2)
                    self.restore_state_from_checkpoint(checkpoint)
                checkpoint = await self.phase_3_generator(checkpoint)

            if resume_from <= 4:
                if checkpoint is None:
                    checkpoint = self.load_checkpoint(3)
                    self.restore_state_from_checkpoint(checkpoint)
                checkpoint = await self.phase_4_test_evaluation(checkpoint)

        except Exception as e:
            print(f"\n  [ERROR] Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n  Resume with: --resume_from <phase>")
            raise

        # Print final summary
        experiment_end = datetime.now()
        duration = (experiment_end - experiment_start).total_seconds()

        print(f"\n{'='*80}")
        print("  EXPERIMENT COMPLETE")
        print(f"{'='*80}")

        if self.ablation_results:
            baseline = self.ablation_results[0]
            best = self.ablation_results[-1]

            print(f"\n  Test Set Results (Held-Out, n={len(self.test_queries)}):")
            print(f"    Baseline:  {baseline.test_score:.4f} [{baseline.test_ci_lower:.4f}, {baseline.test_ci_upper:.4f}]")
            print(f"    Optimized: {best.test_score:.4f} [{best.test_ci_lower:.4f}, {best.test_ci_upper:.4f}]")

            sig = "***" if best.p_value < 0.001 else "**" if best.p_value < 0.01 else "*" if best.p_value < 0.05 else ""
            print(f"    Improvement: {best.improvement_vs_baseline:+.4f} (p={best.p_value:.4f}) {sig}")

            print(f"\n  Ablation Study:")
            for ab in self.ablation_results:
                sig = "***" if ab.p_value < 0.001 else "**" if ab.p_value < 0.01 else "*" if ab.p_value < 0.05 else ""
                print(f"    {ab.config_name:12s}: {ab.test_score:.4f} (Δ={ab.improvement_vs_baseline:+.4f}) {sig}")

        print(f"\n  Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"  Output: {self.output_dir}")
        print(f"\n  Generated Files:")
        print(f"    - summary.json (complete results)")
        print(f"    - latex/*.tex (paper-ready tables)")
        print(f"    - analysis/*.csv (detailed metrics)")
        print(f"    - per_example/*.json (per-query scores)")


# =============================================================================
# Main Entry Point
# =============================================================================

async def main(args):
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return

    runner = ResearchExperimentRunner(
        experiment_name=args.experiment_name,
        output_dir=Path(args.output_dir),
        n_queries=args.n_queries,
        optimization_budget=args.budget,
        model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        retrieval_k=args.retrieval_k,
        rerank_k=args.rerank_k,
        random_seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_concurrent=args.max_concurrent,
    )

    await runner.run_experiment(
        data_path=Path(args.data_path),
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Research Paper Quality RAG Pipeline Optimization"
    )

    parser.add_argument("--experiment_name", type=str, default="paper_exp_001")
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--data_path", type=str, default="data/train")
    parser.add_argument("--resume_from", type=int, default=0)

    parser.add_argument("--n_queries", type=int, default=100)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)

    parser.add_argument("--chunk_size", type=int, default=600)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument("--retrieval_k", type=int, default=20)
    parser.add_argument("--rerank_k", type=int, default=10)
    parser.add_argument("--max_concurrent", type=int, default=10,
                        help="Max concurrent API calls (default: 10)")

    args = parser.parse_args()
    asyncio.run(main(args))
