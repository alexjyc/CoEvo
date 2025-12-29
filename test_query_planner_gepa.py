"""
Query Planner GEPA Optimization Test

Tests the query_planner module optimization using gepa.optimize()
following the crag_benchmark implementation pattern from gepa-ai/gepa.

Usage:
    python test_query_planner_gepa.py --n_queries 10

Steps:
1. Load first N queries from train.json (document-level)
2. Setup preprocessor to chunk documents and create chunk-level relevance
3. Initialize query planner + retriever modules
4. Create training data in RAGDataInst format
5. Call gepa.optimize() with QueryPlannerAdapter
6. Report optimization results
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from modules.preprocessor import DocumentPreprocessor
from modules.query_planner.planner import QueryPlannerModule
from modules.query_planner.retrieval import HybridRetriever
from modules.evaluation import RAGASEvaluator

# Import GEPA adapter
from gepa_adapters.query_planner_adapter import QueryPlannerAdapter
from gepa_adapters.base import RAGDataInst, optimize_prompt, GEPA_AVAILABLE


def load_train_data(data_path: Path, n_queries: int = 10) -> tuple:
    """
    Load first N queries and their associated documents from train.json.

    Returns:
        Tuple of (documents, queries, relevance_labels_doc)
    """
    print(f"\n{'='*60}")
    print("Step 1: Loading Training Data")
    print(f"{'='*60}")

    # Load documents
    docs_path = data_path / "documents.json"
    if docs_path.exists():
        with open(docs_path) as f:
            all_documents = json.load(f)
        print(f"  Loaded {len(all_documents)} total documents")
    else:
        raise FileNotFoundError(f"documents.json not found at {docs_path}")

    # Load queries
    queries_path = data_path / "queries.json"
    if queries_path.exists():
        with open(queries_path) as f:
            all_queries = json.load(f)
        print(f"  Loaded {len(all_queries)} total queries")
    else:
        raise FileNotFoundError(f"queries.json not found at {queries_path}")

    # Load relevance labels
    labels_path = data_path / "relevance_labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            all_labels = {int(k): v for k, v in json.load(f).items()}
        print(f"  Loaded relevance labels for {len(all_labels)} queries")
    else:
        all_labels = {}
        print("  No relevance labels found")

    # Filter to first N queries
    queries = all_queries[:n_queries]
    query_ids = {q["query_id"] for q in queries}

    # Get relevant document IDs for these queries
    relevant_doc_ids = set()
    for q in queries:
        relevant_doc_ids.update(q.get("reference_doc_ids", []))

    # Filter documents to only those relevant to our queries
    documents = [d for d in all_documents if d["doc_id"] in relevant_doc_ids]

    # Filter labels
    labels = {k: v for k, v in all_labels.items() if k in query_ids}

    print(f"\n  Selected {n_queries} queries with {len(documents)} relevant documents")
    print(f"  Sample query: '{queries[0]['query'][:80]}...'")

    return documents, queries, labels


def setup_pipeline(
    documents: List[Dict],
    relevance_labels_doc: Dict[int, List[str]],
    queries: List[Dict],
    chunk_size: int = 600,
    chunk_overlap: int = 50,
) -> tuple:
    """
    Setup preprocessor and modules.

    Returns:
        Tuple of (preprocessor, query_planner, retriever, chunk_level_labels)
    """
    print(f"\n{'='*60}")
    print("Step 2: Setting Up Pipeline")
    print(f"{'='*60}")

    # Initialize preprocessor
    preprocessor = DocumentPreprocessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Process documents (creates chunks and embeddings)
    print("  Processing documents (chunking + embedding)...")
    preprocessor.process_documents(documents, use_context=False)
    print(f"  Created {len(preprocessor.chunks)} chunks")

    # Convert document-level labels to chunk-level
    if relevance_labels_doc:
        print("  Converting document-level to chunk-level relevance...")
        chunk_level_labels = preprocessor.create_relevance_labels(queries)
    else:
        chunk_level_labels = {}

    # Initialize query planner
    query_planner = QueryPlannerModule(model_name="gpt-4o-mini")
    print(f"  Query planner initialized with seed prompt ({len(query_planner.prompt)} chars)")

    # Initialize retriever
    retriever = HybridRetriever(
        preprocessor=preprocessor,
        rrf_k=60,
        default_dense_weight=0.5,
    )
    print(f"  Hybrid retriever initialized (dense_weight=0.5)")

    return preprocessor, query_planner, retriever, chunk_level_labels


def create_training_data(
    queries: List[Dict],
    chunk_level_labels: Dict[int, List[int]],
) -> List[RAGDataInst]:
    """
    Create training data in RAGDataInst format for GEPA.

    Args:
        queries: List of query dicts from train.json
        chunk_level_labels: {query_id: [chunk_indices]}

    Returns:
        List of RAGDataInst dicts
    """
    print(f"\n{'='*60}")
    print("Step 3: Creating Training Data (RAGDataInst format)")
    print(f"{'='*60}")

    training_data: List[RAGDataInst] = []

    for q in queries:
        query_id = q["query_id"]

        # Get chunk-level relevant indices
        relevant_indices = chunk_level_labels.get(query_id, [])

        data_inst: RAGDataInst = {
            "query": q["query"],
            "ground_truth": q.get("ground_truth", ""),
            "relevant_chunk_indices": relevant_indices,
            "contexts": None,  # Query planner doesn't need pre-retrieved contexts
            "metadata": {
                "query_id": query_id,
                "reference_doc_ids": q.get("reference_doc_ids", []),
                "domain": q.get("metadata", {}).get("domain", "unknown"),
                "split": "train",
            }
        }
        training_data.append(data_inst)

        print(f"  Query {query_id}: '{q['query'][:50]}...'")
        print(f"    → {len(relevant_indices)} relevant chunks")

    print(f"\n  Total training instances: {len(training_data)}")

    return training_data


async def run_baseline_evaluation(
    adapter: QueryPlannerAdapter,
    training_data: List[RAGDataInst],
) -> Dict[str, Any]:
    """
    Run baseline evaluation with seed prompt.

    Returns:
        Dict with baseline scores and metrics
    """
    print(f"\n{'='*60}")
    print("Step 4: Baseline Evaluation")
    print(f"{'='*60}")

    seed_candidate = adapter.get_candidate()
    print(f"  Evaluating with seed prompt...")
    print(f"  Prompt preview: '{seed_candidate[adapter.component_name][:100]}...'")

    # Run evaluation (use async version directly to avoid event loop issues)
    eval_batch = await adapter._evaluate_async(
        batch=training_data,
        candidate=seed_candidate,
        capture_traces=True,
    )

    baseline_score = eval_batch.aggregate_score
    success_rate = eval_batch.success_rate

    print(f"\n  Baseline Results:")
    print(f"    Aggregate Score (F1): {baseline_score:.4f}")
    print(f"    Success Rate: {success_rate:.1%}")

    # Show per-query scores
    print(f"\n  Per-Query Scores:")
    for i, (score, trajectory) in enumerate(zip(eval_batch.scores, eval_batch.trajectories or [])):
        if trajectory:
            metrics = trajectory.get("metrics", {})
            print(f"    Query {i}: F1={score:.3f}, Precision={metrics.get('context_precision', 0):.3f}, Recall={metrics.get('context_recall', 0):.3f}")

    return {
        "baseline_score": baseline_score,
        "scores": eval_batch.scores,
        "success_rate": success_rate,
        "seed_prompt": seed_candidate[adapter.component_name],
    }


def run_gepa_optimization(
    adapter: QueryPlannerAdapter,
    training_data: List[RAGDataInst],
    max_metric_calls: int = 20,
    reflection_lm: str = "openai/gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Run GEPA optimization following gepa-ai/gepa crag_benchmark pattern.

    Args:
        adapter: QueryPlannerAdapter instance
        training_data: List of RAGDataInst
        max_metric_calls: Optimization budget
        reflection_lm: Model for reflection

    Returns:
        Optimization results
    """
    print(f"\n{'='*60}")
    print("Step 5: GEPA Optimization")
    print(f"{'='*60}")

    if not GEPA_AVAILABLE:
        print("\n  ⚠️  GEPA not installed. Install with: pip install gepa-ai")
        print("  Simulating optimization steps...\n")

        # Simulate optimization for demonstration
        return simulate_optimization(adapter, training_data, max_metric_calls)

    # Split data into train/val
    split_idx = max(1, len(training_data) * 8 // 10)
    trainset = training_data[:split_idx]
    valset = training_data[split_idx:] if split_idx < len(training_data) else training_data[:2]

    print(f"  Training set: {len(trainset)} examples")
    print(f"  Validation set: {len(valset)} examples")
    print(f"  Budget: {max_metric_calls} metric calls")
    print(f"  Reflection LM: {reflection_lm}")

    # Use the optimize_prompt convenience function
    result = optimize_prompt(
        adapter=adapter,
        trainset=trainset,
        valset=valset,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
    )

    return result


def simulate_optimization(
    adapter: QueryPlannerAdapter,
    training_data: List[RAGDataInst],
    max_metric_calls: int,
) -> Dict[str, Any]:
    """
    Simulate GEPA optimization for demonstration when GEPA is not installed.
    Shows what would happen at each step.
    """
    print("  SIMULATION MODE (GEPA not installed)")
    print("  " + "-"*50)

    seed_candidate = adapter.get_candidate()
    seed_prompt = seed_candidate[adapter.component_name]

    print(f"\n  Initial Setup:")
    print(f"    Component: {adapter.component_name}")
    print(f"    Seed prompt: {len(seed_prompt)} chars")
    print(f"    Training examples: {len(training_data)}")

    print(f"\n  GEPA Optimization Loop (would run for {max_metric_calls} calls):")
    print(f"    1. Evaluate seed candidate on minibatch")
    print(f"    2. Capture execution traces (trajectories)")
    print(f"    3. Build reflective dataset from traces")
    print(f"    4. LLM proposes improved prompt based on reflection")
    print(f"    5. Evaluate new candidate")
    print(f"    6. Accept if score improves (Pareto update)")
    print(f"    7. Repeat until budget exhausted")

    print(f"\n  Key GEPA Methods:")
    print(f"    - adapter.evaluate(batch, candidate, capture_traces=True)")
    print(f"    - adapter.make_reflective_dataset(candidate, eval_batch, components)")
    print(f"    - reflection_lm proposes new text based on feedback")

    print(f"\n  Expected Output:")
    print(f"    - best_prompt: Optimized query planning prompt")
    print(f"    - best_score: Improved F1 score")
    print(f"    - pareto_frontier: All non-dominated candidates")

    return {
        "best_prompt": seed_prompt + "\n[OPTIMIZED - placeholder]",
        "best_score": 0.0,  # Would be computed
        "component_name": adapter.component_name,
        "simulated": True,
    }


async def main(args):
    """Main test workflow"""
    load_dotenv()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return

    print("\n" + "="*70)
    print("  QUERY PLANNER GEPA OPTIMIZATION TEST")
    print("="*70)
    print(f"\n  Configuration:")
    print(f"    Data path: {args.data_path}")
    print(f"    N queries: {args.n_queries}")
    print(f"    Budget: {args.budget} metric calls")
    print(f"    Reflection LM: {args.reflection_lm}")

    # Step 1: Load data
    data_path = Path(args.data_path)
    documents, queries, relevance_labels_doc = load_train_data(
        data_path, n_queries=args.n_queries
    )

    # Step 2: Setup pipeline
    preprocessor, query_planner, retriever, chunk_level_labels = setup_pipeline(
        documents=documents,
        relevance_labels_doc=relevance_labels_doc,
        queries=queries,
    )

    # Step 3: Create training data
    training_data = create_training_data(queries, chunk_level_labels)

    # Step 4: Create adapter
    print(f"\n{'='*60}")
    print("Setting Up GEPA Adapter")
    print(f"{'='*60}")

    evaluator = RAGASEvaluator(model="gpt-4o-mini")

    adapter = QueryPlannerAdapter(
        query_planner_module=query_planner,
        retriever_module=retriever,
        evaluator=evaluator,
    )
    print(f"  Adapter created: {adapter.component_name}")

    # Step 5: Run baseline evaluation
    baseline_results = await run_baseline_evaluation(adapter, training_data)

    # Step 6: Run GEPA optimization
    optimization_results = run_gepa_optimization(
        adapter=adapter,
        training_data=training_data,
        max_metric_calls=args.budget,
        reflection_lm=args.reflection_lm,
    )

    # Summary
    print(f"\n{'='*70}")
    print("  OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Baseline Score: {baseline_results['baseline_score']:.4f}")
    if not optimization_results.get("simulated"):
        print(f"  Optimized Score: {optimization_results['best_score']:.4f}")
        improvement = optimization_results['best_score'] - baseline_results['baseline_score']
        print(f"  Improvement: {improvement:+.4f}")
    else:
        print(f"  [SIMULATION] Install gepa-ai for actual optimization")

    print(f"\n  Output: {optimization_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Query Planner GEPA Optimization"
    )
    parser.add_argument(
        "--data_path", type=str, default="data/train",
        help="Path to training data directory"
    )
    parser.add_argument(
        "--n_queries", type=int, default=10,
        help="Number of queries to use for testing"
    )
    parser.add_argument(
        "--budget", type=int, default=20,
        help="GEPA optimization budget (metric calls)"
    )
    parser.add_argument(
        "--reflection_lm", type=str, default="openai/gpt-4o-mini",
        help="Model for GEPA reflection"
    )

    args = parser.parse_args()
    asyncio.run(main(args))
