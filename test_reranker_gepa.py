"""
Reranker GEPA Optimization Test

Tests the reranker module optimization using gepa.optimize()
following the crag_benchmark implementation pattern from gepa-ai/gepa.

Usage:
    python test_reranker_gepa.py --n_queries 10

Steps:
1. Load first N queries from train.json
2. Run query planner + retrieval to get initial documents
3. Create training data with pre-retrieved documents
4. Initialize reranker module and adapter
5. Call gepa.optimize() with RerankerAdapter
6. Report optimization results

Note: Reranker depends on Module 1 output (retrieved documents)
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
from modules.reranker.reranker import RerankerModule
from modules.evaluation import RAGASEvaluator
from modules.base import QueryPlannerInput, RetrievalInput

# Import GEPA adapter
from gepa_adapters.reranker_adapter import RerankerAdapter
from gepa_adapters.base import RAGDataInst, optimize_prompt, GEPA_AVAILABLE


def load_train_data(data_path: Path, n_queries: int = 10) -> tuple:
    """Load first N queries and their associated documents from train.json."""
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

    # Filter to first N queries
    queries = all_queries[:n_queries]
    query_ids = {q["query_id"] for q in queries}

    # Get relevant document IDs for these queries
    relevant_doc_ids = set()
    for q in queries:
        relevant_doc_ids.update(q.get("reference_doc_ids", []))

    # Filter documents to only those relevant to our queries
    documents = [d for d in all_documents if d["doc_id"] in relevant_doc_ids]

    print(f"\n  Selected {n_queries} queries with {len(documents)} relevant documents")

    return documents, queries


async def setup_module1_pipeline(
    documents: List[Dict],
    queries: List[Dict],
    chunk_size: int = 600,
    chunk_overlap: int = 50,
) -> tuple:
    """
    Setup Module 1 (Query Planner + Retrieval) and run to get documents.

    Returns:
        Tuple of (preprocessor, query_planner, retriever, retrieved_contexts)
    """
    print(f"\n{'='*60}")
    print("Step 2: Setting Up Module 1 (Query Planner + Retrieval)")
    print(f"{'='*60}")

    # Initialize preprocessor
    preprocessor = DocumentPreprocessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Process documents
    print("  Processing documents (chunking + embedding)...")
    preprocessor.process_documents(documents, use_context=False)
    print(f"  Created {len(preprocessor.chunks)} chunks")

    # Initialize query planner and retriever
    query_planner = QueryPlannerModule(model_name="gpt-4o-mini")
    retriever = HybridRetriever(
        preprocessor=preprocessor,
        rrf_k=60,
        default_dense_weight=0.5,
    )

    # Run Module 1 for each query to get retrieved documents
    print("\n  Running Module 1 to retrieve documents for each query...")
    retrieved_contexts: Dict[int, List[str]] = {}

    for q in queries:
        query_id = q["query_id"]
        query_text = q["query"]

        # Run query planner
        planner_input = QueryPlannerInput(query=query_text)
        planner_output = await query_planner.run(planner_input)

        # Run retriever
        retrieval_input = RetrievalInput(
            queries=planner_output.queries,
            top_k=20
        )
        retrieval_output = await retriever.run(retrieval_input)

        retrieved_contexts[query_id] = retrieval_output.document_texts
        print(f"    Query {query_id}: Retrieved {len(retrieval_output.document_texts)} documents")

    return preprocessor, query_planner, retriever, retrieved_contexts


def create_reranker_training_data(
    queries: List[Dict],
    retrieved_contexts: Dict[int, List[str]],
) -> List[RAGDataInst]:
    """
    Create training data for reranker in RAGDataInst format.

    Reranker needs:
    - query: The user query
    - contexts: Pre-retrieved documents (from Module 1)
    - ground_truth: Expected answer (for LLM-based evaluation)
    """
    print(f"\n{'='*60}")
    print("Step 3: Creating Reranker Training Data")
    print(f"{'='*60}")

    training_data: List[RAGDataInst] = []

    for q in queries:
        query_id = q["query_id"]
        contexts = retrieved_contexts.get(query_id, [])

        if not contexts:
            print(f"  Skipping query {query_id}: No retrieved documents")
            continue

        data_inst: RAGDataInst = {
            "query": q["query"],
            "ground_truth": q.get("ground_truth", ""),
            "relevant_chunk_indices": None,  # Reranker uses LLM-based eval
            "contexts": contexts,  # Pre-retrieved documents from Module 1
            "metadata": {
                "query_id": query_id,
                "num_contexts": len(contexts),
                "split": "train",
            }
        }
        training_data.append(data_inst)

        print(f"  Query {query_id}: '{q['query'][:50]}...'")
        print(f"    → {len(contexts)} documents to rerank")

    print(f"\n  Total training instances: {len(training_data)}")

    return training_data


async def run_baseline_evaluation(
    adapter: RerankerAdapter,
    training_data: List[RAGDataInst],
) -> Dict[str, Any]:
    """Run baseline evaluation with seed prompt."""
    print(f"\n{'='*60}")
    print("Step 4: Baseline Evaluation")
    print(f"{'='*60}")

    seed_candidate = adapter.get_candidate()
    print(f"  Evaluating with seed prompt...")
    print(f"  Prompt preview: '{seed_candidate[adapter.component_name][:100]}...'")

    # Run evaluation
    eval_batch = await adapter._evaluate_async(
        batch=training_data,
        candidate=seed_candidate,
        capture_traces=True,
    )

    baseline_score = eval_batch.aggregate_score
    success_rate = eval_batch.success_rate

    print(f"\n  Baseline Results:")
    print(f"    Aggregate Score: {baseline_score:.4f}")
    print(f"    Success Rate: {success_rate:.1%}")

    # Show per-query scores
    print(f"\n  Per-Query Scores:")
    for i, (score, trajectory) in enumerate(zip(eval_batch.scores, eval_batch.trajectories or [])):
        if trajectory:
            metrics = trajectory.get("metrics", {})
            print(f"    Query {i}: Score={score:.3f}, "
                  f"Precision={metrics.get('context_precision', 0):.3f}, "
                  f"Recall={metrics.get('context_recall', 0):.3f}")

    return {
        "baseline_score": baseline_score,
        "scores": eval_batch.scores,
        "success_rate": success_rate,
        "seed_prompt": seed_candidate[adapter.component_name],
    }


def run_gepa_optimization(
    adapter: RerankerAdapter,
    training_data: List[RAGDataInst],
    max_metric_calls: int = 20,
    reflection_lm: str = "openai/gpt-4o-mini",
) -> Dict[str, Any]:
    """Run GEPA optimization for reranker."""
    print(f"\n{'='*60}")
    print("Step 5: GEPA Optimization")
    print(f"{'='*60}")

    if not GEPA_AVAILABLE:
        print("\n  GEPA not installed. Install with: pip install gepa-ai")
        print("  Simulating optimization steps...\n")
        return simulate_optimization(adapter, training_data, max_metric_calls)

    # Split data
    split_idx = max(1, len(training_data) * 8 // 10)
    trainset = training_data[:split_idx]
    valset = training_data[split_idx:] if split_idx < len(training_data) else training_data[:2]

    print(f"  Training set: {len(trainset)} examples")
    print(f"  Validation set: {len(valset)} examples")
    print(f"  Budget: {max_metric_calls} metric calls")

    # Run optimization
    result = optimize_prompt(
        adapter=adapter,
        trainset=trainset,
        valset=valset,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
    )

    return result


def simulate_optimization(
    adapter: RerankerAdapter,
    training_data: List[RAGDataInst],
    max_metric_calls: int,
) -> Dict[str, Any]:
    """Simulate GEPA optimization when GEPA is not installed."""
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
    print(f"       - Run reranker on each query's documents")
    print(f"       - Compute precision/recall using LLM (compares to ground_truth)")
    print(f"       - Calculate position improvement score")
    print(f"    2. Capture execution traces (input docs, ranked output, scores)")
    print(f"    3. Build reflective dataset with feedback:")
    print(f"       - Positive: 'Effective reranking, moved relevant docs up'")
    print(f"       - Negative: 'Irrelevant docs ranked too high, improve criteria'")
    print(f"    4. LLM proposes improved reranking prompt based on feedback")
    print(f"    5. Evaluate new candidate")
    print(f"    6. Accept if score improves (Pareto update)")
    print(f"    7. Repeat until budget exhausted")

    print(f"\n  Reranker-Specific Metrics:")
    print(f"    - context_precision: Relevant docs in top-k after reranking")
    print(f"    - context_recall: Coverage of answer-relevant docs")
    print(f"    - position_improvement: How much relevant docs moved up")
    print(f"    - Combined score: F1 * 0.8 + position_improvement * 0.2")

    print(f"\n  Expected Prompt Evolution:")
    print(f"    Iteration 1: Add specific ranking criteria")
    print(f"    Iteration 2: Improve relevance detection")
    print(f"    Iteration 3: Better handling of partial matches")
    print(f"    Iteration N: Refined criteria for query-document alignment")

    return {
        "best_prompt": seed_prompt + "\n[OPTIMIZED - placeholder]",
        "best_score": 0.0,
        "component_name": adapter.component_name,
        "simulated": True,
    }


async def main(args):
    """Main test workflow for reranker optimization."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return

    print("\n" + "="*70)
    print("  RERANKER GEPA OPTIMIZATION TEST")
    print("="*70)
    print(f"\n  Configuration:")
    print(f"    Data path: {args.data_path}")
    print(f"    N queries: {args.n_queries}")
    print(f"    Budget: {args.budget} metric calls")
    print(f"    Top-k for reranking: {args.top_k}")

    # Step 1: Load data
    data_path = Path(args.data_path)
    documents, queries = load_train_data(data_path, n_queries=args.n_queries)

    # Step 2: Run Module 1 to get retrieved documents
    preprocessor, query_planner, retriever, retrieved_contexts = await setup_module1_pipeline(
        documents=documents,
        queries=queries,
    )

    # Step 3: Create reranker training data
    training_data = create_reranker_training_data(queries, retrieved_contexts)

    # Step 4: Setup reranker and adapter
    print(f"\n{'='*60}")
    print("Setting Up Reranker Module and Adapter")
    print(f"{'='*60}")

    reranker = RerankerModule(model_name="gpt-4o-mini")
    print(f"  Reranker initialized with seed prompt ({len(reranker.prompt)} chars)")

    evaluator = RAGASEvaluator(model="gpt-4o-mini")

    adapter = RerankerAdapter(
        reranker_module=reranker,
        evaluator=evaluator,
        top_k=args.top_k,
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
    print("  RERANKER OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Baseline Score: {baseline_results['baseline_score']:.4f}")
    if not optimization_results.get("simulated"):
        print(f"  Optimized Score: {optimization_results['best_score']:.4f}")
        improvement = optimization_results['best_score'] - baseline_results['baseline_score']
        print(f"  Improvement: {improvement:+.4f}")
    else:
        print(f"  [SIMULATION] Install gepa-ai for actual optimization")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Reranker GEPA Optimization")
    parser.add_argument("--data_path", type=str, default="data/train",
                        help="Path to training data directory")
    parser.add_argument("--n_queries", type=int, default=10,
                        help="Number of queries to use for testing")
    parser.add_argument("--budget", type=int, default=20,
                        help="GEPA optimization budget (metric calls)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top documents after reranking")
    parser.add_argument("--reflection_lm", type=str, default="openai/gpt-4o-mini",
                        help="Model for GEPA reflection")

    args = parser.parse_args()
    asyncio.run(main(args))
