"""
Generator GEPA Optimization Test

Tests the generator module optimization using gepa.optimize()
following the crag_benchmark implementation pattern from gepa-ai/gepa.

Usage:
    python test_generator_gepa.py --n_queries 10

Steps:
1. Load first N queries from train.json
2. Run Module 1 (Query Planner + Retrieval) to get documents
3. Run Module 2 (Reranker) to get reranked documents
4. Create training data with reranked context
5. Initialize generator module and adapter
6. Call gepa.optimize() with GeneratorAdapter
7. Report optimization results

Note: Generator depends on Module 1 + Module 2 output (reranked documents)
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
from modules.generator.generator import GeneratorModule
from modules.evaluation import RAGASEvaluator
from modules.base import QueryPlannerInput, RetrievalInput, RerankerInput

# Import GEPA adapter
from gepa_adapters.generator_adapter import GeneratorAdapter
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

    # Get relevant document IDs
    relevant_doc_ids = set()
    for q in queries:
        relevant_doc_ids.update(q.get("reference_doc_ids", []))

    # Filter documents
    documents = [d for d in all_documents if d["doc_id"] in relevant_doc_ids]

    print(f"\n  Selected {n_queries} queries with {len(documents)} relevant documents")

    return documents, queries


async def setup_pipeline_and_get_contexts(
    documents: List[Dict],
    queries: List[Dict],
    chunk_size: int = 600,
    chunk_overlap: int = 50,
    top_k: int = 10,
) -> Dict[int, List[str]]:
    """
    Run Module 1 + Module 2 to get reranked contexts for generator.

    Returns:
        Dict mapping query_id to list of reranked document texts
    """
    print(f"\n{'='*60}")
    print("Step 2: Running Module 1 + Module 2 Pipeline")
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

    # Initialize modules
    query_planner = QueryPlannerModule(model_name="gpt-4o-mini")
    retriever = HybridRetriever(
        preprocessor=preprocessor,
        rrf_k=60,
        default_dense_weight=0.5,
    )
    reranker = RerankerModule(model_name="gpt-4o-mini")

    print(f"  Query planner, retriever, and reranker initialized")

    # Run pipeline for each query
    print("\n  Running full pipeline for each query...")
    reranked_contexts: Dict[int, List[str]] = {}

    for q in queries:
        query_id = q["query_id"]
        query_text = q["query"]

        # Module 1: Query Planning
        planner_input = QueryPlannerInput(query=query_text)
        planner_output = await query_planner.run(planner_input)

        # Module 1: Retrieval
        retrieval_input = RetrievalInput(
            queries=planner_output.queries,
            top_k=20
        )
        retrieval_output = await retriever.run(retrieval_input)

        # Module 2: Reranking
        reranker_input = RerankerInput(
            query=query_text,
            documents=retrieval_output.document_texts,
        )
        reranker_output = await reranker.run(reranker_input)

        # Take top-k reranked documents as context for generator
        reranked_docs = reranker_output.ranked_documents[:top_k]
        reranked_contexts[query_id] = reranked_docs

        print(f"    Query {query_id}: {len(planner_output.queries)} sub-queries "
              f"→ {len(retrieval_output.document_texts)} retrieved "
              f"→ {len(reranked_docs)} reranked")

    return reranked_contexts


def create_generator_training_data(
    queries: List[Dict],
    reranked_contexts: Dict[int, List[str]],
) -> List[RAGDataInst]:
    """
    Create training data for generator in RAGDataInst format.

    Generator needs:
    - query: The user query
    - contexts: Reranked documents (from Module 2)
    - ground_truth: Expected answer (for correctness evaluation)
    """
    print(f"\n{'='*60}")
    print("Step 3: Creating Generator Training Data")
    print(f"{'='*60}")

    training_data: List[RAGDataInst] = []

    for q in queries:
        query_id = q["query_id"]
        contexts = reranked_contexts.get(query_id, [])

        if not contexts:
            print(f"  Skipping query {query_id}: No reranked documents")
            continue

        data_inst: RAGDataInst = {
            "query": q["query"],
            "ground_truth": q.get("ground_truth", ""),
            "relevant_chunk_indices": None,  # Generator uses LLM-based eval
            "contexts": contexts,  # Reranked documents from Module 2
            "metadata": {
                "query_id": query_id,
                "num_contexts": len(contexts),
                "total_context_length": sum(len(c) for c in contexts),
                "split": "train",
            }
        }
        training_data.append(data_inst)

        print(f"  Query {query_id}: '{q['query'][:50]}...'")
        print(f"    → {len(contexts)} context documents ({sum(len(c) for c in contexts)} chars)")
        if q.get("ground_truth"):
            print(f"    → Ground truth: '{q['ground_truth'][:60]}...'")

    print(f"\n  Total training instances: {len(training_data)}")

    return training_data


async def run_baseline_evaluation(
    adapter: GeneratorAdapter,
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
    print(f"    Aggregate Score (Generation Quality): {baseline_score:.4f}")
    print(f"    Success Rate: {success_rate:.1%}")

    # Show per-query scores
    print(f"\n  Per-Query Scores:")
    for i, (score, trajectory) in enumerate(zip(eval_batch.scores, eval_batch.trajectories or [])):
        if trajectory:
            metrics = trajectory.get("metrics", {})
            print(f"    Query {i}: Quality={score:.3f}, "
                  f"Faithfulness={metrics.get('faithfulness', 0):.3f}, "
                  f"Correctness={metrics.get('answer_correctness', 0):.3f}")

    return {
        "baseline_score": baseline_score,
        "scores": eval_batch.scores,
        "success_rate": success_rate,
        "seed_prompt": seed_candidate[adapter.component_name],
    }


def run_gepa_optimization(
    adapter: GeneratorAdapter,
    training_data: List[RAGDataInst],
    max_metric_calls: int = 20,
    reflection_lm: str = "openai/gpt-4o-mini",
) -> Dict[str, Any]:
    """Run GEPA optimization for generator."""
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
    adapter: GeneratorAdapter,
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
    print(f"       - Generate answer for each query using context")
    print(f"       - Compute faithfulness (LLM judges grounding in context)")
    print(f"       - Compute answer_correctness (LLM compares to ground truth)")
    print(f"    2. Capture execution traces:")
    print(f"       - Query and context input")
    print(f"       - Generated answer, reference, rationale")
    print(f"       - Ground truth comparison")
    print(f"    3. Build reflective dataset with feedback:")
    print(f"       - Positive: 'Well-grounded answer, matches expected response'")
    print(f"       - Negative: 'Contains hallucinations' or 'Incorrect answer'")
    print(f"    4. LLM proposes improved generation prompt based on feedback")
    print(f"    5. Evaluate new candidate")
    print(f"    6. Accept if score improves (Pareto update)")
    print(f"    7. Repeat until budget exhausted")

    print(f"\n  Generator-Specific Metrics:")
    print(f"    - faithfulness: Is answer grounded in context? (0-1)")
    print(f"    - answer_correctness: Does answer match ground truth? (0-1)")
    print(f"    - generation_quality: (faithfulness + correctness) / 2")

    print(f"\n  Expected Prompt Evolution:")
    print(f"    Iteration 1: Add explicit grounding instructions")
    print(f"    Iteration 2: Improve answer structure")
    print(f"    Iteration 3: Better reference extraction")
    print(f"    Iteration 4: Handle edge cases (insufficient context)")
    print(f"    Iteration N: Refined instructions for accuracy + faithfulness")

    print(f"\n  Key Optimization Targets:")
    print(f"    - Reduce hallucinations (improve faithfulness)")
    print(f"    - Match expected answer format (improve correctness)")
    print(f"    - Include supporting evidence (references)")
    print(f"    - Explain reasoning (rationale)")

    return {
        "best_prompt": seed_prompt + "\n[OPTIMIZED - placeholder]",
        "best_score": 0.0,
        "component_name": adapter.component_name,
        "simulated": True,
    }


async def main(args):
    """Main test workflow for generator optimization."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return

    print("\n" + "="*70)
    print("  GENERATOR GEPA OPTIMIZATION TEST")
    print("="*70)
    print(f"\n  Configuration:")
    print(f"    Data path: {args.data_path}")
    print(f"    N queries: {args.n_queries}")
    print(f"    Budget: {args.budget} metric calls")
    print(f"    Top-k context documents: {args.top_k}")

    # Step 1: Load data
    data_path = Path(args.data_path)
    documents, queries = load_train_data(data_path, n_queries=args.n_queries)

    # Step 2: Run Module 1 + Module 2 to get reranked contexts
    reranked_contexts = await setup_pipeline_and_get_contexts(
        documents=documents,
        queries=queries,
        top_k=args.top_k,
    )

    # Step 3: Create generator training data
    training_data = create_generator_training_data(queries, reranked_contexts)

    # Step 4: Setup generator and adapter
    print(f"\n{'='*60}")
    print("Setting Up Generator Module and Adapter")
    print(f"{'='*60}")

    generator = GeneratorModule(model_name="gpt-4o-mini")
    print(f"  Generator initialized with seed prompt ({len(generator.prompt)} chars)")

    evaluator = RAGASEvaluator(model="gpt-4o-mini")

    adapter = GeneratorAdapter(
        generator_module=generator,
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
    print("  GENERATOR OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Baseline Score: {baseline_results['baseline_score']:.4f}")
    if not optimization_results.get("simulated"):
        print(f"  Optimized Score: {optimization_results['best_score']:.4f}")
        improvement = optimization_results['best_score'] - baseline_results['baseline_score']
        print(f"  Improvement: {improvement:+.4f}")
    else:
        print(f"  [SIMULATION] Install gepa-ai for actual optimization")

    print(f"\n  Module Dependencies:")
    print(f"    Module 1 (Query Planner + Retrieval) → provides initial documents")
    print(f"    Module 2 (Reranker) → provides reranked documents")
    print(f"    Module 3 (Generator) → generates answer from reranked context")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Generator GEPA Optimization")
    parser.add_argument("--data_path", type=str, default="data/train",
                        help="Path to training data directory")
    parser.add_argument("--n_queries", type=int, default=10,
                        help="Number of queries to use for testing")
    parser.add_argument("--budget", type=int, default=20,
                        help="GEPA optimization budget (metric calls)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top documents to use as context")
    parser.add_argument("--reflection_lm", type=str, default="openai/gpt-4o-mini",
                        help="Model for GEPA reflection")

    args = parser.parse_args()
    asyncio.run(main(args))
