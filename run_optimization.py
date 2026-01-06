"""
RAG Pipeline Optimization Runner V2 (Iterative/Staged)

Implements Iterative (Staged) Optimization:
1. Optimize Query Planner -> Update Pipeline
2. Optimize Reranker (using optimized QP) -> Update Pipeline
3. Optimize Generator (using optimized QP + Reranker) -> Update Pipeline

This ensures downstream modules are trained on the improved input distribution
from upstream modules.

Usage:
    python run_optimization_v2.py --data_path data/train/ --output_dir gepa_runs_v2/ 
"""

import asyncio
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Import modules
from modules.preprocessor import DocumentPreprocessor
from modules.pipeline import ModularRAGPipeline, PipelineConfig
from modules.evaluation import RAGASEvaluator

# Import GEPA adapters
from gepa_adapters import (
    QueryPlannerAdapter,
    RerankerAdapter,
    GeneratorAdapter,
)

# Import training components
from training import (
    CRAGTrainingGenerator,
    GEPAOptimizationRunner,
    TrainingDataset,
)

# Reuse load_data from original script
from run_optimization import load_data


async def main(args):
    """Main optimization workflow (Iterative)"""
    load_dotenv()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return

    # Setup paths
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    documents, evaluation_queries, relevance_labels_doc = await load_data(data_path)

    # Step 2: Setup preprocessor and pipeline
    print("\n" + "="*60)
    print("Setting up RAG Pipeline")
    print("="*60)

    preprocessor = DocumentPreprocessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    preprocessor.process_documents(documents, use_context=args.use_context)

    if relevance_labels_doc:
        labels_path = data_path / "relevance_labels.json"
        relevance_labels = preprocessor.load_relevance_labels(str(labels_path))
    else:
        relevance_labels = preprocessor.create_relevance_labels(evaluation_queries)

    # Setup pipeline
    config = PipelineConfig(
        llm_model=args.model,
        retrieval_k=args.retrieval_k,
        final_k=args.final_k,
        prompt_dir=output_dir / "prompts"  # Save prompts to output dir
    )
    pipeline = ModularRAGPipeline(preprocessor, config)

    print(f"Pipeline ready with {len(preprocessor.chunks)} chunks")

    # Step 3: Setup evaluator and GEPA adapters
    evaluator = RAGASEvaluator(model=args.model)

    query_planner_adapter = QueryPlannerAdapter(
        query_planner_module=pipeline.query_planner,
        retriever_module=pipeline.retriever,
        evaluator=evaluator,
    )

    reranker_adapter = RerankerAdapter(
        reranker_module=pipeline.reranker,
        evaluator=evaluator,
        top_k=args.final_k,
    )

    generator_adapter = GeneratorAdapter(
        generator_module=pipeline.generator,
        evaluator=evaluator,
    )

    # Setup Generator and Optimizer
    training_generator = CRAGTrainingGenerator(
        pipeline=pipeline,
        evaluator=evaluator,
    )

    optimizer = GEPAOptimizationRunner(
        query_planner_adapter=query_planner_adapter,
        reranker_adapter=reranker_adapter,
        generator_adapter=generator_adapter,
        output_dir=output_dir,
        budget=args.budget,
    )

    # =========================================================================
    # STAGE 1: Optimize Query Planner
    # =========================================================================
    print("\n" + "="*60)
    print("STAGE 1: Optimizing Query Planner")
    print("="*60)

    # Generate data with baseline pipeline
    dataset_1 = await training_generator.generate_training_data(
        evaluation_queries=evaluation_queries,
        relevance_labels=relevance_labels,
        output_dir=output_dir / "iter_1_data",
    )

    # Optimize Query Planner
    if dataset_1.query_planner_examples:
        qp_results = optimizer.optimize_module(
            "query_planner",
            dataset_1.query_planner_examples,
        )
        
        # UPDATE PIPELINE
        print(f"\n[UPDATE] Applying optimized Query Planner prompt for next stage")
        pipeline.query_planner.prompt = qp_results["optimized_prompt"]
    else:
        print("No query planner examples generated. Skipping optimization.")

    # =========================================================================
    # STAGE 2: Optimize Reranker
    # =========================================================================
    print("\n" + "="*60)
    print("STAGE 2: Optimizing Reranker")
    print("="*60)
    print("Generating data with optimized Query Planner...")

    # Generate data with optimized QP
    dataset_2 = await training_generator.generate_training_data(
        evaluation_queries=evaluation_queries,
        relevance_labels=relevance_labels,
        output_dir=output_dir / "iter_2_data",
    )

    # Optimize Reranker
    if dataset_2.reranker_examples:
        rr_results = optimizer.optimize_module(
            "reranker",
            dataset_2.reranker_examples,
        )
        
        # UPDATE PIPELINE
        print(f"\n[UPDATE] Applying optimized Reranker prompt for next stage")
        pipeline.reranker.prompt = rr_results["optimized_prompt"]
    else:
        print("No reranker examples generated. Skipping optimization.")

    # =========================================================================
    # STAGE 3: Optimize Generator
    # =========================================================================
    print("\n" + "="*60)
    print("STAGE 3: Optimizing Generator")
    print("="*60)
    print("Generating data with optimized QP and Reranker...")

    # Generate data with optimized QP + RR
    dataset_3 = await training_generator.generate_training_data(
        evaluation_queries=evaluation_queries,
        relevance_labels=relevance_labels,
        output_dir=output_dir / "iter_3_data",
    )

    # Optimize Generator
    if dataset_3.generator_examples:
        gen_results = optimizer.optimize_module(
            "generator",
            dataset_3.generator_examples,
        )
        
        # UPDATE PIPELINE
        print(f"\n[UPDATE] Applying optimized Generator prompt")
        pipeline.generator.prompt = gen_results["optimized_prompt"]
    else:
        print("No generator examples generated. Skipping optimization.")

    # =========================================================================
    # Finalize
    # =========================================================================
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)

    # Save all final prompts
    pipeline.save_prompts(version="final_optimized")
    print(f"Final prompts saved to: {output_dir}/prompts/")
    print(f"Full results in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline Optimization V2 (Iterative)")

    parser.add_argument("--data_path", type=str, default="data/train",
                        help="Path to training data directory")
    parser.add_argument("--output_dir", type=str, default="gepa_runs_v2",
                        help="Output directory for optimization results")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model to use")
    parser.add_argument("--chunk_size", type=int, default=600,
                        help="Document chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=50,
                        help="Chunk overlap size")
    parser.add_argument("--retrieval_k", type=int, default=20,
                        help="Number of documents to retrieve")
    parser.add_argument("--final_k", type=int, default=10,
                        help="Number of documents after reranking")
    parser.add_argument("--budget", type=int, default=100,
                        help="GEPA optimization budget (iterations)")
    parser.add_argument("--use_context", action="store_true",
                        help="Use contextual retrieval")

    args = parser.parse_args()
    asyncio.run(main(args))