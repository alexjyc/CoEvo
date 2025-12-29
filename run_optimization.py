"""
RAG Pipeline Optimization Runner

Main script to run GEPA optimization for all modules using CRAG benchmark.

Usage:
    python run_optimization.py --data_path data/train/ --output_dir gepa_runs/

Workflow:
1. Load documents and evaluation queries
2. Setup modular RAG pipeline
3. Generate CRAG training data
4. Create GEPA adapters for each module
5. Run optimization for each module
6. Save optimized prompts and results
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


async def load_data(data_path: Path) -> tuple:
    """
    Load documents and evaluation queries from data directory.

    Expected structure (document-level from preprocess_crag.py):
        data_path/
        ├── documents.json      # List of {doc_id, text, title, url} - full documents
        ├── queries.json        # List of {query_id, query, ground_truth, reference_doc_ids}
        └── relevance_labels.json  # {query_id: [doc_ids]} - document-level ground truth

    Returns:
        Tuple of (documents, evaluation_queries, relevance_labels_doc)

    Note: Documents will be chunked by DocumentPreprocessor.
          Document-level relevance labels will be converted to chunk-level.
    """
    print(f"Loading data from {data_path}...")

    # Load documents (full documents, will be chunked by preprocessor)
    docs_file = data_path / "documents.json"
    if docs_file.exists():
        with open(docs_file) as f:
            documents = json.load(f)
        print(f"  Loaded {len(documents)} documents")
    else:
        # Create sample documents if not found
        documents = create_sample_documents()
        print(f"  Created {len(documents)} sample documents")

    # Load evaluation queries
    queries_file = data_path / "queries.json"
    if queries_file.exists():
        with open(queries_file) as f:
            evaluation_queries = json.load(f)
        print(f"  Loaded {len(evaluation_queries)} queries")
    else:
        # Create sample queries if not found
        evaluation_queries = create_sample_queries()
        print(f"  Created {len(evaluation_queries)} sample queries")

    # Load document-level relevance labels (from preprocess_crag.py)
    # These will be converted to chunk-level by DocumentPreprocessor
    labels_file = data_path / "relevance_labels.json"
    if labels_file.exists():
        with open(labels_file) as f:
            relevance_labels_doc = {int(k): v for k, v in json.load(f).items()}
        print(f"  Loaded document-level relevance labels for {len(relevance_labels_doc)} queries")
    else:
        relevance_labels_doc = None
        print("  No relevance labels found (will be generated from queries)")

    return documents, evaluation_queries, relevance_labels_doc


def create_sample_documents() -> list:
    """
    Create sample documents for testing (document-level, not chunked).

    Format matches preprocess_crag.py output:
    - doc_id: Unique document identifier
    - text: Full document text (will be chunked by DocumentPreprocessor)
    - title: Document title (optional)
    - url: Source URL (optional)

    The DocumentPreprocessor will:
    1. Chunk these documents
    2. Generate chunk IDs: {doc_id}_chunk_{index:02d}
    3. Create chunk-level relevance labels
    """
    return [
        {
            "doc_id": "revenue_2023",
            "text": """Company XYZ Financial Report - Fiscal Year 2023

            Revenue Performance:
            Total revenue for fiscal year 2023 reached $150 million, representing a 25%
            year-over-year increase. This growth was primarily driven by the cloud services
            division, which experienced 45% growth, contributing $67 million to total revenue.

            Geographic Distribution:
            - North America: $85 million (57%)
            - Europe: $40 million (27%)
            - Asia Pacific: $25 million (16%)

            Key Growth Drivers:
            1. Enterprise cloud adoption accelerated post-pandemic
            2. New product launches in Q2 and Q3
            3. Strategic partnerships with major tech companies"""
        },
        {
            "doc_id": "expenses_2023",
            "text": """Operating Expenses Analysis - 2023

            Total operating expenses for fiscal 2023 were $120 million, representing
            an 18% increase from the previous year. The breakdown is as follows:

            Research & Development: $50 million
            - AI/ML capabilities: $20 million
            - Platform improvements: $15 million
            - Security enhancements: $15 million

            Sales & Marketing: $30 million
            - Digital marketing: $12 million
            - Sales team expansion: $10 million
            - Brand campaigns: $8 million

            General & Administrative: $40 million
            - Employee compensation: $25 million
            - Infrastructure: $10 million
            - Legal & compliance: $5 million"""
        },
        {
            "doc_id": "profit_2023",
            "text": """Profitability Summary - Fiscal 2023

            Net Profit: $30 million
            Profit Margin: 20%

            This represents a significant improvement from the 15% profit margin achieved
            in fiscal 2022. The improved profitability was attributed to:

            1. Revenue Growth: 25% increase in total revenue
            2. Operational Efficiency: Streamlined operations reduced costs by 8%
            3. Scale Benefits: Cloud infrastructure costs decreased per-unit

            EBITDA: $45 million (30% margin)
            Free Cash Flow: $28 million

            The company maintains a strong balance sheet with $75 million in cash
            reserves and no long-term debt."""
        },
        {
            "doc_id": "strategy_2024",
            "text": """Strategic Outlook - 2024 and Beyond

            Growth Targets:
            - Revenue target: $190 million (27% growth)
            - Profit margin target: 22%
            - R&D investment: Increase to $65 million

            Strategic Priorities:
            1. AI Integration: Embed AI capabilities across all products
            2. International Expansion: Focus on APAC and LATAM markets
            3. Enterprise Focus: Increase enterprise customer base by 40%
            4. Platform Development: Launch next-gen platform in Q2

            Risk Factors:
            - Competitive pressure from major tech companies
            - Economic uncertainty in key markets
            - Talent acquisition challenges"""
        },
    ]


def create_sample_queries() -> list:
    """
    Create sample evaluation queries (document-level references).

    Format matches preprocess_crag.py output:
    - reference_doc_ids: List of relevant document IDs
    - reference_evidence_texts: Evidence snippets from relevant documents

    The DocumentPreprocessor will convert document-level relevance to
    chunk-level relevance after chunking.
    """
    return [
        {
            "query_id": 0,
            "query": "What was the company's total revenue in 2023?",
            "ground_truth": "Total revenue for fiscal year 2023 reached $150 million, representing a 25% year-over-year increase.",
            "reference_doc_ids": ["revenue_2023"],
            "reference_evidence_texts": [
                {"doc_id": "revenue_2023",
                 "evidence_text": "Total revenue for fiscal year 2023 reached $150 million"}
            ]
        },
        {
            "query_id": 1,
            "query": "How much did the company spend on R&D in 2023?",
            "ground_truth": "Research & Development spending was $50 million in 2023.",
            "reference_doc_ids": ["expenses_2023"],
            "reference_evidence_texts": [
                {"doc_id": "expenses_2023",
                 "evidence_text": "Research & Development: $50 million"}
            ]
        },
        {
            "query_id": 2,
            "query": "What was the profit margin in 2023 and how did it compare to 2022?",
            "ground_truth": "The profit margin was 20% in 2023, up from 15% in 2022.",
            "reference_doc_ids": ["profit_2023"],
            "reference_evidence_texts": [
                {"doc_id": "profit_2023",
                 "evidence_text": "Profit Margin: 20%"},
                {"doc_id": "profit_2023",
                 "evidence_text": "improvement from the 15% profit margin achieved in fiscal 2022"}
            ]
        },
        {
            "query_id": 3,
            "query": "What are the company's revenue targets for 2024?",
            "ground_truth": "The revenue target for 2024 is $190 million, representing 27% growth.",
            "reference_doc_ids": ["strategy_2024"],
            "reference_evidence_texts": [
                {"doc_id": "strategy_2024",
                 "evidence_text": "Revenue target: $190 million (27% growth)"}
            ]
        },
        {
            "query_id": 4,
            "query": "What drove the cloud services growth?",
            "ground_truth": "Cloud services grew 45% year-over-year, driven by enterprise cloud adoption and new product launches.",
            "reference_doc_ids": ["revenue_2023"],
            "reference_evidence_texts": [
                {"doc_id": "revenue_2023",
                 "evidence_text": "cloud services division, which experienced 45% growth"},
                {"doc_id": "revenue_2023",
                 "evidence_text": "Enterprise cloud adoption accelerated"}
            ]
        },
    ]


async def main(args):
    """Main optimization workflow"""
    load_dotenv()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return

    # Setup paths
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data (document-level from preprocess_crag.py)
    documents, evaluation_queries, relevance_labels_doc = await load_data(data_path)

    # Step 2: Setup preprocessor and pipeline
    print("\n" + "="*60)
    print("Setting up RAG Pipeline")
    print("="*60)

    preprocessor = DocumentPreprocessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Process documents: chunks them and generates chunk IDs
    # Chunk IDs are formatted as: {doc_id}_chunk_{index:02d}
    preprocessor.process_documents(documents, use_context=args.use_context)

    # Convert document-level labels to chunk-level indices
    # All chunks from a relevant document are considered relevant
    if relevance_labels_doc:
        # Load from file and convert doc_ids to chunk indices
        labels_path = data_path / "relevance_labels.json"
        relevance_labels = preprocessor.load_relevance_labels(str(labels_path))
    else:
        # No labels file: create from queries using reference_doc_ids
        relevance_labels = preprocessor.create_relevance_labels(evaluation_queries)

    # Setup pipeline
    config = PipelineConfig(
        llm_model=args.model,
        retrieval_k=args.retrieval_k,
        final_k=args.final_k,
    )
    pipeline = ModularRAGPipeline(preprocessor, config)

    print(f"Pipeline ready with {len(preprocessor.chunks)} chunks")

    # Step 3: Setup evaluator and GEPA adapters
    print("\n" + "="*60)
    print("Setting up GEPA Adapters")
    print("="*60)

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

    print("Adapters created for: query_planner, reranker, generator")

    # Step 4: Generate CRAG training data
    print("\n" + "="*60)
    print("Generating CRAG Training Data")
    print("="*60)

    training_generator = CRAGTrainingGenerator(
        pipeline=pipeline,
        evaluator=evaluator,
    )

    training_dataset = await training_generator.generate_training_data(
        evaluation_queries=evaluation_queries,
        relevance_labels=relevance_labels,
        output_dir=output_dir / "training_data",
    )

    # Step 5: Run GEPA optimization
    print("\n" + "="*60)
    print("Running GEPA Optimization")
    print("="*60)

    optimizer = GEPAOptimizationRunner(
        query_planner_adapter=query_planner_adapter,
        reranker_adapter=reranker_adapter,
        generator_adapter=generator_adapter,
        output_dir=output_dir,
        budget=args.budget,
    )

    results = await optimizer.optimize_all_modules(training_dataset)

    # Step 6: Summary
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)

    for module_name, module_results in results.items():
        print(f"\n{module_name}:")
        print(f"  Baseline: {module_results['baseline_score']:.4f}")
        print(f"  Best: {module_results['best_score']:.4f}")
        print(f"  Improvement: {module_results['improvement']:+.4f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline Optimization with GEPA")

    parser.add_argument("--data_path", type=str, default="data/train",
                        help="Path to training data directory")
    parser.add_argument("--output_dir", type=str, default="gepa_runs",
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
