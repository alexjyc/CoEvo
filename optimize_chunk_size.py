#!/usr/bin/env python3
"""
Chunk Size Optimization Script for RAG Pipeline

Finds the optimal chunk size by evaluating retrieval quality (NDCG) on
representative queries selected via KMeans clustering.

Usage:
    python optimize_chunk_size.py --data_path data/train/ --output_dir data/chunk_optimization/
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from modules.chunk_optimizer import ChunkSizeOptimizer


def load_json(path: Path) -> dict | list:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


async def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)

    print(f"\nLoading data from: {data_path}")

    # Load documents
    documents_path = data_path / "documents.json"
    if not documents_path.exists():
        print(f"ERROR: {documents_path} not found")
        sys.exit(1)
    documents = load_json(documents_path)
    print(f"  Loaded {len(documents)} documents")

    # Load queries
    queries_path = data_path / "queries.json"
    if not queries_path.exists():
        print(f"ERROR: {queries_path} not found")
        sys.exit(1)
    queries = load_json(queries_path)
    print(f"  Loaded {len(queries)} queries")

    # Load relevance labels
    labels_path = data_path / "relevance_labels.json"
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found")
        sys.exit(1)
    relevance_labels = load_json(labels_path)
    print(f"  Loaded relevance labels for {len(relevance_labels)} queries")

    # Parse chunk sizes
    chunk_sizes = args.chunk_sizes
    if not chunk_sizes:
        chunk_sizes = [256, 512, 1024, 2048]

    print(f"\nChunk sizes to test: {chunk_sizes}")
    print(f"Overlap ratio: {args.overlap_ratio}")
    print(f"Retrieval k: {args.retrieval_k}")
    print(f"Output directory: {output_dir}")

    # Initialize optimizer
    optimizer = ChunkSizeOptimizer(
        documents=documents,
        queries=queries,
        relevance_labels=relevance_labels,
        chunk_sizes=chunk_sizes,
        overlap_ratio=args.overlap_ratio,
        embedding_model=args.embedding_model,
        retrieval_k=args.retrieval_k,
        rrf_k=args.rrf_k,
        rrf_weight=args.rrf_weight,
        cache_dir=output_dir,
        use_cache=not args.no_cache,
    )

    # Run optimization
    result = await optimizer.optimize()

    # Print final recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"\nBest chunk size: {result.best.chunk_size}")
    print(f"Best overlap: {result.best.chunk_overlap}")
    print(f"NDCG@{args.retrieval_k}: {result.best.ndcg_at_k:.4f}")
    print("\nCommand to use:")
    print(
        f"  python run_research_experiment.py \\\n"
        f"    --chunk_size {result.best.chunk_size} \\\n"
        f"    --chunk_overlap {result.best.chunk_overlap}"
    )
    print("\nOr load from config:")
    print(
        f"  python run_research_experiment.py \\\n"
        f"    --chunk_size_config {output_dir / 'chunk_size_results.json'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize chunk size for RAG pipeline retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test default chunk sizes [256, 512, 1024, 2048]
    python scripts/optimize_chunk_size.py --data_path data/train/

    # Test specific chunk sizes
    python scripts/optimize_chunk_size.py --data_path data/train/ --chunk_sizes 300 500 700 1000

    # Use different embedding model
    python scripts/optimize_chunk_size.py --data_path data/train/ --embedding_model text-embedding-3-large
        """,
    )

    # Required arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/train",
        help="Path to data directory containing documents.json, queries.json, relevance_labels.json",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/chunk_optimization",
        help="Directory to save results and cache embeddings",
    )

    # Chunk size options
    parser.add_argument(
        "--chunk_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Chunk sizes to test (default: 256 512 1024 2048)",
    )
    parser.add_argument(
        "--overlap_ratio",
        type=float,
        default=0.1,
        help="Chunk overlap as fraction of chunk size (default: 0.1 = 10%%)",
    )

    # Retrieval options
    parser.add_argument(
        "--retrieval_k",
        type=int,
        default=20,
        help="Number of documents to retrieve per query (default: 20)",
    )
    parser.add_argument(
        "--rrf_k",
        type=int,
        default=60,
        help="RRF fusion constant (default: 60)",
    )
    parser.add_argument(
        "--rrf_weight",
        type=float,
        default=0.5,
        help="Dense retrieval weight in wRRF fusion (default: 0.5)",
    )

    # Embedding options
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model (default: text-embedding-3-small)",
    )

    # Cache options
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache - regenerate all embeddings",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
