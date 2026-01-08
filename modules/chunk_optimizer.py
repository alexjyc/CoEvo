"""
Chunk Size Optimizer for RAG Pipeline

Finds optimal chunk size by evaluating retrieval quality on representative queries.
Uses KMeans clustering to select diverse representative queries that cover the query space.

Primary metric: NDCG@k (order-aware, rewards relevant docs at top)
Secondary metrics: Precision@k, Recall@k, MRR

Usage:
    optimizer = ChunkSizeOptimizer(documents, queries, relevance_labels)
    best_config = await optimizer.optimize()
    print(f"Optimal chunk size: {best_config.chunk_size}")
"""

import asyncio
import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import AsyncOpenAI, OpenAI
from rank_bm25 import BM25Okapi
from sklearn.cluster import KMeans
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from modules.evaluation.metrics import RetrievalMetrics


@dataclass
class ChunkSizeConfig:
    """Configuration and results for a single chunk size evaluation."""

    chunk_size: int
    chunk_overlap: int
    num_chunks: int
    avg_chunk_length: float

    # Primary metric
    ndcg_at_k: float

    # Secondary metrics
    precision_at_k: float
    recall_at_k: float
    mrr: float
    f1: float

    # Timing
    embedding_time_seconds: float = 0.0
    eval_time_seconds: float = 0.0

    # Metadata
    k: int = 20  # retrieval k used for evaluation


@dataclass
class ChunkOptimizationResult:
    """Complete results of chunk size optimization."""

    timestamp: str
    data_path: str
    num_documents: int
    num_queries: int
    num_representative_queries: int
    embedding_model: str
    retrieval_k: int
    rrf_weight: float

    results: list[ChunkSizeConfig]
    best: ChunkSizeConfig

    total_time_seconds: float = 0.0


class ChunkSizeOptimizer:
    """
    Optimizes chunk size for maximum retrieval quality.

    Uses representative query sampling via KMeans clustering to reduce
    evaluation cost while maintaining coverage of the query space.

    Features:
    - Caches embeddings per chunk size for reuse
    - Hybrid retrieval (BM25 + Dense + wRRF fusion)
    - Multiple evaluation metrics (NDCG, Precision, Recall, MRR)
    """

    def __init__(
        self,
        documents: list[dict[str, Any]],
        queries: list[dict[str, Any]],
        relevance_labels: dict[str, list[str]],
        chunk_sizes: list[int] | None = None,
        overlap_ratio: float = 0.1,
        embedding_model: str = "text-embedding-3-small",
        retrieval_k: int = 20,
        rrf_k: int = 60,
        rrf_weight: float = 0.5,
        cache_dir: Path | None = None,
        use_cache: bool = True,
    ):
        """
        Initialize the chunk size optimizer.

        Args:
            documents: List of documents with 'doc_id' and 'content' keys
            queries: List of queries with 'query_id', 'query', 'ground_truth' keys
            relevance_labels: Dict mapping query_id -> list of relevant doc_ids
            chunk_sizes: List of chunk sizes to test (default: [256, 512, 1024, 2048])
            overlap_ratio: Chunk overlap as fraction of chunk size (default: 0.1 = 10%)
            embedding_model: OpenAI embedding model to use
            retrieval_k: Number of documents to retrieve per query
            rrf_k: RRF fusion constant
            rrf_weight: Dense retrieval weight in wRRF (0.5 = equal BM25 + Dense)
            cache_dir: Directory to cache embeddings (default: data/chunk_optimization/)
            use_cache: Whether to use cached embeddings if available
        """
        self.documents = documents
        self.queries = queries
        self.relevance_labels = relevance_labels
        self.chunk_sizes = chunk_sizes or [256, 512, 1024, 2048]
        self.overlap_ratio = overlap_ratio
        self.embedding_model = embedding_model
        self.retrieval_k = retrieval_k
        self.rrf_k = rrf_k
        self.rrf_weight = rrf_weight
        self.use_cache = use_cache

        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path("data/chunk_optimization")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # OpenAI clients
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key, timeout=60.0)
        self.async_client = AsyncOpenAI(api_key=api_key, timeout=60.0, max_retries=2)

        # Determine embedding dimension
        if "3-small" in embedding_model:
            self.embedding_dim = 1536
        elif "3-large" in embedding_model:
            self.embedding_dim = 3072
        else:
            self.embedding_dim = 1536

        # State
        self.representative_queries: list[dict[str, Any]] = []

    async def optimize(self) -> ChunkOptimizationResult:
        """
        Run full chunk size optimization.

        Returns:
            ChunkOptimizationResult with all configurations and best choice
        """
        start_time = time.time()

        print("\n" + "=" * 60)
        print("CHUNK SIZE OPTIMIZATION")
        print("=" * 60)

        # Step 1: Select representative queries
        print("\n1. Selecting representative queries...")
        self.representative_queries = await self._get_representative_queries()
        print(f"   Selected {len(self.representative_queries)} representative queries")

        # Step 2: Evaluate each chunk size
        print(f"\n2. Evaluating chunk sizes: {self.chunk_sizes}")
        results: list[ChunkSizeConfig] = []

        for chunk_size in self.chunk_sizes:
            overlap = int(chunk_size * self.overlap_ratio)
            print(f"\n   Testing chunk_size={chunk_size}, overlap={overlap}...")

            config = await self._evaluate_chunk_size(chunk_size, overlap)
            results.append(config)

            print(f"   NDCG@{self.retrieval_k}: {config.ndcg_at_k:.4f}")
            print(f"   P@{self.retrieval_k}: {config.precision_at_k:.4f}")
            print(f"   R@{self.retrieval_k}: {config.recall_at_k:.4f}")

        # Step 3: Select best configuration
        best_config = max(results, key=lambda c: c.ndcg_at_k)

        total_time = time.time() - start_time

        # Create result object
        result = ChunkOptimizationResult(
            timestamp=datetime.now().isoformat(),
            data_path=str(self.cache_dir.parent),
            num_documents=len(self.documents),
            num_queries=len(self.queries),
            num_representative_queries=len(self.representative_queries),
            embedding_model=self.embedding_model,
            retrieval_k=self.retrieval_k,
            rrf_weight=self.rrf_weight,
            results=results,
            best=best_config,
            total_time_seconds=total_time,
        )

        # Save results
        self._save_results(result)

        # Print summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"\nBest configuration:")
        print(f"  Chunk size: {best_config.chunk_size}")
        print(f"  Overlap: {best_config.chunk_overlap}")
        print(f"  NDCG@{self.retrieval_k}: {best_config.ndcg_at_k:.4f}")
        print(f"  Precision@{self.retrieval_k}: {best_config.precision_at_k:.4f}")
        print(f"  Recall@{self.retrieval_k}: {best_config.recall_at_k:.4f}")
        print(f"\nTotal time: {total_time:.1f} seconds")
        print(f"\nTo use in experiments:")
        print(
            f"  python run_research_experiment.py --chunk_size {best_config.chunk_size} --chunk_overlap {best_config.chunk_overlap}"
        )

        return result

    async def _get_representative_queries(self) -> list[dict[str, Any]]:
        """
        Select representative queries using KMeans clustering.

        Uses elbow method to determine optimal number of clusters.
        Returns the query closest to each cluster centroid.
        """
        if len(self.queries) <= 10:
            # Too few queries to cluster
            return self.queries

        # Generate embeddings for all queries
        query_texts = [q["query"] for q in self.queries]
        print("   Generating query embeddings...")
        embeddings = await self._generate_embeddings_batch(query_texts)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)

        # Find optimal k using elbow method
        max_k = min(30, len(self.queries) - 1)
        min_k = 5

        if max_k <= min_k:
            return self.queries

        print(f"   Finding optimal cluster count (k={min_k}-{max_k})...")
        inertias = []
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings_array)
            inertias.append(kmeans.inertia_)

        # Elbow detection using second derivative
        if len(inertias) > 2:
            deltas = np.diff(inertias)
            second_deltas = np.diff(deltas)
            elbow_idx = np.argmax(second_deltas) + min_k + 1
            n_clusters = elbow_idx
        else:
            n_clusters = min_k

        n_clusters = min(n_clusters, len(self.queries))
        print(f"   Optimal clusters: {n_clusters}")

        # Run final clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_array)

        # Select representative from each cluster (closest to centroid)
        representatives = []
        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_indices = np.where(cluster_mask)[0]
            cluster_embeddings = embeddings_array[cluster_indices]

            centroid = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            rep_idx = cluster_indices[np.argmin(distances)]

            query = self.queries[rep_idx]
            representatives.append(
                {
                    "query_id": query.get("query_id"),
                    "query": query.get("query"),
                    "ground_truth": query.get("ground_truth"),
                    "reference_doc_ids": query.get("reference_doc_ids"),
                    "cluster_id": i,
                    "cluster_size": len(cluster_indices),
                }
            )

        return representatives

    async def _evaluate_chunk_size(
        self, chunk_size: int, overlap: int
    ) -> ChunkSizeConfig:
        """
        Evaluate a single chunk size configuration.

        Steps:
        1. Chunk all documents
        2. Generate/load cached embeddings
        3. Build FAISS + BM25 indices
        4. Run hybrid retrieval on representative queries
        5. Compute metrics
        """
        embed_start = time.time()

        # Check cache
        cache_data = self._load_cache(chunk_size)

        if cache_data is not None:
            print(f"      Loaded cached embeddings for chunk_size={chunk_size}")
            chunks = cache_data["chunks"]
            chunk_metadata = cache_data["chunk_metadata"]
            embeddings = cache_data["embeddings"]
        else:
            # Chunk documents
            print(f"      Chunking documents...")
            chunks, chunk_metadata = self._chunk_documents(chunk_size, overlap)
            print(f"      Created {len(chunks)} chunks")

            # Generate embeddings
            print(f"      Generating embeddings...")
            embeddings = await self._generate_embeddings_batch(chunks)

            # Cache results
            self._save_cache(chunk_size, chunks, chunk_metadata, embeddings)

        embed_time = time.time() - embed_start

        # Build chunk_id -> index mapping
        chunk_id_to_index = {
            meta["chunk_id"]: i for i, meta in enumerate(chunk_metadata)
        }

        # Build doc_id -> chunk indices mapping
        doc_id_to_chunks: dict[str, list[int]] = {}
        for i, meta in enumerate(chunk_metadata):
            doc_id = meta["doc_id"]
            if doc_id not in doc_id_to_chunks:
                doc_id_to_chunks[doc_id] = []
            doc_id_to_chunks[doc_id].append(i)

        # Build indices
        bm25_index = self._build_bm25_index(chunks)
        faiss_index = self._build_faiss_index(embeddings)

        # Evaluate on representative queries
        eval_start = time.time()
        metrics_list = []

        for query_data in tqdm(
            self.representative_queries,
            desc="      Evaluating queries",
            leave=False,
        ):
            query_text = query_data["query"]
            query_id = query_data.get("query_id")

            # Get relevant chunk indices from relevance labels
            # Note: relevance_labels uses string keys, query_id may be int
            relevant_doc_ids = self.relevance_labels.get(str(query_id), [])
            relevant_indices = []
            for doc_id in relevant_doc_ids:
                if doc_id in doc_id_to_chunks:
                    relevant_indices.extend(doc_id_to_chunks[doc_id])

            if not relevant_indices:
                # Skip queries without relevance labels
                continue

            # Run hybrid retrieval
            retrieved_indices = await self._hybrid_retrieve(
                query_text,
                chunks,
                bm25_index,
                faiss_index,
                embeddings,
            )

            # Compute metrics
            metrics = RetrievalMetrics.evaluate(
                retrieved_indices=retrieved_indices,
                relevant_indices=relevant_indices,
                k=self.retrieval_k,
            )
            metrics_list.append(metrics)

        eval_time = time.time() - eval_start

        # Aggregate metrics
        if metrics_list:
            avg_ndcg = np.mean([m["ndcg"] for m in metrics_list])
            avg_precision = np.mean([m["precision"] for m in metrics_list])
            avg_recall = np.mean([m["recall"] for m in metrics_list])
            avg_mrr = np.mean([m["mrr"] for m in metrics_list])
            avg_f1 = np.mean([m["f1"] for m in metrics_list])
        else:
            avg_ndcg = avg_precision = avg_recall = avg_mrr = avg_f1 = 0.0

        return ChunkSizeConfig(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            num_chunks=len(chunks),
            avg_chunk_length=np.mean([len(c) for c in chunks]) if chunks else 0,
            ndcg_at_k=avg_ndcg,
            precision_at_k=avg_precision,
            recall_at_k=avg_recall,
            mrr=avg_mrr,
            f1=avg_f1,
            embedding_time_seconds=embed_time,
            eval_time_seconds=eval_time,
            k=self.retrieval_k,
        )

    def _chunk_documents(
        self, chunk_size: int, overlap: int
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """
        Chunk all documents with given parameters.

        Returns:
            Tuple of (chunks, chunk_metadata)
        """
        chunks = []
        chunk_metadata = []

        for doc in self.documents:
            doc_id = doc.get("doc_id", doc.get("id", f"doc_{len(chunks)}"))
            content = doc.get("content", doc.get("text", ""))

            # Clean text
            content = self._normalize_text(content)

            # Chunk with smart boundary detection
            doc_chunks = self._chunk_text(content, chunk_size, overlap)

            for i, chunk in enumerate(doc_chunks):
                chunk_id = f"{doc_id}_chunk_{i:02d}"
                chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "chunk_index": i,
                    }
                )

        return chunks, chunk_metadata

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """
        Split text into overlapping chunks with sentence boundary detection.
        """
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            # Look for sentence boundary in last 100 chars
            boundary_zone = text[max(start, end - 100) : end]
            best_boundary = -1

            for marker in [". ", "! ", "? ", "\n\n"]:
                pos = boundary_zone.rfind(marker)
                if pos > best_boundary:
                    best_boundary = pos

            if best_boundary != -1:
                actual_end = max(start, end - 100) + best_boundary + 1
            else:
                actual_end = end

            chunk = text[start:actual_end].strip()
            if chunk:
                chunks.append(chunk)

            start = actual_end - overlap

        return chunks

    def _normalize_text(self, text: str) -> str:
        """Normalize text for chunking."""
        import re

        # Add spaces around digits for better tokenization
        text = re.sub(r"(\d)", r" \1 ", text)
        # Collapse multiple whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    async def _generate_embeddings_batch(
        self, texts: list[str], batch_size: int = 512
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts using async batch processing.
        """
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            for attempt in range(5):
                try:
                    response = await self.async_client.embeddings.create(
                        model=self.embedding_model,
                        input=batch,
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt == 4:
                        raise
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)

        return all_embeddings

    def _build_bm25_index(self, chunks: list[str]) -> BM25Okapi:
        """Build BM25 index from chunks."""
        tokenized = [
            [w for w in chunk.lower().split() if len(w) >= 2] for chunk in chunks
        ]
        return BM25Okapi(tokenized)

    def _build_faiss_index(self, embeddings: list[list[float]]) -> faiss.IndexFlatIP:
        """Build FAISS index from embeddings."""
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)

        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings_array)
        return index

    async def _hybrid_retrieve(
        self,
        query: str,
        chunks: list[str],
        bm25_index: BM25Okapi,
        faiss_index: faiss.IndexFlatIP,
        embeddings: list[list[float]],
    ) -> list[int]:
        """
        Run hybrid retrieval with wRRF fusion.

        Returns:
            List of chunk indices sorted by relevance
        """
        # BM25 retrieval
        tokenized_query = [w for w in query.lower().split() if len(w) >= 2]
        bm25_scores = bm25_index.get_scores(tokenized_query)
        bm25_ranking = np.argsort(bm25_scores)[::-1][: self.retrieval_k * 2]

        # Dense retrieval
        response = await self.async_client.embeddings.create(
            model=self.embedding_model,
            input=[query],
        )
        query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        _, dense_indices = faiss_index.search(query_embedding, self.retrieval_k * 2)
        dense_ranking = dense_indices[0]

        # wRRF fusion
        scores: dict[int, float] = {}

        for rank, idx in enumerate(bm25_ranking):
            if idx not in scores:
                scores[idx] = 0.0
            scores[idx] += (1 - self.rrf_weight) / (self.rrf_k + rank + 1)

        for rank, idx in enumerate(dense_ranking):
            if idx not in scores:
                scores[idx] = 0.0
            scores[idx] += self.rrf_weight / (self.rrf_k + rank + 1)

        # Sort by score and return top-k indices
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_indices[: self.retrieval_k]

    def _get_cache_path(self, chunk_size: int) -> Path:
        """Get cache file path for a chunk size."""
        return self.cache_dir / f"embeddings_{chunk_size}" / "cache.pkl"

    def _load_cache(self, chunk_size: int) -> dict | None:
        """Load cached embeddings if available and valid."""
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(chunk_size)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            # Validate cache
            if data.get("embedding_model") != self.embedding_model:
                return None
            if data.get("overlap_ratio") != self.overlap_ratio:
                return None
            if data.get("num_documents") != len(self.documents):
                return None

            return data
        except Exception:
            return None

    def _save_cache(
        self,
        chunk_size: int,
        chunks: list[str],
        chunk_metadata: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """Save embeddings to cache."""
        cache_path = self._get_cache_path(chunk_size)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "chunk_size": chunk_size,
            "overlap_ratio": self.overlap_ratio,
            "embedding_model": self.embedding_model,
            "num_documents": len(self.documents),
            "chunks": chunks,
            "chunk_metadata": chunk_metadata,
            "embeddings": embeddings,
            "timestamp": datetime.now().isoformat(),
        }

        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def _save_results(self, result: ChunkOptimizationResult) -> None:
        """Save optimization results to JSON."""
        results_path = self.cache_dir / "chunk_size_results.json"

        # Convert dataclasses to dicts
        output = {
            "timestamp": result.timestamp,
            "data_path": result.data_path,
            "num_documents": result.num_documents,
            "num_queries": result.num_queries,
            "num_representative_queries": result.num_representative_queries,
            "embedding_model": result.embedding_model,
            "retrieval_k": result.retrieval_k,
            "rrf_weight": result.rrf_weight,
            "total_time_seconds": result.total_time_seconds,
            "results": [asdict(r) for r in result.results],
            "best": asdict(result.best),
            "recommendation": f"Use --chunk_size {result.best.chunk_size} --chunk_overlap {result.best.chunk_overlap}",
        }

        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {results_path}")
