"""
Retrieval Component for Query Planner Module

Handles hybrid retrieval (Dense + BM25) with weighted Reciprocal Rank Fusion (wRRF).
This component works with the Query Planner to form Module 1.

Optimizations:
- Batch query embedding (single API call for multiple queries)
- Async embedding with AsyncOpenAI
- Optimal RRF weight finding via grid search

No prompt optimization here - retrieval is algorithmic.
Optimization is done via wRRF weight tuning.
"""

import os

# Fix OpenMP conflict on macOS - must be set before importing faiss/numpy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import asyncio
import hashlib
from typing import Any

import faiss
import numpy as np
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

from modules.base import (
    Module,
    ModuleType,
    RetrievalInput,
    RetrievalOutput,
)


class HybridRetriever(Module[RetrievalInput, RetrievalOutput]):
    """
    Hybrid Retrieval Module combining BM25 and Dense retrieval with wRRF fusion.

    This module:
    1. Takes queries from QueryPlannerModule
    2. Retrieves documents using BM25 and Dense (FAISS)
    3. Fuses results using weighted Reciprocal Rank Fusion

    No LLM prompt - optimization is via wRRF weight tuning.
    """

    def __init__(
        self,
        preprocessor,  # DocumentPreprocessor instance
        rrf_k: int = 60,
        default_dense_weight: float = 0.5,
        relevance_labels: dict[int, list[int]] | None = None,
        evaluator: Any | None = None,
    ):
        super().__init__(ModuleType.RETRIEVAL)
        self.preprocessor = preprocessor
        self.rrf_k = rrf_k
        self.dense_weight = default_dense_weight
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key, timeout=60.0)
        self.async_client = AsyncOpenAI(api_key=api_key, timeout=60.0, max_retries=2)

        # For optimal RRF weight finding
        self.relevance_labels = relevance_labels or {}
        self.query_id_map: dict[str, int] = {}
        self.evaluator = evaluator

        # Embedding cache for repeated queries
        self._embedding_cache: dict[str, list[float]] = {}

        # No prompt for retrieval module
        self._prompt = None

    def get_default_prompt(self) -> str:
        """Retrieval has no prompt - it's algorithmic"""
        return ""

    def set_relevance_labels(
        self, evaluation_queries: list[dict[str, Any]], relevance_labels: dict[int, list[int]]
    ) -> None:
        """Set relevance labels for optimal weight finding."""
        self.relevance_labels = relevance_labels
        self.query_id_map = {
            self._hash_query(q["query"]): q["query_id"] for q in evaluation_queries
        }

    @staticmethod
    def _hash_query(query: str) -> str:
        """Hash query for lookup."""
        normalized = " ".join(query.lower().strip().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query (with caching)."""
        # Check cache first
        cache_key = self._hash_query(query)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        model_name = getattr(self.preprocessor, "embedding_model", "text-embedding-3-small")

        try:
            response = self.client.embeddings.create(input=query, model=model_name)
            emb = np.array(response.data[0].embedding, dtype=np.float32)
            faiss.normalize_L2(emb.reshape(1, -1))
            result = emb.tolist()

            # Cache the result
            self._embedding_cache[cache_key] = result
            return result

        except Exception as e:
            print(f"Error generating query embedding: {e}")
            dim = (
                len(self.preprocessor.embeddings[0])
                if getattr(self.preprocessor, "embeddings", None)
                else 1536
            )
            return [0.0] * dim

    async def embed_queries_batch_async(self, queries: list[str]) -> list[list[float]]:
        """
        Batch embed multiple queries in a single API call.

        Uses sync client via thread pool for better reliability in nested async contexts.
        """
        if not queries:
            return []

        model_name = getattr(self.preprocessor, "embedding_model", "text-embedding-3-small")
        dim = 1536 if "3-small" in model_name else 3072

        # Check cache and identify queries that need embedding
        embeddings = [None] * len(queries)
        uncached_indices = []
        uncached_queries = []

        for i, query in enumerate(queries):
            cache_key = self._hash_query(query)
            if cache_key in self._embedding_cache:
                embeddings[i] = self._embedding_cache[cache_key]
            else:
                uncached_indices.append(i)
                uncached_queries.append(query)

        # If all cached, return immediately
        if not uncached_queries:
            return embeddings

        # Use sync client via thread pool for reliability in complex async contexts
        max_retries = 5
        base_delay = 1.0

        def _sync_embed():
            """Sync embedding with retry logic."""
            import time

            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        input=uncached_queries, model=model_name
                    )
                    return response
                except Exception as e:
                    error_str = str(e).lower()
                    is_retryable = any(
                        x in error_str
                        for x in ["429", "rate limit", "connection", "timeout", "network"]
                    )

                    if is_retryable and attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        time.sleep(delay)
                    else:
                        raise

        try:
            response = await asyncio.to_thread(_sync_embed)

            # Process and cache results
            for j, data in enumerate(response.data):
                emb = np.array(data.embedding, dtype=np.float32)
                faiss.normalize_L2(emb.reshape(1, -1))
                emb_list = emb.tolist()

                original_idx = uncached_indices[j]
                embeddings[original_idx] = emb_list

                # Cache
                cache_key = self._hash_query(uncached_queries[j])
                self._embedding_cache[cache_key] = emb_list

            return embeddings

        except Exception as e:
            print(f"Error batch embedding queries: {e}")
            # Fill with zero vectors for failed queries
            for j in range(len(uncached_queries)):
                original_idx = uncached_indices[j]
                if embeddings[original_idx] is None:
                    embeddings[original_idx] = [0.0] * dim
            return embeddings

    def _retrieve_bm25(self, query: str, k: int) -> list[dict[str, Any]]:
        """Retrieve using BM25"""
        if self.preprocessor.bm25_index is None:
            return []

        query_tokens = self.preprocessor.tokenize_text(query)
        bm25_scores = self.preprocessor.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(bm25_scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_indices):
            if idx < len(self.preprocessor.chunks) and bm25_scores[idx] > 0:
                chunk_text = (
                    self.preprocessor.contextualized_chunks[idx]
                    if getattr(self.preprocessor, "contextualized_chunks", None)
                    else self.preprocessor.chunks[idx]
                )
                results.append(
                    {
                        "rank": rank + 1,
                        "chunk_text": chunk_text,
                        "bm25_score": float(bm25_scores[idx]),
                        "chunk_index": int(idx),
                        "chunk_metadata": self.preprocessor.chunk_metadata[idx],
                        "retrieval_method": "bm25",
                    }
                )
        return results

    def _retrieve_dense(self, query: str, k: int) -> list[dict[str, Any]]:
        """Retrieve using dense embeddings (FAISS)"""
        if self.preprocessor.faiss_index is None:
            return []

        query_embedding = self.embed_query(query)
        query_vector = np.array([query_embedding]).astype(np.float32)
        faiss.normalize_L2(query_vector)

        similarities, indices = self.preprocessor.faiss_index.search(query_vector, k)
        sims, indices = similarities[0], indices[0]

        results = []
        for rank, (sim, idx) in enumerate(zip(sims, indices)):
            if idx < len(self.preprocessor.chunks):
                chunk_text = (
                    self.preprocessor.contextualized_chunks[idx]
                    if getattr(self.preprocessor, "contextualized_chunks", None)
                    else self.preprocessor.chunks[idx]
                )
                results.append(
                    {
                        "rank": rank + 1,
                        "chunk_text": chunk_text,
                        "dense_score": float(sim),
                        "chunk_index": int(idx),
                        "chunk_metadata": self.preprocessor.chunk_metadata[idx],
                        "retrieval_method": "dense",
                    }
                )
        return results

    async def _retrieve_dense_batch_async(
        self, queries: list[str], k: int
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Batch retrieve using dense embeddings for multiple queries (OPTIMIZED).

        Uses single API call to embed all queries, then performs FAISS searches.
        """
        if self.preprocessor.faiss_index is None or not queries:
            return {q: [] for q in queries}

        # Batch embed all queries in one API call
        embeddings = await self.embed_queries_batch_async(queries)

        # Validate embeddings before FAISS search (prevent segfault)
        if not embeddings or any(e is None for e in embeddings):
            print("Warning: Some embeddings are None, falling back to sequential retrieval")
            return {q: [] for q in queries}

        # Stack embeddings for batch search
        def _safe_search():
            try:
                query_vectors = np.array(embeddings, dtype=np.float32)

                # Validate shape before FAISS operations
                if query_vectors.ndim != 2 or query_vectors.shape[0] != len(queries):
                    print(
                        f"Warning: Invalid embedding shape {query_vectors.shape}, expected ({len(queries)}, dim)"
                    )
                    return None, None

                faiss.normalize_L2(query_vectors)

                return self.preprocessor.faiss_index.search(query_vectors, k)

            except Exception as e:
                print(f"Error preparing query vectors: {e}")
                return None, None

        # FAISS batch search with proper error handling
        try:
            result = await asyncio.to_thread(_safe_search)
            if result is None or result[0] is None:
                return {q: [] for q in queries}
            similarities, indices = result
        except Exception as e:
            print(f"FAISS search error: {e}")
            return {q: [] for q in queries}

        # Build results per query
        results: dict[str, list[dict[str, Any]]] = {}
        for q_idx, query in enumerate(queries):
            query_results = []
            for rank, (sim, idx) in enumerate(zip(similarities[q_idx], indices[q_idx])):
                if idx >= 0 and idx < len(self.preprocessor.chunks):
                    chunk_text = (
                        self.preprocessor.contextualized_chunks[idx]
                        if getattr(self.preprocessor, "contextualized_chunks", None)
                        else self.preprocessor.chunks[idx]
                    )
                    query_results.append(
                        {
                            "rank": rank + 1,
                            "chunk_text": chunk_text,
                            "dense_score": float(sim),
                            "chunk_index": int(idx),
                            "chunk_metadata": self.preprocessor.chunk_metadata[idx],
                            "retrieval_method": "dense",
                        }
                    )
            results[query] = query_results

        return results

    def _rrf_fusion(
        self,
        bm25_results: list[dict[str, Any]],
        dense_results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Fuse BM25 and Dense results using weighted Reciprocal Rank Fusion.

        wRRF score = (1 - w) / (k + rank_bm25) + w / (k + rank_dense)
        """
        bm25_ranks = {item["chunk_index"]: rank for rank, item in enumerate(bm25_results)}
        dense_ranks = {item["chunk_index"]: rank for rank, item in enumerate(dense_results)}

        candidate_indices = set(bm25_ranks) | set(dense_ranks)
        if not candidate_indices:
            return []

        # Build candidate map
        candidate_map: dict[int, dict[str, Any]] = {}
        for source in (bm25_results, dense_results):
            for item in source:
                idx = item["chunk_index"]
                if idx not in candidate_map:
                    candidate_map[idx] = dict(item)
                else:
                    for key, value in item.items():
                        candidate_map[idx].setdefault(key, value)

        # Calculate wRRF scores
        fusion_scores: dict[int, float] = {}
        for idx in candidate_indices:
            score = 0.0
            if idx in bm25_ranks:
                score += (1.0 - self.dense_weight) / (self.rrf_k + bm25_ranks[idx] + 1)
            if idx in dense_ranks:
                score += self.dense_weight / (self.rrf_k + dense_ranks[idx] + 1)
            fusion_scores[idx] = score

        # Normalize scores
        max_score = max(fusion_scores.values()) if fusion_scores else 1.0

        # Build final results
        fused: list[dict[str, Any]] = []
        for idx, raw_score in fusion_scores.items():
            candidate = dict(candidate_map[idx])
            candidate["fusion_score"] = raw_score / max_score if max_score > 0 else 0
            candidate["retrieval_method"] = "hybrid_wrrf"
            candidate["in_bm25"] = idx in bm25_ranks
            candidate["in_dense"] = idx in dense_ranks
            fused.append(candidate)

        fused.sort(key=lambda x: x["fusion_score"], reverse=True)
        return fused[:top_k]

    async def run(self, input: RetrievalInput) -> RetrievalOutput:
        """
        Execute hybrid retrieval for the given queries with batch optimization.

        Args:
            input: RetrievalInput with queries and top_k

        Returns:
            RetrievalOutput with retrieved documents
        """
        try:
            queries = input.queries

            # BM25 retrieval (run in parallel threads for each query)
            bm25_tasks = [asyncio.to_thread(self._retrieve_bm25, q, input.top_k) for q in queries]
            bm25_results_list = await asyncio.gather(*bm25_tasks)
            all_bm25_results = [r for results in bm25_results_list for r in results]

            # Dense retrieval (batch API call for all queries at once)
            dense_results_map = await self._retrieve_dense_batch_async(queries, input.top_k)
            all_dense_results = [r for q in queries for r in dense_results_map.get(q, [])]

            # Deduplicate by chunk_index (keep highest scores)
            seen_bm25 = {}
            for r in all_bm25_results:
                idx = r["chunk_index"]
                if idx not in seen_bm25 or r.get("bm25_score", 0) > seen_bm25[idx].get(
                    "bm25_score", 0
                ):
                    seen_bm25[idx] = r

            seen_dense = {}
            for r in all_dense_results:
                idx = r["chunk_index"]
                if idx not in seen_dense or r.get("dense_score", 0) > seen_dense[idx].get(
                    "dense_score", 0
                ):
                    seen_dense[idx] = r

            # Re-rank deduplicated results
            dedup_bm25 = sorted(
                seen_bm25.values(), key=lambda x: x.get("bm25_score", 0), reverse=True
            )
            dedup_dense = sorted(
                seen_dense.values(), key=lambda x: x.get("dense_score", 0), reverse=True
            )

            # Fuse results
            fused = self._rrf_fusion(dedup_bm25, dedup_dense, input.top_k)

            # Extract doc_ids for document-level evaluation
            retrieved_doc_ids = []
            for d in fused:
                doc_id = d.get("chunk_metadata", {}).get("doc_id")
                if doc_id and doc_id not in retrieved_doc_ids:
                    retrieved_doc_ids.append(doc_id)

            return RetrievalOutput(
                status="success",
                documents=fused,
                document_texts=[d["chunk_text"] for d in fused],
                chunk_indices=[d["chunk_index"] for d in fused],
                retrieved_doc_ids=retrieved_doc_ids,
                metadata={
                    "num_queries": len(input.queries),
                    "num_retrieved": len(fused),
                    "num_unique_docs": len(retrieved_doc_ids),
                    "dense_weight": self.dense_weight,
                    "rrf_k": self.rrf_k,
                },
            )

        except Exception as e:
            return RetrievalOutput(
                status="error",
                error_message=str(e),
                documents=[],
                document_texts=[],
                chunk_indices=[],
            )

    async def find_optimal_weight(
        self,
        queries: list[dict[str, Any]],
        top_k: int,
        weight_candidates: list[float] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """
        Find the optimal global RRF weight across the entire dataset.

        Tests each weight candidate on all queries and returns the weight
        that maximizes average F1 score.

        Args:
            queries: List of query dicts with 'query' and 'query_id' fields
            top_k: Number of results to retrieve per query
            weight_candidates: Weights to test (default: 0.0 to 1.0 in 0.1 steps)

        Returns:
            Tuple of (best_weight, scores_per_weight)
        """
        if weight_candidates is None:
            weight_candidates = [round(w * 0.1, 1) for w in range(11)]  # 0.0, 0.1, ..., 1.0

        print(f"\n  Finding optimal RRF weight across {len(queries)} queries...")
        print(f"  Testing weights: {weight_candidates}")

        # Step 1: Get BM25 and Dense results for all queries (batch optimized)
        print("  Step 1: Retrieving BM25 and Dense candidates...")

        query_texts = [q["query"] for q in queries]
        query_ids = [q["query_id"] for q in queries]

        # BM25 retrieval (parallel threads)
        bm25_tasks = [asyncio.to_thread(self._retrieve_bm25, q, top_k) for q in query_texts]
        bm25_results_list = await asyncio.gather(*bm25_tasks)

        # Dense retrieval (single batch API call)
        dense_results_map = await self._retrieve_dense_batch_async(query_texts, top_k)

        # Build per-query results
        query_results = {}
        for i, (q_text, q_id) in enumerate(zip(query_texts, query_ids)):
            query_results[q_id] = {
                "query": q_text,
                "bm25": bm25_results_list[i],
                "dense": dense_results_map.get(q_text, []),
            }

        # Step 2: Test each weight candidate
        print("  Step 2: Evaluating weight candidates...")

        scores_per_weight: dict[float, dict[str, float]] = {}

        for weight in tqdm(weight_candidates, desc="  Testing weights"):
            all_precision = []
            all_recall = []
            all_f1 = []

            for q_id, results in query_results.items():
                # Skip if no relevance labels
                if q_id not in self.relevance_labels:
                    continue

                relevant_indices = set(self.relevance_labels[q_id])
                if not relevant_indices:
                    continue

                # Fuse with this weight
                fused = self._rrf_fusion_with_weight(
                    results["bm25"], results["dense"], top_k, weight
                )

                # Calculate metrics
                retrieved_indices = {c["chunk_index"] for c in fused}
                hits = retrieved_indices & relevant_indices

                precision = len(hits) / len(retrieved_indices) if retrieved_indices else 0.0
                recall = len(hits) / len(relevant_indices) if relevant_indices else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                all_precision.append(precision)
                all_recall.append(recall)
                all_f1.append(f1)

            # Average across all queries
            scores_per_weight[weight] = {
                "precision": sum(all_precision) / len(all_precision) if all_precision else 0.0,
                "recall": sum(all_recall) / len(all_recall) if all_recall else 0.0,
                "f1": sum(all_f1) / len(all_f1) if all_f1 else 0.0,
                "n_queries": len(all_f1),
            }

        # Step 3: Find best weight
        best_weight = max(scores_per_weight.keys(), key=lambda w: scores_per_weight[w]["f1"])
        best_scores = scores_per_weight[best_weight]

        print("\n  Optimal RRF Weight Results:")
        print(f"  {'Weight':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print(f"  {'-' * 44}")
        for w in weight_candidates:
            s = scores_per_weight[w]
            marker = " <-- BEST" if w == best_weight else ""
            print(
                f"  {w:<8.1f} {s['precision']:<12.4f} {s['recall']:<12.4f} {s['f1']:<12.4f}{marker}"
            )

        print(f"\n  ✓ Best weight: {best_weight} (F1: {best_scores['f1']:.4f})")

        return best_weight, scores_per_weight

    async def optimize_fusion_params(
        self,
        query: str,
        bm25_results: list[dict[str, Any]],
        dense_results: list[dict[str, Any]],
        top_k: int,
        weight_candidates: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find optimal RRF weight for a single query using grid search.
        (For per-query optimization - use find_optimal_weight for global optimization)
        """
        if weight_candidates is None:
            weight_candidates = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

        if not weight_candidates:
            raise ValueError("Must provide at least one weight candidate")

        query_id = self.query_id_map.get(self._hash_query(query))

        if query_id is None or query_id not in self.relevance_labels:
            return self._rrf_fusion_with_weight(bm25_results, dense_results, top_k, 0.5)

        relevant_indices = set(self.relevance_labels[query_id])

        async def test_weight(weight: float) -> tuple[float, list[dict[str, Any]], float]:
            fused = self._rrf_fusion_with_weight(bm25_results, dense_results, top_k, weight)
            retrieved_indices = {c["chunk_index"] for c in fused}
            hits = retrieved_indices & relevant_indices

            precision = len(hits) / len(retrieved_indices) if retrieved_indices else 0.0
            recall = len(hits) / len(relevant_indices) if relevant_indices else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            return (weight, fused, f1)

        tasks = [test_weight(w) for w in weight_candidates]
        results = await asyncio.gather(*tasks)

        best_weight, best_fused, best_f1 = max(results, key=lambda x: x[2])

        return best_fused

    def _rrf_fusion_with_weight(
        self,
        bm25_results: list[dict[str, Any]],
        dense_results: list[dict[str, Any]],
        top_k: int,
        weight: float,
    ) -> list[dict[str, Any]]:
        """RRF fusion with a specific weight (for weight optimization)."""
        original_weight = self.dense_weight
        self.dense_weight = weight
        result = self._rrf_fusion(bm25_results, dense_results, top_k)
        self.dense_weight = original_weight
        return result

    def set_dense_weight(self, weight: float) -> None:
        """Set the dense weight for wRRF fusion (0.0 = all BM25, 1.0 = all dense)"""
        self.dense_weight = max(0.0, min(1.0, weight))

    def clear_embedding_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()

    def __repr__(self) -> str:
        return f"HybridRetriever(rrf_k={self.rrf_k}, dense_weight={self.dense_weight})"
