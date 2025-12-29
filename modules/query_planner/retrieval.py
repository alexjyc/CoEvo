"""
Retrieval Component for Query Planner Module

Handles hybrid retrieval (Dense + BM25) with weighted Reciprocal Rank Fusion (wRRF).
This component works with the Query Planner to form Module 1.

No prompt optimization here - retrieval is algorithmic.
Optimization is done via wRRF weight tuning.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from openai import OpenAI

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
    ):
        super().__init__(ModuleType.RETRIEVAL)
        self.preprocessor = preprocessor
        self.rrf_k = rrf_k
        self.dense_weight = default_dense_weight
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # No prompt for retrieval module
        self._prompt = None

    def get_default_prompt(self) -> str:
        """Retrieval has no prompt - it's algorithmic"""
        return ""

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query string"""
        model_name = getattr(self.preprocessor, "embedding_model", "text-embedding-3-small")

        try:
            response = self.client.embeddings.create(
                input=query,
                model=model_name
            )
            emb = np.array(response.data[0].embedding, dtype=np.float32)
            faiss.normalize_L2(emb.reshape(1, -1))
            return emb.tolist()

        except Exception as e:
            print(f"Error generating query embedding: {e}")
            dim = len(self.preprocessor.embeddings[0]) if getattr(self.preprocessor, "embeddings", None) else 1536
            return [0.0] * dim

    def _retrieve_bm25(self, query: str, k: int) -> List[Dict[str, Any]]:
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
                    if getattr(self.preprocessor, 'contextualized_chunks', None)
                    else self.preprocessor.chunks[idx]
                )
                results.append({
                    'rank': rank + 1,
                    'chunk_text': chunk_text,
                    'bm25_score': float(bm25_scores[idx]),
                    'chunk_index': int(idx),
                    'chunk_metadata': self.preprocessor.chunk_metadata[idx],
                    'retrieval_method': 'bm25',
                })
        return results

    def _retrieve_dense(self, query: str, k: int) -> List[Dict[str, Any]]:
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
                    if getattr(self.preprocessor, 'contextualized_chunks', None)
                    else self.preprocessor.chunks[idx]
                )
                results.append({
                    'rank': rank + 1,
                    'chunk_text': chunk_text,
                    'dense_score': float(sim),
                    'chunk_index': int(idx),
                    'chunk_metadata': self.preprocessor.chunk_metadata[idx],
                    'retrieval_method': 'dense',
                })
        return results

    def _rrf_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
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
        candidate_map: Dict[int, Dict[str, Any]] = {}
        for source in (bm25_results, dense_results):
            for item in source:
                idx = item["chunk_index"]
                if idx not in candidate_map:
                    candidate_map[idx] = dict(item)
                else:
                    for key, value in item.items():
                        candidate_map[idx].setdefault(key, value)

        # Calculate wRRF scores
        fusion_scores: Dict[int, float] = {}
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
        fused: List[Dict[str, Any]] = []
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
        Execute hybrid retrieval for the given queries.

        Args:
            input: RetrievalInput with queries and top_k

        Returns:
            RetrievalOutput with retrieved documents
        """
        try:
            all_bm25_results = []
            all_dense_results = []

            # Retrieve for each query (from query planner)
            for query in input.queries:
                bm25_results = await asyncio.to_thread(
                    self._retrieve_bm25, query, input.top_k
                )
                dense_results = await asyncio.to_thread(
                    self._retrieve_dense, query, input.top_k
                )
                all_bm25_results.extend(bm25_results)
                all_dense_results.extend(dense_results)

            # Deduplicate by chunk_index
            seen_bm25 = {}
            for r in all_bm25_results:
                idx = r["chunk_index"]
                if idx not in seen_bm25 or r.get("bm25_score", 0) > seen_bm25[idx].get("bm25_score", 0):
                    seen_bm25[idx] = r

            seen_dense = {}
            for r in all_dense_results:
                idx = r["chunk_index"]
                if idx not in seen_dense or r.get("dense_score", 0) > seen_dense[idx].get("dense_score", 0):
                    seen_dense[idx] = r

            # Re-rank deduplicated results
            dedup_bm25 = sorted(seen_bm25.values(), key=lambda x: x.get("bm25_score", 0), reverse=True)
            dedup_dense = sorted(seen_dense.values(), key=lambda x: x.get("dense_score", 0), reverse=True)

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
                retrieved_doc_ids=retrieved_doc_ids,  # For document-level evaluation
                metadata={
                    "num_queries": len(input.queries),
                    "num_retrieved": len(fused),
                    "num_unique_docs": len(retrieved_doc_ids),
                    "dense_weight": self.dense_weight,
                    "rrf_k": self.rrf_k,
                }
            )

        except Exception as e:
            return RetrievalOutput(
                status="error",
                error_message=str(e),
                documents=[],
                document_texts=[],
                chunk_indices=[],
            )

    def set_dense_weight(self, weight: float) -> None:
        """Set the dense weight for wRRF fusion (0.0 = all BM25, 1.0 = all dense)"""
        self.dense_weight = max(0.0, min(1.0, weight))

    def __repr__(self) -> str:
        return f"HybridRetriever(rrf_k={self.rrf_k}, dense_weight={self.dense_weight})"
