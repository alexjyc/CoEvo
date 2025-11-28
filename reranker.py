import hashlib
from typing import Callable, List, Dict, Any, Optional, Tuple
import asyncio
import numpy as np
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from evaluation import Evaluator


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        default_k: int = 10,
        device: str = "mps",
        batch_size: int = 16,
        fusion_rrf_k: int = 60,
    ) -> None:
        self.model_name = model_name
        self.default_k = default_k
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device)
        self.fusion_rrf_k = fusion_rrf_k
        self.query_id_map = {}
        self.relevance_labels = {}
        self.evaluator = Evaluator()


    def set_relevance_labels(self, evaluation_queries: List[Dict[str, Any]], relevance_labels: Dict[int, List[int]]) -> None:
        self.relevance_labels = relevance_labels
        self.query_id_map = { self._hash_query(q['query']): q['query_id'] for q in evaluation_queries }

    @staticmethod
    def _hash_query(query: str) -> str:
        normalized = ' '.join(query.lower().strip().split())

        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    async def optimize_fusion_params(
            self, 
            query: str,
            bm25_results: List[Dict[str, Any]],
            dense_results: List[Dict[str, Any]],
            top_k: int,
            weight_candidates: List[float] = None
    ) -> List[Dict[str, Any]]:
            
            if weight_candidates is None:
                weight_candidates = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
            
            if not weight_candidates:
                raise ValueError("Must provide at least one weight candidate")

            query_id = self.query_id_map.get(self._hash_query(query))

            if query_id is None:
                raise ValueError(f"Query ID not found for query: {query}")

            async def _test_single_weight(weight: float) -> Tuple[float, List[Dict[str, Any]], float, float]:
                """
                Test a single weight and return (weight, chunks, precision, recall).
                This function is designed to be called in parallel.
                """
                try:
                    # Fuse with this weight
                    fused_chunks = self.rank_fusion(
                        bm25_results=bm25_results,
                        dense_results=dense_results,
                        top_k=top_k,
                        weight=weight,
                        rrf_k=self.fusion_rrf_k
                    )
                    
                    # Extract contexts from chunks for evaluation
                    contexts = [chunk.get("chunk_index", "") for chunk in fused_chunks]
                    print(f"Contexts: {contexts}")  
                    print(f"Relevance labels: {self.relevance_labels[query_id]}")
                    
                    # Evaluate quality
                    quality_metrics = await self.evaluator.evaluate_retrieval(
                        retrieved_contexts=contexts,
                        reference_contexts=self.relevance_labels[query_id]
                    )

                    precision = quality_metrics.get("retrieval_context_precision", 0.0)
                    recall = quality_metrics.get("retrieval_context_recall", 0.0)

                    return weight, fused_chunks, precision, recall
                    
                except Exception as e:
                    # Return error indicator
                    print(f"Warning: Error testing weight {weight}: {e}")
                    return weight, [], -1.0, -1.0
            
            # Execute weight testing with asyncio
            results = {}
            quality_scores = {}
            
            # Create tasks for all weights
            tasks = [_test_single_weight(weight) for weight in weight_candidates]
            
            # Execute all tasks concurrently with progress bar
            with tqdm(total=len(weight_candidates), desc="Testing weights (async)") as pbar:
                for coro in asyncio.as_completed(tasks):
                    weight, chunks, precision, recall = await coro
                    if precision >= 0 and recall >= 0:  # Valid result
                        # Calculate F1 score from precision and recall
                        if precision + recall > 0:
                            f1_score = 2 * (precision * recall) / (precision + recall)
                        else:
                            f1_score = 0.0
                        results[weight] = chunks
                        quality_scores[weight] = f1_score
                    pbar.update(1)

            if not quality_scores:
                raise RuntimeError("All weight tests failed. Check quality_fn and inputs.")
            
            # Select best weight
            best_weight = max(quality_scores, key=quality_scores.get)
            best_chunks = results[best_weight]
            
            return best_chunks

    def rank_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        top_k: int,
        weight: float,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Fuse BM25 and dense retrieval outputs using Reciprocal Rank Fusion (RRF).
        """
        bm25_ranks = {item["chunk_index"]: rank for rank, item in enumerate(bm25_results)}
        dense_ranks = {item["chunk_index"]: rank for rank, item in enumerate(dense_results)}
        candidate_indices = set(bm25_ranks) | set(dense_ranks)
        if not candidate_indices:
            return []

        def rrf(rank: int, component_weight: float) -> float:
            return component_weight / (rrf_k + rank + 1)

        # Build consolidated candidate map
        candidate_map: Dict[int, Dict[str, Any]] = {}
        for source in (bm25_results, dense_results):
            for item in source:
                idx = item["chunk_index"]
                if idx not in candidate_map:
                    candidate_map[idx] = dict(item)
                else:
                    for key, value in item.items():
                        candidate_map[idx].setdefault(key, value)

        fusion_scores: Dict[int, float] = {}
        for idx in candidate_indices:
            score = 0.0
            if idx in bm25_ranks:
                score += rrf(bm25_ranks[idx], 1.0 - weight)
            if idx in dense_ranks:
                score += rrf(dense_ranks[idx], weight)
            fusion_scores[idx] = score

        max_score = max(fusion_scores.values()) if fusion_scores else 1.0

        fused: List[Dict[str, Any]] = []
        for idx, raw_score in fusion_scores.items():
            candidate = dict(candidate_map[idx])  # copy to avoid mutating source
            candidate["fusion_raw_score"] = raw_score
            candidate["similarity_score"] = (
                raw_score if max_score == 0 else raw_score / max_score
            )
            candidate["retrieval_method"] = "hybrid_rrf"
            candidate["hybrid_sources"] = {
                "bm25": idx in bm25_ranks,
                "dense": idx in dense_ranks,
            }
            if idx in bm25_ranks:
                candidate["bm25_rank"] = bm25_ranks[idx] + 1
            if idx in dense_ranks:
                candidate["dense_rank"] = dense_ranks[idx] + 1
            fused.append(candidate)

        fused.sort(key=lambda item: item["fusion_raw_score"], reverse=True)

        return fused[:top_k]

    def process_chunks(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not retrieved_chunks:
            return []

        k = k or self.default_k
        if k <= 0:
            return []

        pairs = []
        for chunk in retrieved_chunks:
            query = chunk.get("query")
            text = chunk.get("chunk_text")
            pairs.append((query, text))

        scores = self.model.predict(pairs, batch_size=self.batch_size)

        scored_chunks = []
        for chunk, score in zip(retrieved_chunks, scores):
            annotated = dict(chunk)
            annotated["cross_encoder_score"] = float(score)
            scored_chunks.append(annotated)

        scored_chunks.sort(key=lambda item: item["cross_encoder_score"], reverse=True)
        return scored_chunks[:k]
