from typing import Callable, List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
from sentence_transformers import CrossEncoder


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

    def optimize_fusion_params(
            self, 
            bm25_results: List[Dict[str, Any]],
            dense_results: List[Dict[str, Any]],
            top_k: int,
            quality_fn: Callable[[List[Dict[str, Any]]], float], 
            weight_candidates: List[float] = None
    ) -> List[Dict[str, Any]]:
            
            if weight_candidates is None:
                weight_candidates = [0.3, 0.5, 0.7, 0.9]
            
            if not weight_candidates:
                raise ValueError("Must provide at least one weight candidate")
            
            def _test_single_weight(weight: float) -> Tuple[float, List[Dict[str, Any]], float]:
                """
                Test a single weight and return (weight, chunks, score).
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
                    
                    # Evaluate quality
                    quality_metrics = quality_fn(fused_chunks)
                    
                    # Extract primary score (use first numeric value)
                    score = None
                    if isinstance(quality_metrics, dict):
                        # Find first numeric metric
                        for key, value in quality_metrics.items():
                            if isinstance(value, (int, float)):
                                score = float(value)
                                break
                        if score is None:
                            raise ValueError(
                                f"quality_fn must return dict with at least one numeric value. "
                                f"Got: {quality_metrics}"
                            )
                    elif isinstance(quality_metrics, (int, float)):
                        score = float(quality_metrics)
                    else:
                        raise ValueError(
                            f"quality_fn must return dict or numeric value. "
                            f"Got: {type(quality_metrics)}"
                        )
                    
                    return weight, fused_chunks, score
                    
                except Exception as e:
                    # Return error indicator
                    print(f"Warning: Error testing weight {weight}: {e}")
                    return weight, [], -1.0
            
            # Execute weight testing (parallel or sequential)
            results = {}
            quality_scores = {}
            
            if len(weight_candidates) > 1:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_weight = {
                        executor.submit(_test_single_weight, weight): weight
                        for weight in weight_candidates
                    }
                    
                    with tqdm(total=len(weight_candidates), desc="Testing weights (parallel)") as pbar:
                        for future in as_completed(future_to_weight):
                            weight, chunks, score = future.result()
                            if score >= 0:  # Valid result
                                results[weight] = chunks
                                quality_scores[weight] = score
                            pbar.update(1)
            else:
                with tqdm(total=len(weight_candidates), desc="Testing weights (sequential)") as pbar:
                    for weight in weight_candidates:
                        weight, chunks, score = _test_single_weight(weight)
                        if score >= 0:
                            results[weight] = chunks
                            quality_scores[weight] = score
                        pbar.update(1)

            if not quality_scores:
                raise RuntimeError("All weight tests failed. Check quality_fn and inputs.")
            
            # Select best weight
            best_weight = max(quality_scores, key=quality_scores.get)
            best_chunks = results[best_weight]
            
            # Compute statistics
            # scores_array = np.array(list(quality_scores.values()))
            # details = {
            #     "best_weight": float(best_weight),
            #     "best_score": float(quality_scores[best_weight]),
            #     "all_scores": {float(w): float(s) for w, s in quality_scores.items()},
            #     "weight_stats": {
            #         "mean": float(np.mean(scores_array)),
            #         "std": float(np.std(scores_array)),
            #         "min": float(np.min(scores_array)),
            #         "max": float(np.max(scores_array)),
            #         "range": float(np.ptp(scores_array))
            #     },
            #     "num_weights_tested": len(weight_candidates),
            #     "execution_time": round(execution_time, 4),
            # }
            
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
