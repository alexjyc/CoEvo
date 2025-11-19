from typing import List, Dict, Any, Optional

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        default_k: int = 10,
        device: str = "mps",
        batch_size: int = 16,
        fusion_rrf_k: int = 60,
        hybrid_dense_weight: float = 0.7
    ) -> None:
        self.model_name = model_name
        self.default_k = default_k
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device)
        self.fusion_rrf_k = fusion_rrf_k
        self.hybrid_dense_weight = self._sanitize_weight(hybrid_dense_weight)

    @staticmethod
    def _sanitize_weight(weight: float) -> float:
        return max(0.0, min(1.0, float(weight)))

    def rank_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        k: int,
        dense_weight: Optional[float] = None,
        rrf_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fuse BM25 and dense retrieval outputs using Reciprocal Rank Fusion (RRF).
        """
        bm25_results = bm25_results or []
        dense_results = dense_results or []
        if not bm25_results and not dense_results:
            return []

        weight = (
            self._sanitize_weight(dense_weight)
            if dense_weight is not None
            else self.hybrid_dense_weight
        )
        rrf_k = rrf_k or self.fusion_rrf_k
        bm25_weight = 1.0 - weight

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
                score += rrf(bm25_ranks[idx], bm25_weight)
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

        limited = fused[:k]
        for rank, item in enumerate(limited, start=1):
            item["rank"] = rank

        return limited

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
