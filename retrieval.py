"""
Retrieval Module for RAG Pipeline
Handles similarity search and context retrieval from FAISS index
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Union
import faiss
from openai import OpenAI
from preprocessor import DocumentPreprocessor
from enum import Enum


class RetrievalMethod(Enum):
    """Enumeration of available retrieval methods"""
    BM25_ONLY = "bm25"
    DENSE_ONLY = "dense"
    HYBRID = "hybrid"


class Retriever:
    """
    Handles basic retrieval of relevant document chunks using FAISS similarity search
    """
    
    def __init__(self, preprocessor: DocumentPreprocessor):
        """
        Initialize the retriever with a preprocessor containing the FAISS index
        
        Args:
            preprocessor: DocumentPreprocessor instance with loaded index
            openai_api_key: OpenAI API key for query embedding generation
        """
        self.preprocessor = preprocessor
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query string
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding vector
        """
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

    def retrieve_bm25(self, query: str, max_candidates: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve using BM25
        
        Args:
            query: Query string
            max_candidates: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with metadata and similarity scores
        """
        if self.preprocessor.bm25_index is None:
            raise ValueError("BM25 index not loaded. Please load or create an index first.")

        contextual_bm25_used = getattr(self.preprocessor, 'use_contextual_retrieval', False)
        method_name = "contextual_bm25" if contextual_bm25_used else "standard_bm25"

        if contextual_bm25_used:
            print("🎯 Using Contextual BM25")

        query_tokens = self.preprocessor.tokenize_text(query)
        bm25_scores = self.preprocessor.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(bm25_scores)[::-1][:max_candidates]

        retrieved_chunks = []
        for rank, idx in enumerate(top_indices):
            if idx < len(self.preprocessor.chunks) and bm25_scores[idx] > 0:
                chunk_data = {
                    'rank': rank + 1,
                    'query': query,
                    'chunk_text': (self.preprocessor.contextualized_chunks[idx] 
               if getattr(self.preprocessor, 'contextualized_chunks', None) 
               else self.preprocessor.chunks[idx]),
                    'similarity_score': float(bm25_scores[idx]),
                    'bm25_score': float(bm25_scores[idx]),
                    'dense_score': 0.0,  # No dense score for BM25-only
                    'chunk_metadata': self.preprocessor.chunk_metadata[idx],
                    'chunk_index': int(idx),
                    'retrieval_method': method_name,
                    'contextual_retrieval_used': contextual_bm25_used
                }

                # if getattr(self.preprocessor, 'contextualized_chunks', None):
                #     chunk_data['contextualized_chunk'] = self.preprocessor.contextualized_chunks[idx]
                #     chunk_data['has_context'] = self.preprocessor.chunks[idx] != self.preprocessor.contextualized_chunks[idx]
                retrieved_chunks.append(chunk_data)
        
        return retrieved_chunks
    
    def retrieve_dense(self, query: str, max_candidates: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar document chunks for a given query
        
        Args:
            query: Query string
            max_candidates: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with metadata and similarity scores
        """
        if self.preprocessor.faiss_index is None:
            raise ValueError("FAISS index not loaded. Please load or create an index first.")
        
        # Generate query embedding
        query_embedding = self.embed_query(query)
        query_vector = np.array([query_embedding]).astype(np.float32)
        
        # Normalize query vector for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Perform similarity search
        similarities, indices = self.preprocessor.faiss_index.search(query_vector, max_candidates)
        sims, indices = similarities[0], indices[0]

        contextual_embeddings_used = getattr(self.preprocessor, 'use_contextual_retrieval', False)
        method_name = "contextual_dense" if contextual_embeddings_used else "standard_dense"

        if contextual_embeddings_used:
            print("🎯 Using Contextual Dense Embeddings")
        
        # Prepare results
        retrieved_chunks = []
        for rank, (sim, idx) in enumerate(zip(sims, indices)):
            if idx < len(self.preprocessor.chunks):
                chunk_data = {
                    'rank': rank + 1,
                    'query': query,
                    'chunk_text': (self.preprocessor.contextualized_chunks[idx] 
               if getattr(self.preprocessor, 'contextualized_chunks', None) 
               else self.preprocessor.chunks[idx]),
                    'similarity_score': float(sim),
                    'dense_score': float(sim),
                    'bm25_score': 0.0,  # No BM25 score for dense-only
                    'chunk_metadata': self.preprocessor.chunk_metadata[idx],
                    'chunk_index': int(idx),
                    'retrieval_method': method_name,
                    'contextual_retrieval_used': contextual_embeddings_used
                }

                # if getattr(self.preprocessor, 'contextualized_chunks', None):
                #     chunk_data['contextualized_chunk'] = self.preprocessor.contextualized_chunks[idx]
                #     chunk_data['has_context'] = self.preprocessor.chunks[idx] != self.preprocessor.contextualized_chunks[idx]
                retrieved_chunks.append(chunk_data)
        
        return retrieved_chunks

    # def retrieve_hybrid_rrf(self, query: str,
    #                     k: int = 50, 
    #                     rrf_k: int = 60,
    #                     dense_weight: float = 0.7,
    #                     bm25_k: int = 40,
    #                     dense_k: int = 40) -> List[Dict[str, Any]]:
    #     """
    #     Retrieve using hybrid approach (BM25 + Dense)
        
    #     Args:
    #         query: Query string
    #         k: Final number of chunks to return
    #         rrf_k: Number of candidates to retrieve from BM25 and dense
    #         bm25_k: Number of candidates to retrieve from BM25
    #         dense_k: Number of candidates to retrieve from dense
            
    #     Returns:
    #         List of retrieved chunks with hybrid scores
    #     """
    #     bm25_results = self.retrieve_bm25(query, bm25_k)
    #     dense_results = self.retrieve_dense(query, dense_k)
   
    #     bm25_ranks = {item['chunk_index']: rank for rank, item in enumerate(bm25_results)}
    #     dense_ranks = {item['chunk_index']: rank for rank, item in enumerate(dense_results)}
    #     bm25_scores = {item['chunk_index']: item.get('bm25_score', 0.0) for item in bm25_results}
    #     dense_scores = {item['chunk_index']: item.get('dense_score', 0.0) for item in dense_results}

    #     candidate_indices = set(bm25_ranks.keys()) | set(dense_ranks.keys())
    #     if not candidate_indices:
    #         return []
        
    #     dense_weight = float(np.clip(dense_weight, 0.0, 1.0))
    #     bm25_weight = 1.0 - dense_weight
    #     def rrf(rank: int, weight: float) -> float:
    #         return weight / (rrf_k + rank + 1)
        
    #     rrf_scores: Dict[int, float] = {}
    #     for idx in candidate_indices:
    #         score = 0.0
    #         if idx in bm25_ranks:
    #             score += rrf(bm25_ranks[idx], bm25_weight)
    #         if idx in dense_ranks:
    #             score += rrf(dense_ranks[idx], dense_weight)
    #         rrf_scores[idx] = score

    #     max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0

    #     scored_candidates = []
    #     for idx, raw_score in rrf_scores.items():
    #         chunk_data = {
    #             'query': query,
    #             'chunk_index': int(idx),
    #             'chunk_text': (
    #                 self.preprocessor.contextualized_chunks[idx]
    #                 if getattr(self.preprocessor, 'contextualized_chunks', None)
    #                 else self.preprocessor.chunks[idx]
    #             ),
    #             'similarity_score': float(raw_score if max_rrf == 0 else raw_score / max_rrf),
    #             'bm25_score': float(bm25_scores.get(idx, 0.0)),
    #             'dense_score': float(dense_scores.get(idx, 0.0)),
    #             'bm25_rank': int(bm25_ranks.get(idx, -1)),
    #             'dense_rank': int(dense_ranks.get(idx, -1)),
    #             'chunk_metadata': self.preprocessor.chunk_metadata[idx],
    #             'retrieval_method': "hybrid_rrf",
    #             'contextual_retrieval_used': getattr(self.preprocessor, 'use_contextual_retrieval', False)
    #         }

    #         # if getattr(self.preprocessor, 'contextualized_chunks', None):
    #         #     chunk_data['contextualized_chunk'] = self.preprocessor.contextualized_chunks[idx]
    #         #     chunk_data['has_context'] = self.preprocessor.chunks[idx] != self.preprocessor.contextualized_chunks[idx]
    #         scored_candidates.append(chunk_data)

    #     scored_candidates.sort(key=lambda item: item['similarity_score'], reverse=True)
    #     results = []
    #     for rank, item in enumerate(scored_candidates[:k]):
    #         item['rank'] = rank + 1
    #         results.append(item)
        
    #     return results
    
    # def retrieve_hybrid_wf(
    #     self,
    #     query: str,
    #     k: int = 50,
    #     dense_weight: float = 0.7,
    #     bm25_k: int = 40,
    #     dense_k: int = 40
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Retrieve using hybrid approach (BM25 + Dense) with weighted scores
        
    #     Args:
    #         query: Query string
    #         k: Final number of chunks to return
    #         dense_weight: Weight for dense scores
    #         bm25_k: Number of candidates to retrieve from BM25
    #         dense_k: Number of candidates to retrieve from dense
            
    #     Returns:
    #         List of retrieved chunks with hybrid scores
    #     """
    #     bm25_results = self.retrieve_bm25(query, bm25_k)
    #     dense_results = self.retrieve_dense(query, dense_k)

    #     bm25_scores = {item['chunk_index']: item['bm25_score'] for item in bm25_results}
    #     dense_scores = {item['chunk_index']: item['dense_score'] for item in dense_results}
    #     candidate_indices = sorted(set(bm25_scores.keys()) | set(dense_scores.keys()))
    #     if not candidate_indices:
    #         return []
        
    #     def min_max_normalize(score_map: Dict[int, float]) -> Dict[int, float]:
    #         if not score_map:
    #             return {}
    #         values = np.array(list(score_map.values()), dtype=np.float32)
    #         max_val = float(values.max())
    #         min_val = float(values.min())
    #         if np.isclose(max_val, min_val):
    #             return {idx: 1.0 for idx in score_map}
    #         range_val = max_val - min_val
    #         return {idx: (score - min_val) / range_val for idx, score in score_map.items()}
        
    #     bm25_norm = min_max_normalize(bm25_scores)
    #     dense_norm = min_max_normalize(dense_scores)

    #     dense_weight = float(np.clip(dense_weight, 0.0, 1.0))
    #     bm25_weight = 1.0 - dense_weight

    #     scored_candidates = []
    #     for idx in candidate_indices:
    #         hybrid_score = (
    #             dense_weight * dense_norm.get(idx, 0.0)
    #             + bm25_weight * bm25_norm.get(idx, 0.0)
    #         )

    #         if hybrid_score == 0.0 and idx not in dense_scores and idx not in bm25_scores:
    #             continue

    #         chunk_data = {
    #             'query': query,
    #             'chunk_text': (
    #                 self.preprocessor.contextualized_chunks[idx]
    #                 if getattr(self.preprocessor, 'contextualized_chunks', None)
    #                 else self.preprocessor.chunks[idx]
    #             ),
    #             'similarity_score': float(hybrid_score),
    #             'bm25_score': float(bm25_scores.get(idx, 0.0)),
    #             'dense_score': float(dense_scores.get(idx, 0.0)),
    #             'bm25_normalized': float(bm25_norm.get(idx, 0.0)),
    #             'dense_normalized': float(dense_norm.get(idx, 0.0)),
    #             'chunk_metadata': self.preprocessor.chunk_metadata[idx],
    #             'retrieval_method': "hybrid_weighted",
    #             'contextual_retrieval_used': getattr(self.preprocessor, 'use_contextual_retrieval', False)
    #         }

    #         # if getattr(self.preprocessor, 'contextualized_chunks', None):
    #         #     chunk_data['contextualized_chunk'] = self.preprocessor.contextualized_chunks[idx]
    #         #     chunk_data['has_context'] = self.preprocessor.chunks[idx] != self.preprocessor.contextualized_chunks[idx]
    #         scored_candidates.append(chunk_data)

    #     scored_candidates.sort(key=lambda item: item['similarity_score'], reverse=True)
    #     for rank, item in enumerate(scored_candidates):
    #         item['rank'] = rank + 1
    #     return scored_candidates[:k]


    def retrieve_hybrid_components(
        self,
        query: str,
        bm25_k: int = 40,
        dense_k: int = 40
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve BM25 and dense candidates separately for downstream hybrid fusion.
        """
        bm25_results = self.retrieve_bm25(query, bm25_k)
        dense_results = self.retrieve_dense(query, dense_k)

        return {
            "bm25": bm25_results,
            "dense": dense_results,
        }

    def retrieve(self, query: str, k: int = 50, 
                method: Optional[RetrievalMethod] = None,
                **kwargs) -> Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Main retrieval method that dispatches to specific retrieval methods
        
        Args:
            query: Query string
            k: Number of chunks to retrieve
            method: Retrieval method to use (defaults to self.default_method)
            **kwargs: Additional arguments for specific retrieval methods
            
        Returns:
            List of retrieved chunks
        """
        
        print(f"Retrieving with method: {method.value}")
        
        if method == RetrievalMethod.BM25_ONLY:
            return self.retrieve_bm25(query, k)
        elif method == RetrievalMethod.DENSE_ONLY:
            return self.retrieve_dense(query, k)
        elif method == RetrievalMethod.HYBRID:
            bm25_k = kwargs.get("bm25_k", k)
            dense_k = kwargs.get("dense_k", k)
            return self.retrieve_hybrid_components(query, bm25_k=bm25_k, dense_k=dense_k)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        
    
