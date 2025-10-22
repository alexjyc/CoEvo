"""
Retrieval Module for RAG Pipeline
Handles similarity search and context retrieval from FAISS index
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import faiss
from openai import OpenAI
from preprocessor import DocumentPreprocessor
from enum import Enum
from rank_bm25 import BM25Okapi
import re
from collections import defaultdict


class RetrievalMethod(Enum):
    """Enumeration of available retrieval methods"""
    BM25_ONLY = "bm25"
    DENSE_ONLY = "dense"
    HYBRID = "hybrid"


class Retriever:
    """
    Handles basic retrieval of relevant document chunks using FAISS similarity search
    """
    
    def __init__(self, preprocessor: DocumentPreprocessor, openai_api_key: str):
        """
        Initialize the retriever with a preprocessor containing the FAISS index
        
        Args:
            preprocessor: DocumentPreprocessor instance with loaded index
            openai_api_key: OpenAI API key for query embedding generation
        """
        self.preprocessor = preprocessor
        self.client = OpenAI(api_key=openai_api_key)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query string
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding vector
        """
        try:
            response = self.client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return [0.0] * 1536  # Return zero vector as fallback

    def retrieve_bm25(self, query: str, max_candidates: int = 20) -> List[Dict[str, Any]]:
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

                if getattr(self.preprocessor, 'contextualized_chunks', None):
                    chunk_data['contextualized_chunk'] = self.preprocessor.contextualized_chunks[idx]
                    chunk_data['has_context'] = self.preprocessor.chunks[idx] != self.preprocessor.contextualized_chunks[idx]
                retrieved_chunks.append(chunk_data)
        
        return retrieved_chunks
    
    def retrieve_dense(self, query: str, max_candidates: int = 20) -> List[Dict[str, Any]]:
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

                if getattr(self.preprocessor, 'contextualized_chunks', None):
                    chunk_data['contextualized_chunk'] = self.preprocessor.contextualized_chunks[idx]
                    chunk_data['has_context'] = self.preprocessor.chunks[idx] != self.preprocessor.contextualized_chunks[idx]
                retrieved_chunks.append(chunk_data)
        
        return retrieved_chunks

    def retrieve_hybrid(self, query: str,
                        k: int = 20, 
                        rrf_k: int = 50,
                        max_candidates: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve using hybrid approach (BM25 + Dense)
        
        Args:
            query: Query string
            k: Final number of chunks to return
            rrf_k: Number of candidates to retrieve from BM25 and dense
            bm25_k: Number of candidates to retrieve from BM25
            
        Returns:
            List of retrieved chunks with hybrid scores
        """
        # Get candidates from both methods
        bm25_results = self.retrieve_bm25(query, max_candidates)
        bm25_scores = np.zeros(len(self.preprocessor.chunks))
        bm25_ranks = np.full(len(self.preprocessor.chunks), np.inf)
        for rank, item in enumerate(bm25_results):
            idx = item['chunk_index']
            bm25_scores[idx] = item['bm25_score']
            bm25_ranks[idx] = rank

        dense_results = self.retrieve_dense(query, max_candidates)
        dense_scores = np.zeros(len(self.preprocessor.chunks))
        dense_ranks = np.full(len(self.preprocessor.chunks), np.inf)
        for rank, item in enumerate(dense_results):
            idx = item['chunk_index']
            dense_scores[idx] = item['dense_score']
            dense_ranks[idx] = rank
        
        rrf_scores = np.zeros(len(self.preprocessor.chunks))
        for idx in range(len(self.preprocessor.chunks)):
            if bm25_ranks[idx] < np.inf:
                rrf_scores[idx] += 1 / (rrf_k + bm25_ranks[idx] + 1)
            if dense_ranks[idx] < np.inf:
                rrf_scores[idx] += 1 / (rrf_k + dense_ranks[idx] + 1)

        top_indices = np.argsort(-rrf_scores)[:k]

        results = []
        for rank, idx in enumerate(top_indices):
            if rrf_scores[idx] == 0:
                continue
            # max_rrf = max(rrf_scores[rrf_scores > 0])
            rrf_score_normalized = rrf_scores[idx]
            chunk_data = {
                'rank': rank + 1,
                'chunk_text': (self.preprocessor.contextualized_chunks[idx] 
               if getattr(self.preprocessor, 'contextualized_chunks', None) 
               else self.preprocessor.chunks[idx]),
                'similarity_score': rrf_score_normalized,
                'bm25_score': float(bm25_scores[idx]),
                'dense_score': float(dense_scores[idx]),
                'chunk_metadata': self.preprocessor.chunk_metadata[idx],
                'chunk_index': int(idx),
                'retrieval_method': "hybrid_rrf",
                'contextual_retrieval_used': getattr(self.preprocessor, 'use_contextual_retrieval', False)
            }
            if getattr(self.preprocessor, 'contextualized_chunks', None):
                chunk_data['contextualized_chunk'] = self.preprocessor.contextualized_chunks[idx]
                chunk_data['has_context'] = self.preprocessor.chunks[idx] != self.preprocessor.contextualized_chunks[idx]
            results.append(chunk_data)
        
        return results
    
    def retrieve(self, query: str, k: int = 20, 
                method: Optional[RetrievalMethod] = None,
                **kwargs) -> List[Dict[str, Any]]:
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
            return self.retrieve_hybrid(query, k, **kwargs)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")