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

        # BM25 components
        self.bm25_corpus = None
        self.bm25_index = None

        if self.preprocessor.chunks:
            self._build_bm25_index()

    def _build_bm25_index(self):
        """Build BM25 index from preprocessed chunks"""
        print("Building BM25 index...")
        
        # Tokenize all chunks for BM25
        self.bm25_corpus = []
        for chunk in self.preprocessor.chunks:
            tokens = self._tokenize_text(chunk)
            self.bm25_corpus.append(tokens)
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        print(f"BM25 index built with {len(self.bm25_corpus)} documents")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split on whitespace and punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        return tokens
    
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
        if self.bm25_index is None:
            raise ValueError("BM25 index not built. Please build the index first.")

        query_tokens = self._tokenize_text(query)

        bm25_scores = self.bm25_index.get_scores(query_tokens)

        top_indices = np.argsort(bm25_scores)[::-1][:max_candidates]

        retrieved_chunks = []

        for i, idx in enumerate(top_indices):
            if idx < len(self.preprocessor.chunks) and bm25_scores[idx] > 0:
                chunk_data = {
                    'rank': i + 1,
                    'chunk_text': self.preprocessor.chunks[idx],
                    'similarity_score': float(bm25_scores[idx]),
                    'bm25_score': float(bm25_scores[idx]),
                    'dense_score': 0.0,  # No dense score for BM25-only
                    'chunk_metadata': self.preprocessor.chunk_metadata[idx],
                    'chunk_index': int(idx),
                    'retrieval_method': 'bm25_only'
                }
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
        
        # Prepare results
        retrieved_chunks = []
        
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.preprocessor.chunks):
                chunk_data = {
                    'rank': i + 1,
                    'chunk_text': self.preprocessor.chunks[idx],
                    'similarity_score': float(similarity),
                    'dense_score': float(similarity),
                    'bm25_score': 0.0,  # No BM25 score for dense-only
                    'chunk_metadata': self.preprocessor.chunk_metadata[idx],
                    'chunk_index': int(idx),
                    'retrieval_method': 'dense_only'
                }
                retrieved_chunks.append(chunk_data)
        
        return retrieved_chunks

    def retrieve_hybrid(self, query: str, k: int = 20, 
                       bm25_weight: float = 0.3, 
                       dense_weight: float = 0.7,
                       bm25_k: int = 50,
                       dense_k: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve using hybrid approach (BM25 + Dense)
        
        Args:
            query: Query string
            k: Final number of chunks to return
            bm25_weight: Weight for BM25 scores (0.0 to 1.0)
            dense_weight: Weight for dense scores (0.0 to 1.0)
            bm25_k: Number of candidates to retrieve from BM25
            dense_k: Number of candidates to retrieve from dense
            
        Returns:
            List of retrieved chunks with hybrid scores
        """
        # Get candidates from both methods
        bm25_results = self.retrieve_bm25(query, bm25_k)
        dense_results = self.retrieve_dense(query, dense_k)
        
        # Normalize scores to [0, 1] range
        bm25_scores_norm = self._normalize_scores([r['bm25_score'] for r in bm25_results])
        dense_scores_norm = self._normalize_scores([r['dense_score'] for r in dense_results])
        
        # Update normalized scores
        for i, result in enumerate(bm25_results):
            result['bm25_score_norm'] = bm25_scores_norm[i] if i < len(bm25_scores_norm) else 0.0
        
        for i, result in enumerate(dense_results):
            result['dense_score_norm'] = dense_scores_norm[i] if i < len(dense_scores_norm) else 0.0
        
        # Combine results and calculate hybrid scores
        chunk_scores = defaultdict(lambda: {'bm25_score': 0.0, 'dense_score': 0.0, 'bm25_score_norm': 0.0, 'dense_score_norm': 0.0, 'chunk_data': None})
        
        # Add BM25 results
        for result in bm25_results:
            idx = result['chunk_index']
            chunk_scores[idx]['bm25_score'] = result['bm25_score']
            chunk_scores[idx]['bm25_score_norm'] = result.get('bm25_score_norm', 0.0)
            chunk_scores[idx]['chunk_data'] = result
        
        # Add dense results
        for result in dense_results:
            idx = result['chunk_index']
            chunk_scores[idx]['dense_score'] = result['dense_score']
            chunk_scores[idx]['dense_score_norm'] = result.get('dense_score_norm', 0.0)
            if chunk_scores[idx]['chunk_data'] is None:
                chunk_scores[idx]['chunk_data'] = result
        
        # Calculate hybrid scores and create final results
        hybrid_results = []
        
        for idx, scores in chunk_scores.items():
            # Calculate hybrid score using normalized scores
            hybrid_score = (bm25_weight * scores['bm25_score_norm'] + 
                          dense_weight * scores['dense_score_norm'])
            
            chunk_data = scores['chunk_data'].copy()
            chunk_data.update({
                'similarity_score': hybrid_score,
                'hybrid_score': hybrid_score,
                'bm25_score': scores['bm25_score'],
                'dense_score': scores['dense_score'],
                'bm25_score_norm': scores['bm25_score_norm'],
                'dense_score_norm': scores['dense_score_norm'],
                'bm25_weight': bm25_weight,
                'dense_weight': dense_weight,
                'retrieval_method': 'hybrid'
            })
            
            hybrid_results.append(chunk_data)
        
        # Sort by hybrid score and return top k
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(hybrid_results[:k]):
            result['rank'] = i + 1
        
        return hybrid_results[:k]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization
        
        Args:
            scores: List of scores to normalize
            
        Returns:
            List of normalized scores
        """
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)  # All scores are the same
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
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
        method = method or self.default_method
        
        print(f"Retrieving with method: {method.value}")
        
        if method == RetrievalMethod.BM25_ONLY:
            return self.retrieve_bm25(query, k)
        elif method == RetrievalMethod.DENSE_ONLY:
            return self.retrieve_dense(query, k)
        elif method == RetrievalMethod.HYBRID:
            return self.retrieve_hybrid(query, k, **kwargs)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")