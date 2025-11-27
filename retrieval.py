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

                retrieved_chunks.append(chunk_data)
        
        return retrieved_chunks


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
        
    
