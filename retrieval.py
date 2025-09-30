"""
Retrieval Module for RAG Pipeline
Handles similarity search and context retrieval from FAISS index
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import faiss
from openai import OpenAI
from preprocessor import DocumentPreprocessor


class BasicRetriever:
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
    
    def retrieve(self, query: str, max_candidates: int = 20) -> List[Dict[str, Any]]:
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
                    'chunk_metadata': self.preprocessor.chunk_metadata[idx],
                    'chunk_index': int(idx)
                }
                retrieved_chunks.append(chunk_data)
        
        return retrieved_chunks
    
    def retrieve_with_threshold(self, query: str, max_candidates: int = 50, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Retrieve document chunks that exceed a similarity threshold
        
        Args:
            query: Query string
            max_candidates: Maximum number of chunks to consider
            similarity_threshold: Minimum similarity score for inclusion
            
        Returns:
            List of retrieved chunks above the similarity threshold
        """
        # First retrieve top_k candidates
        candidates = self.retrieve(query, max_candidates)
        
        # Filter by similarity threshold
        filtered_chunks = [
            chunk for chunk in candidates 
            if chunk['similarity_score'] >= similarity_threshold
        ]
        
        return filtered_chunks
    
    def get_retrieval_stats(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Get detailed statistics about the retrieval process
        
        Args:
            query: Query string
            top_k: Number of top chunks to analyze
            
        Returns:
            Dictionary containing retrieval statistics
        """
        retrieved_chunks = self.retrieve(query, top_k)
        
        if not retrieved_chunks:
            return {"error": "No chunks retrieved"}
        
        similarities = [chunk['similarity_score'] for chunk in retrieved_chunks]
        
        stats = {
            'query': query,
            'num_retrieved': len(retrieved_chunks),
            'max_similarity': max(similarities),
            'min_similarity': min(similarities),
            'avg_similarity': sum(similarities) / len(similarities),
            'similarity_std': np.std(similarities),
            'unique_documents': len(set(chunk['chunk_metadata']['doc_id'] for chunk in retrieved_chunks))
        }
        
        return stats
    
    def format_context(self, retrieved_chunks: List[Dict[str, Any]], max_context_length: int = 4000) -> str:
        """
        Format retrieved chunks into a context string for the generation module
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            max_context_length: Maximum length of context in characters
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for chunk in retrieved_chunks:
            chunk_text = chunk['chunk_text']
            score = chunk['similarity_score']
            
            # Format with similarity score for better context understanding
            formatted_chunk = f"[Relevance: {score:.3f}] {chunk_text}"
            
            if current_length + len(formatted_chunk) > max_context_length:
                break
                
            context_parts.append(formatted_chunk)
            current_length += len(formatted_chunk)
        
        return "\n\n".join(context_parts)
