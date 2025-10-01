"""
Reranker Module for RAG Pipeline  
Handles top-k selection and prepares for future reranking implementation
"""

import numpy as np
from typing import List, Dict, Any, Optional
import re


class Reranker:
    """
    Handles reranking of retrieved chunks including top-k selection
    and preparation for future reranking capabilities
    """
    
    def __init__(self, default_k: int = 5):
        """
        Initialize the Reranker
        
        Args:
            default_k: Default number of top chunks to select
        """
        self.default_k = default_k
    
    def select_top_k(self, retrieved_chunks: List[Dict[str, Any]], k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Select top-k chunks based on similarity scores
        
        Args:
            retrieved_chunks: List of retrieved chunks with similarity scores
            k: Number of top chunks to select (uses default_k if None)
            
        Returns:
            List of top-k chunks
        """
        k = k or self.default_k
        
        # Sort by similarity score in descending order
        sorted_chunks = sorted(
            retrieved_chunks, 
            key=lambda x: x['similarity_score'], 
            reverse=True
        )
        
        return sorted_chunks[:k]
    
    def remove_duplicates(self, retrieved_chunks: List[Dict[str, Any]], 
                         similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate chunks based on text similarity
        
        Args:
            retrieved_chunks: List of retrieved chunks
            similarity_threshold: Threshold for considering chunks as duplicates
            
        Returns:
            List of chunks with duplicates removed
        """
        if not retrieved_chunks:
            return []
        
        unique_chunks = [retrieved_chunks[0]]  # Keep the first chunk
        
        for chunk in retrieved_chunks[1:]:
            is_duplicate = False
            current_text = chunk['chunk_text'].lower().strip()
            
            for unique_chunk in unique_chunks:
                unique_text = unique_chunk['chunk_text'].lower().strip()
                
                # Simple text overlap similarity
                overlap_ratio = self._calculate_text_overlap(current_text, unique_text)
                
                if overlap_ratio >= similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate text overlap ratio between two strings
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Overlap ratio between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based overlap calculation
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def filter_by_length(self, retrieved_chunks: List[Dict[str, Any]], 
                        min_length: int = 50, max_length: int = 1000) -> List[Dict[str, Any]]:
        """
        Filter chunks by text length to ensure quality
        
        Args:
            retrieved_chunks: List of retrieved chunks
            min_length: Minimum chunk length in characters
            max_length: Maximum chunk length in characters
            
        Returns:
            List of chunks within length bounds
        """
        filtered_chunks = []
        
        for chunk in retrieved_chunks:
            text_length = len(chunk['chunk_text'])
            if min_length <= text_length <= max_length:
                filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def diversify_sources(self, retrieved_chunks: List[Dict[str, Any]], 
                         max_per_document: int = 2) -> List[Dict[str, Any]]:
        """
        Diversify chunks to avoid over-representation from single documents
        
        Args:
            retrieved_chunks: List of retrieved chunks
            max_per_document: Maximum chunks allowed per source document
            
        Returns:
            List of diversified chunks
        """
        doc_count = {}
        diversified_chunks = []
        
        for chunk in retrieved_chunks:
            doc_id = chunk['chunk_metadata']['doc_id']
            current_count = doc_count.get(doc_id, 0)
            
            if current_count < max_per_document:
                diversified_chunks.append(chunk)
                doc_count[doc_id] = current_count + 1
        
        return diversified_chunks
    
    def process_chunks(self, retrieved_chunks: List[Dict[str, Any]], 
                      k: Optional[int] = None,
                      remove_duplicates: bool = True,
                      filter_length: bool = True,
                      diversify: bool = True,
                      **kwargs) -> List[Dict[str, Any]]:
        """
        Main processing pipeline that applies all reranking steps
        
        Args:
            retrieved_chunks: List of retrieved chunks
            k: Number of final chunks to return
            remove_duplicates: Whether to remove duplicate chunks
            filter_length: Whether to filter by chunk length
            diversify: Whether to diversify source documents
            **kwargs: Additional parameters for specific processing steps
            
        Returns:
            List of processed chunks ready for generation
        """
        processed_chunks = retrieved_chunks.copy()
        
        # Step 1: Remove duplicates if requested
        if remove_duplicates:
            duplicate_threshold = kwargs.get('duplicate_threshold', 0.95)
            processed_chunks = self.remove_duplicates(processed_chunks, duplicate_threshold)
        
        # Step 2: Filter by length if requested
        if filter_length:
            min_length = kwargs.get('min_length', 50)
            max_length = kwargs.get('max_length', 1000)
            processed_chunks = self.filter_by_length(processed_chunks, min_length, max_length)
        
        # Step 3: Diversify sources if requested
        if diversify:
            max_per_doc = kwargs.get('max_per_document', 2)
            processed_chunks = self.diversify_sources(processed_chunks, max_per_doc)
        
        # Step 4: Select top-k chunks
        final_chunks = self.select_top_k(processed_chunks, k)
        
        return final_chunks
    
    def get_processing_stats(self, original_chunks: List[Dict[str, Any]], 
                           processed_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the processing pipeline
        
        Args:
            original_chunks: Original list of retrieved chunks
            processed_chunks: Processed list of chunks
            
        Returns:
            Dictionary containing processing statistics
        """
        original_docs = set(chunk['chunk_metadata']['doc_id'] for chunk in original_chunks)
        processed_docs = set(chunk['chunk_metadata']['doc_id'] for chunk in processed_chunks)
        
        stats = {
            'original_count': len(original_chunks),
            'final_count': len(processed_chunks),
            'reduction_ratio': 1 - (len(processed_chunks) / len(original_chunks)) if original_chunks else 0,
            'original_doc_count': len(original_docs),
            'final_doc_count': len(processed_docs),
            'avg_similarity_original': np.mean([chunk['similarity_score'] for chunk in original_chunks]) if original_chunks else 0,
            'avg_similarity_final': np.mean([chunk['similarity_score'] for chunk in processed_chunks]) if processed_chunks else 0
        }
        
        return stats