"""
Generation Module for RAG Pipeline
Handles LLM integration for response generation using retrieved context
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI
import json


class ResponseGenerator:
    """
    Handles response generation using OpenAI's LLM with retrieved context
    """
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.1):
        """
        Initialize the response generator
        
        Args:
            openai_api_key: OpenAI API key
            model: OpenAI model to use for generation
            temperature: Generation temperature (lower = more deterministic)
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.temperature = temperature
    
    def create_prompt(self, query: str, context: str, prompt_template: Optional[str] = None) -> str:
        """
        Create a structured prompt combining query and context
        
        Args:
            query: User's question
            context: Retrieved context from documents
            prompt_template: Custom prompt template (uses default if None)
            
        Returns:
            Formatted prompt string
        """
        if prompt_template is None:
            prompt_template = """You are a helpful assistant that answers questions based on the provided context. 
Use only the information from the context to answer the question. If the context doesn't contain enough information to answer the question completely, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt_template.format(context=context, query=query)
    
    def generate_response(self, query: str, context: str, 
                         prompt_template: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response using the LLM
        
        Args:
            query: User's question
            context: Retrieved context
            prompt_template: Custom prompt template
            
        Returns:
            Dictionary containing response and metadata
        """
        prompt = self.create_prompt(query, context, prompt_template)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
            )
            
            generated_text = response.choices[0].message.content
            
            return {
                "response": generated_text,
                "model": self.model,
                "temperature": self.temperature,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "finish_reason": response.choices[0].finish_reason,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "response": None,
                "model": self.model,
                "temperature": self.temperature,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "finish_reason": None,
                "success": False,
                "error": str(e)
            }
    
    def generate_with_chunks(self, query: str, processed_chunks: List[Dict[str, Any]],
                           prompt_template: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate response using processed chunk data
        
        Args:
            query: User's question
            processed_chunks: List of processed chunks from reranker
            prompt_template: Custom prompt template
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary containing response and metadata including chunk info
        """
        # Format context from chunks
        context_parts = []
        for i, chunk in enumerate(processed_chunks, 1):
            context_parts.append(f"[Source {i}] {chunk['chunk_text']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate response
        result = self.generate_response(query, context, prompt_template)
        
        # Add chunk metadata to result
        result["num_chunks_used"] = len(processed_chunks)
        result["chunk_sources"] = [
            {
                "rank": i + 1,
                "doc_id": chunk['chunk_metadata']['doc_id'],
                "similarity_score": chunk['similarity_score']
            }
            for i, chunk in enumerate(processed_chunks)
        ]
        
        return result
