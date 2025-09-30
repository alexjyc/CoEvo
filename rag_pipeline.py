"""
Main RAG Pipeline
Integrates all components: Preprocessor, Retrieval, Reranking, Generation, and Evaluation
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from preprocessor import DocumentPreprocessor
from retrieval import BasicRetriever
from reranker import Reranker
from generation import ResponseGenerator
from evaluation import LLMJudge


class RAGPipeline:
    """
    Complete RAG Pipeline integrating all components for end-to-end functionality
    """
    
    def __init__(self, openai_api_key: str, 
                 chunk_size: int = 500, 
                 chunk_overlap: int = 50,
                 generation_model: str = "gpt-4o-mini",
                 temperature: float = 0,
                 judge_model: str = "gpt-4o-mini",
                 judge_confidence_threshold: float = 0.8):
        """
        Initialize the complete RAG pipeline
        
        Args:
            openai_api_key: OpenAI API key
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            generation_model: OpenAI model for generation
            temperature: Generation temperature
        """
        self.api_key = openai_api_key
        
        # Initialize all components
        self.preprocessor = DocumentPreprocessor(openai_api_key, chunk_size, chunk_overlap)
        self.retriever = None  # Will be initialized after preprocessing
        self.reranker = Reranker(default_k=5)
        self.generator = ResponseGenerator(openai_api_key, generation_model, temperature)
        self.llm_judge = LLMJudge(openai_api_key, judge_model, judge_confidence_threshold)
        # self.evaluator = RAGEvaluator()
        
        self.is_ready = False
    
    def load_and_process_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Load documents and create searchable index
        
        Args:
            documents: List of documents with 'text' field
        """
        print("Loading and processing documents...")
        
        # Process documents with preprocessor
        self.preprocessor.process_documents(documents)
        
        # Initialize retriever with processed index
        self.retriever = BasicRetriever(self.preprocessor, self.api_key)
        
        self.is_ready = True
        print(f"Pipeline ready! Processed {len(documents)} documents.")
    
    def load_from_saved_index(self, index_path: str, metadata_path: str) -> None:
        """
        Load pipeline from saved FAISS index and metadata
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
        """
        print("Loading from saved index...")
        
        # Load preprocessed data
        self.preprocessor.load_index(index_path, metadata_path)
        
        # Initialize retriever
        self.retriever = BasicRetriever(self.preprocessor, self.api_key)
        
        self.is_ready = True
        print("Pipeline loaded successfully!")
    
    def save_index(self, index_path: str, metadata_path: str) -> None:
        """
        Save the current index and metadata
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Process documents first.")
        
        self.preprocessor.save_index(index_path, metadata_path)
    
    def query(self, query: str, 
              retrieval_k: int = 10,
              final_k: int = 5,
              remove_duplicates: bool = True,
              filter_length: bool = True,
              diversify: bool = True,
              max_tokens: int = 500,
              use_financial_prompt: bool = False) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline
        
        Args:
            query: User's question
            retrieval_k: Number of chunks to retrieve initially
            final_k: Final number of chunks to use for generation
            remove_duplicates: Whether to remove duplicate chunks
            filter_length: Whether to filter chunks by length
            diversify: Whether to diversify source documents
            max_tokens: Maximum tokens for generation
            use_financial_prompt: Whether to use financial-specific prompt
            
        Returns:
            Dictionary containing complete pipeline results
        """
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Load documents or saved index first.")
        
        print(f"Processing query: {query}")
        
        # Step 1: Retrieval
        print("1. Retrieving relevant chunks...")
        retrieved_chunks = self.retriever.retrieve(query, retrieval_k)
        
        # Step 2: Reranking
        print("2. Reranking chunks...")
        processed_chunks = self.reranker.process_chunks(
            retrieved_chunks,
            k=final_k,
            remove_duplicates=remove_duplicates,
            filter_length=filter_length,
            diversify=diversify
        )
        
        # Step 3: Generation
        print("3. Generating response...")
        if use_financial_prompt:
            generation_result = self.generator.generate_financial_response(
                query, processed_chunks, max_tokens
            )
        else:
            generation_result = self.generator.generate_with_chunks(
                query, processed_chunks, max_tokens=max_tokens
            )
        
        # Step 4: Compile results
        pipeline_result = {
            'query': query,
            'retrieved_chunks': retrieved_chunks,
            'processed_chunks': processed_chunks,
            'generation_result': generation_result,
            'response': generation_result.get('response', 'No response generated'),
            'retrieval_stats': {
                'initial_retrieved': len(retrieved_chunks),
                'final_processed': len(processed_chunks),
                'avg_similarity': sum(c['similarity_score'] for c in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0
            }
        }
        
        print("4. Query processing complete!")
        return pipeline_result


    def batch_query(self, queries: List[str], **query_kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple queries through the complete RAG pipeline
        """
        return [self.query(query, **query_kwargs) for query in queries]
    
    
    def batch_judge(self, evaluation_data: List[Dict[str, Any]], **query_kwargs) -> Dict[str, Any]:
        """
        Process multiple queries and judge them against ground truth
        
        Args:
            evaluation_data: List of dicts with 'query' and 'ground_truth' keys
            **query_kwargs: Additional arguments for query method
            
        Returns:
            Dictionary containing batch results and statistics
        """
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Load documents or saved index first.")
        
        print(f"Processing {len(evaluation_data)} queries with LLM judgment...")
        
        results = []
        
        for i, data in enumerate(evaluation_data):
            print(f"\nProcessing query {i+1}/{len(evaluation_data)}")

            result = self.llm_judge.judge_response(
                generated_response=data['generated_response'],
                ground_truth=data['ground_truth'],
                query=data['query'],
                context=data['context']
            )
            
            results.append(result)

        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the current pipeline configuration
        
        Returns:
            Dictionary containing pipeline information
        """
        info = {
            'is_ready': self.is_ready,
            'generation_model': self.generator.model,
            'temperature': self.generator.temperature,
            'default_k': self.reranker.default_k,
            'judge_model': self.llm_judge.judge_model,
            'judge_confidence_threshold': self.llm_judge.confidence_threshold,
        }
        
        if self.is_ready:
            info.update({
                'num_chunks': len(self.preprocessor.chunks),
                'chunk_size': self.preprocessor.chunk_size,
                'chunk_overlap': self.preprocessor.chunk_overlap,
                'index_size': self.preprocessor.faiss_index.ntotal if self.preprocessor.faiss_index else 0
            })
        
        return info


def create_pipeline_from_config(config_path: Optional[str] = None) -> RAGPipeline:
    """
    Create a RAG pipeline from configuration file or environment variables
    
    Args:
        config_path: Path to configuration file (uses .env if None)
        
    Returns:
        Initialized RAGPipeline instance
    """
    # Load environment variables
    if config_path:
        load_dotenv(config_path)
    else:
        load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Get optional configuration parameters
    chunk_size = int(os.getenv('CHUNK_SIZE', 500))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))
    model = os.getenv('GENERATION_MODEL', 'gpt-4o-mini')
    temperature = float(os.getenv('TEMPERATURE', 0.1))
    judge_model = os.getenv('JUDGE_MODEL', 'gpt-4o-mini')
    judge_threshold = float(os.getenv('JUDGE_CONFIDENCE_THRESHOLD', 0.8))
    
    return RAGPipeline(
        openai_api_key=api_key,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        generation_model=model,
        temperature=temperature,
        judge_model=judge_model,
        judge_confidence_threshold=judge_threshold
    )