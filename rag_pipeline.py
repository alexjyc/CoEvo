"""
Main RAG Pipeline
Integrates all components: Preprocessor, Retrieval, Reranking, Generation, and Evaluation
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
import time
from dotenv import load_dotenv


from LLMBase import LLMBase
from preprocessor import DocumentPreprocessor
from retrieval import Retriever, RetrievalMethod
from reranker import Reranker
from generation import ResponseGenerator
from evaluation import Evaluator
from langfuse import get_client


class RAGPipeline:
    """
    Complete RAG Pipeline integrating all components for end-to-end functionality
    """
    
    def __init__(self, openai_api_key: str, 
                 chunk_size: int = 600, 
                 chunk_overlap: int = 50,
                 generation_model: str = "gpt-4o-mini",
                 temperature: float = 0,
                 retrieval_method: str = "hybrid",
                 hybrid_dense_weight: float = 0.7):
        """
        Initialize the complete RAG pipeline
        
        Args:
            openai_api_key: OpenAI API key
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            generation_model: OpenAI model for generation
            temperature: Generation temperature
            cross_encoder_strategy: Optional reranking strategy ('pointwise' or 'pairwise')
            cross_encoder_options: Optional configuration passed to the Reranker
            hybrid_dense_weight: Default dense weight for hybrid rank fusion
        """
        self.api_key = openai_api_key
        
        # Determine retrieval method and required indices
        if retrieval_method == "hybrid":
            self.retrieval_method = RetrievalMethod.HYBRID
        elif retrieval_method == "bm25":
            self.retrieval_method = RetrievalMethod.BM25_ONLY
        elif retrieval_method == "dense":
            self.retrieval_method = RetrievalMethod.DENSE_ONLY
        else:
            raise ValueError(f"Invalid retrieval method: {retrieval_method}")
        
        # Initialize all components with retrieval method info
        self.preprocessor = DocumentPreprocessor(
            chunk_size, 
            chunk_overlap
        )
        self.retriever = None  # Will be initialized after preprocessing
        self.base_component = LLMBase(model_name=generation_model)
        self.hybrid_dense_weight = max(0.0, min(1.0, float(hybrid_dense_weight)))
        self.reranker = Reranker(default_k=10, hybrid_dense_weight=self.hybrid_dense_weight)
        self.generator = ResponseGenerator(generation_model, temperature)
        self.evaluator = Evaluator()
        self.is_ready = False

        self.langfuse_context = get_client()
    
    def load_and_process_documents(self, documents: List[Dict[str, Any]], use_context: bool = True) -> None:
        """
        Load documents and create searchable index
        
        Args:
            documents: List of documents with 'text' field
        """
        print("Loading and processing documents...")
        
        # Process documents with preprocessor
        self.preprocessor.process_documents(documents, use_context=use_context)
        
        # Initialize retriever with processed index
        self.retriever = Retriever(self.preprocessor, self.api_key)
        
        self.is_ready = True
        print(f"Pipeline ready! Processed {len(documents)} documents.")
    
    # def load_from_saved_index(self, index_path: str, metadata_path: str, use_context: bool = True) -> None:
    #     """
    #     Load pipeline from saved FAISS index and metadata
        
    #     Args:
    #         index_path: Path to FAISS index file
    #         metadata_path: Path to metadata file
    #     """
    #     print("Loading from saved index...")
        
    #     # Load preprocessed data
    #     self.preprocessor.load_index(index_path, metadata_path, use_context=use_context)
        
    #     # Initialize retriever
    #     self.retriever = Retriever(self.preprocessor, self.api_key)
        
    #     self.is_ready = True
    #     print("Pipeline loaded successfully!")
    
    def save_index(self, index_path: str, metadata_path: str, use_context: bool = True) -> None:
        """
        Save the current index and metadata
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Process documents first.")
        
        self.preprocessor.save_index(index_path, metadata_path, use_context=use_context)
    
    async def query_with_eval(self, query: str, 
            ground_truth: str,
            retrieval_k: int = 20,
            final_k: int = 10,
            dense_weight: Optional[float] = None,
            bm25_k: Optional[int] = None,
            dense_k: Optional[int] = None,
            ) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline
        
        Args:
            query: User's question
            retrieval_k: Number of chunks to retrieve initially
            final_k: Final number of chunks to use for generation
            
        Returns:
            Dictionary containing complete pipeline results
        """
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Load documents or saved index first.")
        
        print(f"Processing query with Langfuse tracing: {query}")

        with self.langfuse_context.start_as_current_span(name=f"rag-{self.retrieval_method.value}") as trace:
            # Store trace_id for later use
            trace_id = trace.trace_id

            with trace.start_as_current_span(
                name="query_reformation",
                input={'query': query}
            ) as reformulation_span:
                print("0. Reformulating query...")
                response = self.base_component.reformate_query(query)
                reformation_output = {
                    'reformulated_query': response.reformatted_query,
                    'rationale': response.rationale
                }
                reformulation_span.update(output=reformation_output)

            with trace.start_as_current_span(
                name="retrieval",
                input={'question': reformation_output['reformulated_query'], 'retrieval_k': retrieval_k, 'method': self.retrieval_method.value}
            ) as retrieval_span:
                print("1. Retrieving relevant chunks...")
                effective_query = reformation_output['reformulated_query']
                retrieval_kwargs: Dict[str, Any] = {}
                if self.retrieval_method == RetrievalMethod.HYBRID:
                    retrieval_kwargs['bm25_k'] = bm25_k or retrieval_k
                    retrieval_kwargs['dense_k'] = dense_k or retrieval_k

                retrieval_payload = self.retriever.retrieve(
                    effective_query,
                    retrieval_k,
                    method=self.retrieval_method,
                    **retrieval_kwargs
                )

                hybrid_metadata: Dict[str, Any] = {}
                if self.retrieval_method == RetrievalMethod.HYBRID and isinstance(retrieval_payload, dict):
                    effective_weight = dense_weight if dense_weight is not None else self.hybrid_dense_weight
                    retrieved_chunks = self.reranker.rank_fusion(
                        bm25_results=retrieval_payload.get('bm25', []),
                        dense_results=retrieval_payload.get('dense', []),
                        k=retrieval_k,
                        dense_weight=effective_weight
                    )
                    hybrid_metadata = {
                        'dense_weight': effective_weight,
                        'hybrid_sources': {
                            'bm25': len(retrieval_payload.get('bm25', [])),
                            'dense': len(retrieval_payload.get('dense', []))
                        }
                    }
                else:
                    retrieved_chunks = retrieval_payload  # type: ignore[assignment]

                contexts = [chunk['chunk_text'] for chunk in retrieved_chunks]
                retrieval_output = {
                    'contexts': contexts,
                    'num_retrieved': len(retrieved_chunks),
                    'avg_similarity': sum(c['similarity_score'] for c in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0,
                    'retrieval_method': self.retrieval_method.value
                }
                if hybrid_metadata:
                    retrieval_output.update(hybrid_metadata)
                retrieval_span.update(output=retrieval_output)

            with trace.start_as_current_span(
                name="reranking",
                input={'retrieved_chunks': len(retrieved_chunks), 'target_k': final_k}
            ) as reranking_span:
                print("2. Reranking chunks...")
                processed_chunks = self.reranker.process_chunks(
                    retrieved_chunks,
                    k=final_k
                )
                rerank_output = {
                    'processed_chunks': len(processed_chunks),
                    'final_contexts': [chunk['chunk_text'] for chunk in processed_chunks]
                }
                reranking_span.update(output=rerank_output)

            with trace.start_as_current_span(
                name="generation",
                input={'question': query, 'contexts': [chunk['chunk_text'] for chunk in processed_chunks]}
            ) as generation_span:
                print("3. Generating response...")
                response = self.base_component.generate_answer(
                    reformation_output['reformulated_query'],
                    rerank_output['final_contexts']
                )

                generation_output = {
                    'answer': response.answer,
                    'references': response.reference,
                    'rationale': response.rationale
                }

                generation_span.update(output=generation_output)

            print("4. Evaluating response...")
            evaluation_scores = await self.evaluator.evaluate_trace(query, contexts, generation_output['answer'], ground_truth)

            for key, value in evaluation_scores.items():
                trace.score(
                    name=key,
                    value=value,
                    data_type="NUMERIC"
                )

        return evaluation_scores

    async def batch_query(self, dataset: List[Dict[str, str]], 
                               retrieval_k: int = 50, final_k: int = 12,
                               max_concurrent: int = 5,
                               dense_weight: Optional[float] = None,
                               bm25_k: Optional[int] = None,
                               dense_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process large datasets using asyncio concurrency (BEST for API-heavy workloads)
        
        Args:
            dataset: List of {'query': str, 'ground_truth': str}
            max_concurrent: Maximum concurrent tasks
            dense_weight: Optional override for hybrid dense weighting
            bm25_k: Optional override for BM25 candidate count
            dense_k: Optional override for dense candidate count
            
        Returns:
            List of evaluation results
        """
        print(f"🚀 Processing {len(dataset)} queries with asyncio (max_concurrent={max_concurrent})")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        start_time = time.time()
        
        async def process_with_semaphore(item: Dict[str, str]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    result = await self.query_with_eval(
                        query=item['query'],
                        ground_truth=item['ground_truth'], 
                        retrieval_k=retrieval_k,
                        final_k=final_k,
                        dense_weight=dense_weight,
                        bm25_k=bm25_k,
                        dense_k=dense_k
                    )
                    return result
                except Exception as e:
                    print(f"❌ Error processing query '{item['query'][:50]}...': {e}")
                    return {
                        'error': str(e),
                        'query': item['query'],
                        'ground_truth': item['ground_truth']
                    }
        
        # Execute all tasks concurrently
        tasks = [process_with_semaphore(item) for item in dataset]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'error': str(result),
                    'query': dataset[i]['query'],
                    'ground_truth': dataset[i]['ground_truth']
                })
            else:
                processed_results.append(result)
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in processed_results if 'error' not in r)
        
        print(f"✅ Async processing complete: {success_count}/{len(dataset)} successful in {total_time:.2f}s")
        print(f"📊 Throughput: {len(dataset)/total_time:.2f} queries/second")
        
        return processed_results

def create_pipeline_from_config(**kwargs) -> RAGPipeline:
    """
    Create a RAG pipeline from configuration file or environment variables
    
    Args:
        config_path: Path to configuration file (uses .env if None)
        
    Returns:
        Initialized RAGPipeline instance
    """
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set in environment variables")
    
    return RAGPipeline(
        openai_api_key=api_key,
        **kwargs
    )
