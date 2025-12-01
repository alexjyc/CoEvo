"""
Unified RAG Pipeline
Consolidates RAGPipeline and AgentRAG into a single, mode-based architecture

Modes:
- FAST: Single-pass, optimized for production (no LangGraph overhead)
- ITERATIVE: Multi-iteration with quality-based feedback loops
- OPTIMIZE: Full prompt optimization with RL (uses LangGraph)
"""

import os
import asyncio
from enum import Enum
from typing import List, Dict, Any, Optional, Annotated
import time
from dotenv import load_dotenv
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict as ExtendedTypedDict

from LLMBase import LLMBase
from preprocessor import DocumentPreprocessor
from retrieval import Retriever
from reranker import Reranker
from evaluation import Evaluator
from langfuse import get_client
from memory_system import (
    MemoryStore,
    ExecutionMemory,
    ModuleMemory,
    ModuleType
)


class PipelineMode(Enum):
    """Pipeline execution modes"""
    FAST = "fast"           # Single-pass, optimized (production)
    ITERATIVE = "iterative" # Multiple iterations with feedback
    OPTIMIZE = "optimize"   # Full prompt optimization (training)


class RAGState(ExtendedTypedDict):
    """State schema for iterative/optimize modes"""
    # Input
    query: str
    ground_truth: Optional[str]
    
    # Retrieval
    retrieved_chunks: Optional[Dict[str, List[Dict[str, Any]]]]
    contexts: Optional[List[str]]
    generated_queries: Optional[Any]  # QueryPlannerResponse.query
    
    # Reranking
    reranked_docs: Optional[List[str]]
    
    # Generation
    answer: Optional[str]
    rationale: Optional[str]
    
    # Evaluation scores
    retrieval_scores: Optional[Dict[str, float]]
    reranking_scores: Optional[Dict[str, float]]
    generation_scores: Optional[Dict[str, float]]
    retrieval_f1: Optional[float]
    reranking_f1: Optional[float]
    generation_quality: Optional[float]
    overall_quality: Optional[float]
    
    # Memory system
    memory_store: Optional[MemoryStore]
    retrieved_memories: Optional[Dict[str, List[ExecutionMemory]]]  # {module_name: memories}
    feedback: Optional[Dict[str, str]]  # {module_name: feedback_text}
    
    # Iteration control
    iteration: int
    max_iterations: int
    converged: bool
    improvement_history: Annotated[List[Dict[str, Any]], operator.add]
    
    # Configuration
    retrieval_k: int
    final_k: int
    quality_threshold: float


class UnifiedRAGPipeline:
    """
    Unified RAG Pipeline with multiple execution modes
    
    Usage:
        # Production (fast)
        pipeline = UnifiedRAGPipeline(mode=PipelineMode.FAST)
        result = await pipeline.query("What is revenue?")
        
        # Experimentation (iterative)
        pipeline = UnifiedRAGPipeline(mode=PipelineMode.ITERATIVE, max_iterations=3)
        result = await pipeline.query("What is revenue?")
        
        # Training (optimize)
        pipeline = UnifiedRAGPipeline(mode=PipelineMode.OPTIMIZE)
        result = await pipeline.query("What is revenue?")
    """
    
    def __init__(
        self,
        mode: PipelineMode = PipelineMode.FAST,
        chunk_size: int = 600,
        chunk_overlap: int = 50,
        generation_model: str = "gpt-4o-mini",
        max_iterations: int = 3,
        quality_threshold: float = 0.80,
        enable_tracing: bool = True,
        **kwargs
    ):
        """
        Initialize unified RAG pipeline
        
        Args:
            mode: Execution mode (FAST/ITERATIVE/OPTIMIZE)
            chunk_size: Text chunk size
            chunk_overlap: Chunk overlap
            generation_model: LLM model name
            max_iterations: Max iterations for ITERATIVE/OPTIMIZE modes
            quality_threshold: Quality threshold for convergence
            enable_tracing: Enable Langfuse tracing (FAST mode only)
        """
        self.mode = mode
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.enable_tracing = enable_tracing
        
        # Initialize shared components
        self.preprocessor = DocumentPreprocessor(chunk_size, chunk_overlap)
        self.retriever = None
        self.base_component = LLMBase(model_name=generation_model)
        self.reranker = Reranker()
        self.evaluator = Evaluator()
        self.is_ready = False
        
        # Memory system for OPTIMIZE mode
        self.memory_store = None
        if mode == PipelineMode.OPTIMIZE:
            self.memory_store = MemoryStore(
                judge_model=generation_model,
                use_judge_threshold=10,
                max_memories=100
            )
        
        # Mode-specific initialization
        if mode == PipelineMode.FAST and enable_tracing:
            self.langfuse_context = get_client()
        else:
            self.langfuse_context = None
        
        if mode in [PipelineMode.ITERATIVE, PipelineMode.OPTIMIZE]:
            self.graph = self._build_langgraph()
        else:
            self.graph = None
    
    def setup_pipeline(
        self,
        documents: List[Dict[str, Any]],
        evaluation_queries: Optional[List[Dict[str, Any]]] = None,
        use_context: bool = False,
        overlap_threshold: float = 0.5,
        get_representatives: bool = False
    ) -> Dict[str, Any]:
        """Setup pipeline with documents and optional evaluation infrastructure"""
        print(f"Setting up pipeline in {self.mode.value.upper()} mode...")
        
        self.preprocessor.process_documents(documents, use_context=use_context)
        self.retriever = Retriever(self.preprocessor)
        self.is_ready = True
        
        results = {
            'mode': self.mode.value,
            'documents_processed': len(documents),
            'chunks_created': len(self.preprocessor.chunks),
            'relevance_labels': None,
            'representatives': None
        }
        
        if evaluation_queries:
            relevance_labels = self.preprocessor.create_relevance_labels(
                evaluation_queries, overlap_threshold
            )
            results['relevance_labels'] = relevance_labels
            self.reranker.set_relevance_labels(evaluation_queries, relevance_labels)
            
            if get_representatives:
                representatives = self.preprocessor.get_representative_data(evaluation_queries)
                results['representatives'] = representatives
                print(f"✅ Selected {len(representatives)} representative queries")
            
            print(f"✅ Pipeline ready with evaluation infrastructure!")
        else:
            print(f"✅ Pipeline ready (no evaluation queries provided)")
        
        return results
    
    async def query(
        self,
        query: str,
        ground_truth: Optional[str] = None,
        retrieval_k: int = 20,
        final_k: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main query method - routes to appropriate mode
        
        Args:
            query: User query
            ground_truth: Optional ground truth for evaluation
            retrieval_k: Number of chunks to retrieve
            final_k: Final number of chunks for generation
            **kwargs: Mode-specific kwargs
        """
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Call setup_pipeline() first.")
        
        if self.mode == PipelineMode.FAST:
            return await self._fast_query(
                query, ground_truth, retrieval_k, final_k, **kwargs
            )
        elif self.mode == PipelineMode.ITERATIVE:
            return await self._iterative_query(
                query, ground_truth, retrieval_k, final_k, **kwargs
            )
        else:  # OPTIMIZE
            return await self._optimize_query(
                query, ground_truth, retrieval_k, final_k, **kwargs
            )
    
    async def _fast_query(
        self,
        query: str,
        ground_truth: Optional[str],
        retrieval_k: int,
        final_k: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        FAST mode: Single-pass optimized query
        Uses current RAGPipeline logic with parallel execution
        """
        print(f"[FAST MODE] Processing query: {query[:50]}...")
        
        if self.enable_tracing and self.langfuse_context:
            return await self._fast_query_with_tracing(
                query, ground_truth, retrieval_k, final_k
            )
        else:
            return await self._fast_query_no_tracing(
                query, ground_truth, retrieval_k, final_k
            )
    
    async def _fast_query_with_tracing(
        self,
        query: str,
        ground_truth: Optional[str],
        retrieval_k: int,
        final_k: int
    ) -> Dict[str, Any]:
        """Fast mode with Langfuse tracing"""
        with self.langfuse_context.start_as_current_span(name="rag-fast") as trace:
            trace_id = trace.trace_id
            
            # Step 1: Query Reformation
            with trace.start_as_current_span(
                name="query_reformation",
                input={'original_query': query}
            ) as reformation_span:
                response = await self.base_component.query_planner(query)
                reformation_output = {
                    'mode': response.mode,
                    'generated_queries': response.query,
                    'num_queries': len(response.query) if isinstance(response.query, list) else 1
                }
                reformation_span.update(output=reformation_output)
            
            # Step 2: Retrieval (parallel)
            with trace.start_as_current_span(
                name="retrieval",
                input={'query': query, 'retrieval_k': retrieval_k}
            ) as retrieval_span:
                retrieved_chunks, contexts = await self._retrieve_parallel(
                    response, query, retrieval_k
                )
                retrieval_span.update(output={'num_contexts': len(contexts)})
            
            # Step 3: Parallel Reranking + Retrieval Eval
            with trace.start_as_current_span(
                name="parallel_rerank_eval",
                input={'num_contexts': len(contexts)}
            ) as parallel_span:
                reranking_task = self.base_component.rerank_documents(query, contexts)
                retrieval_eval_task = self.evaluator.evaluate_llm(
                    query, contexts, ground_truth
                )
                
                rerank_response, retrieval_scores = await asyncio.gather(
                    reranking_task, retrieval_eval_task
                )
                
                ranked_docs = rerank_response.ranked_documents[:final_k]
                retrieval_f1 = self._calculate_f1(
                    retrieval_scores.get('context_precision', 0),
                    retrieval_scores.get('context_recall', 0)
                )
                
                parallel_span.update(output={
                    'ranked_docs': len(ranked_docs),
                    'retrieval_f1': retrieval_f1
                })
            
            # Step 4: Parallel Generation + Reranking Eval
            with trace.start_as_current_span(
                name="parallel_gen_eval",
                input={'num_contexts': len(ranked_docs)}
            ) as gen_span:
                context_text = "\n\n".join(contexts)
                generation_task = self.base_component.generate_answer(query, context_text)
                reranking_eval_task = self.evaluator.evaluate_llm(
                    query, ranked_docs, ground_truth
                )
                
                gen_response, reranking_scores = await asyncio.gather(
                    generation_task, reranking_eval_task
                )
                
                reranking_f1 = self._calculate_f1(
                    reranking_scores.get('context_precision', 0),
                    reranking_scores.get('context_recall', 0)
                )
                
                gen_span.update(output={
                    'answer_length': len(gen_response.answer),
                    'reranking_f1': reranking_f1
                })
            
            # Step 5: Generation Evaluation
            with trace.start_as_current_span(
                name="generation_eval",
                input={'answer': gen_response.answer[:100]}
            ) as eval_span:
                generation_scores = await self.evaluator.evaluate_generation(
                    query, contexts, gen_response.answer, ground_truth
                )
                
                generation_quality = (
                    generation_scores.get('faithfulness', 0) + 
                    generation_scores.get('answer_correctness', 0)
                ) / 2
                
                eval_span.update(output={'generation_quality': generation_quality})
            
            # Calculate overall quality
            overall_quality = (
                retrieval_f1 * 0.3 +
                reranking_f1 * 0.2 +
                generation_quality * 0.5
            )
            
            trace.score(name="overall_quality", value=overall_quality, data_type="NUMERIC")
            
            return {
                'answer': gen_response.answer,
                'rationale': gen_response.rationale,
                'contexts': contexts,
                'ranked_documents': ranked_docs,
                'retrieval_scores': retrieval_scores,
                'reranking_scores': reranking_scores,
                'generation_scores': generation_scores,
                'retrieval_f1': retrieval_f1,
                'reranking_f1': reranking_f1,
                'generation_quality': generation_quality,
                'overall_quality': overall_quality,
                'trace_id': trace_id,
                'mode': 'fast'
            }
    
    async def _fast_query_no_tracing(
        self,
        query: str,
        ground_truth: Optional[str],
        retrieval_k: int,
        final_k: int
    ) -> Dict[str, Any]:
        """Fast mode without tracing overhead"""
        # Query reformation
        response = await self.base_component.query_planner(query)
        
        # Retrieval
        retrieved_chunks, contexts = await self._retrieve_parallel(
            response, query, retrieval_k
        )
        
        # Parallel reranking + retrieval eval
        reranking_task = self.base_component.rerank_documents(query, contexts)
        retrieval_eval_task = self.evaluator.evaluate_llm(query, contexts, ground_truth)
        rerank_response, retrieval_scores = await asyncio.gather(
            reranking_task, retrieval_eval_task
        )
        
        ranked_docs = rerank_response.ranked_documents[:final_k]
        retrieval_f1 = self._calculate_f1(
            retrieval_scores.get('context_precision', 0),
            retrieval_scores.get('context_recall', 0)
        )
        
        # Parallel generation + reranking eval
        context_text = "\n\n".join(contexts)
        generation_task = self.base_component.generate_answer(query, context_text)
        reranking_eval_task = self.evaluator.evaluate_llm(query, ranked_docs, ground_truth)
        gen_response, reranking_scores = await asyncio.gather(
            generation_task, reranking_eval_task
        )
        
        reranking_f1 = self._calculate_f1(
            reranking_scores.get('context_precision', 0),
            reranking_scores.get('context_recall', 0)
        )
        
        # Generation evaluation
        generation_scores = await self.evaluator.evaluate_generation(
            query, contexts, gen_response.answer, ground_truth
        )
        
        generation_quality = (
            generation_scores.get('faithfulness', 0) + 
            generation_scores.get('answer_correctness', 0)
        ) / 2
        
        overall_quality = (
            retrieval_f1 * 0.3 +
            reranking_f1 * 0.2 +
            generation_quality * 0.5
        )
        
        return {
            'answer': gen_response.answer,
            'rationale': gen_response.rationale,
            'contexts': contexts,
            'ranked_documents': ranked_docs,
            'retrieval_scores': retrieval_scores,
            'reranking_scores': reranking_scores,
            'generation_scores': generation_scores,
            'retrieval_f1': retrieval_f1,
            'reranking_f1': reranking_f1,
            'generation_quality': generation_quality,
            'overall_quality': overall_quality,
            'mode': 'fast'
        }
    
    async def _iterative_query(
        self,
        query: str,
        ground_truth: Optional[str],
        retrieval_k: int,
        final_k: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        ITERATIVE mode: Multiple iterations with quality-based feedback
        Simpler than OPTIMIZE - just retries if quality is low
        """
        print(f"[ITERATIVE MODE] Processing query: {query[:50]}...")
        
        best_result = None
        best_score = 0
        iteration_history = []
        
        for iteration in range(self.max_iterations):
            print(f"  Iteration {iteration + 1}/{self.max_iterations}...")
            
            # Run single pass
            result = await self._fast_query_no_tracing(
                query, ground_truth, retrieval_k, final_k
            )
            
            result['iteration'] = iteration + 1
            iteration_history.append({
                'iteration': iteration + 1,
                'overall_quality': result['overall_quality'],
                'retrieval_f1': result['retrieval_f1'],
                'generation_quality': result['generation_quality']
            })
            
            # Track best result
            if result['overall_quality'] > best_score:
                best_result = result
                best_score = result['overall_quality']
                print(f"    ✓ New best score: {best_score:.3f}")
            
            # Check convergence
            if best_score >= self.quality_threshold:
                print(f"    ✓ Converged! Quality {best_score:.3f} >= threshold {self.quality_threshold}")
                break
            
            # Adjust parameters for next iteration
            if result['retrieval_f1'] < 0.6:
                retrieval_k = min(retrieval_k + 10, 50)
                print(f"    ↻ Low retrieval F1, increasing k to {retrieval_k}")
            
            if result['generation_quality'] < 0.7:
                final_k = min(final_k + 5, 20)
                print(f"    ↻ Low generation quality, increasing final_k to {final_k}")
        
        best_result['mode'] = 'iterative'
        best_result['iteration_history'] = iteration_history
        best_result['total_iterations'] = len(iteration_history)
        best_result['converged'] = best_score >= self.quality_threshold
        
        return best_result
    
    async def _optimize_query(
        self,
        query: str,
        ground_truth: Optional[str],
        retrieval_k: int,
        final_k: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        OPTIMIZE mode: Full LangGraph with memory-weighted feedback
        
        Flow:
        1. Retrieve relevant memories (semantic + judge)
        2. Execute RAG with memory-based feedback
        3. Evaluate all modules separately
        4. Update memory weights
        5. Route: continue if not converged, else stop
        """
        print(f"[OPTIMIZE MODE] Processing query: {query[:50]}...")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  Quality threshold: {self.quality_threshold}")
        
        # Build initial state
        initial_state = {
            'query': query,
            'ground_truth': ground_truth,
            'retrieval_k': retrieval_k,
            'final_k': final_k,
            'iteration': 0,
            'max_iterations': self.max_iterations,
            'converged': False,
            'quality_threshold': self.quality_threshold,
            'improvement_history': [],
            'retrieved_chunks': None,
            'contexts': None,
            'generated_queries': None,
            'reranked_docs': None,
            'answer': None,
            'rationale': None,
            'retrieval_scores': None,
            'reranking_scores': None,
            'generation_scores': None,
            'retrieval_f1': None,
            'reranking_f1': None,
            'generation_quality': None,
            'overall_quality': None,
            'memory_store': self.memory_store,
            'retrieved_memories': {},
            'feedback': {}
        }
        
        # Compile and run graph
        checkpointer = MemorySaver()
        compiled_graph = self.graph.compile(checkpointer=checkpointer)
        
        thread_id = f"optimize-{hash(query)}"
        config = {"configurable": {"thread_id": thread_id}}
        
        print("\n" + "="*60)
        print("Starting LangGraph Execution")
        print("="*60)
        
        final_state = None
        async for state_update in compiled_graph.astream(initial_state, config=config):
            # Get the last state update
            final_state = state_update
        
        print("="*60)
        print("LangGraph Execution Complete")
        print("="*60 + "\n")
        
        if final_state:
            # Extract final state from last node
            last_node_key = list(final_state.keys())[-1]
            result = final_state[last_node_key]
            
            # Format result
            formatted_result = {
                'answer': result.get('answer', ''),
                'rationale': result.get('rationale', ''),
                'contexts': result.get('contexts', []),
                'ranked_documents': result.get('reranked_docs', []),
                'retrieval_scores': result.get('retrieval_scores', {}),
                'reranking_scores': result.get('reranking_scores', {}),
                'generation_scores': result.get('generation_scores', {}),
                'retrieval_f1': result.get('retrieval_f1', 0.0),
                'reranking_f1': result.get('reranking_f1', 0.0),
                'generation_quality': result.get('generation_quality', 0.0),
                'overall_quality': result.get('overall_quality', 0.0),
                'total_iterations': result.get('iteration', 0),
                'converged': result.get('overall_quality', 0.0) >= self.quality_threshold,
                'improvement_history': result.get('improvement_history', []),
                'mode': 'optimize'
            }
            
            # Memory stats
            if self.memory_store and self.memory_store.memories:
                formatted_result['memory_stats'] = self.memory_store.get_stats()
            
            return formatted_result
        
        return {'error': 'LangGraph execution failed', 'mode': 'optimize'}
    
    async def _retrieve_parallel(
        self,
        response,
        query: str,
        retrieval_k: int
    ) -> tuple:
        """Parallel retrieval helper"""
        if response.mode == "decomposition":
            async def retrieve_parallel(sub_query: str):
                bm25_task = asyncio.to_thread(
                    self.retriever.retrieve_bm25, sub_query, retrieval_k
                )
                dense_task = asyncio.to_thread(
                    self.retriever.retrieve_dense, sub_query, retrieval_k
                )
                
                try:
                    bm25_results, dense_results = await asyncio.gather(
                        bm25_task, dense_task
                    )
                    return {"bm25": bm25_results, "dense": dense_results}
                except Exception as e:
                    print(f"⚠️ Error retrieving: {e}")
                    return {"bm25": [], "dense": []}
            
            results = await asyncio.gather(
                *[retrieve_parallel(sq) for sq in response.query],
                return_exceptions=True
            )
            
            retrieved_chunks = {
                'bm25': [chunk for r in results if isinstance(r, dict) for chunk in r['bm25']],
                'dense': [chunk for r in results if isinstance(r, dict) for chunk in r['dense']]
            }
        else:
            retrieved_chunks = self.retriever.retrieve_hybrid(response.query, retrieval_k)
        
        # Fusion
        rrf_chunks = await self.reranker.optimize_fusion_params(
            query=query,
            bm25_results=retrieved_chunks.get('bm25', []),
            dense_results=retrieved_chunks.get('dense', []),
            top_k=retrieval_k
        )
        
        contexts = [chunk['chunk_text'] for chunk in rrf_chunks]
        
        return retrieved_chunks, contexts
    
    def _calculate_f1(self, precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall"""
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0
    
    def _build_langgraph(self) -> StateGraph:
        """
        Build LangGraph for OPTIMIZE mode with memory-augmented feedback
        
        Flow:
        START → memory_retrieve → reform_with_feedback → retrieve → 
        rerank_with_feedback → generate_with_feedback → evaluate → 
        memory_update → routing → [continue/END]
        """
        graph = StateGraph(RAGState)
        
        # Add nodes
        graph.add_node("memory_retrieve", self._memory_retrieve_node)
        graph.add_node("reform_with_feedback", self._reform_with_feedback_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("rerank_with_feedback", self._rerank_with_feedback_node)
        graph.add_node("generate_with_feedback", self._generate_with_feedback_node)
        graph.add_node("evaluate", self._evaluate_node)
        graph.add_node("memory_update", self._memory_update_node)
        
        # Add edges
        graph.add_edge(START, "memory_retrieve")
        graph.add_edge("memory_retrieve", "reform_with_feedback")
        graph.add_edge("reform_with_feedback", "retrieve")
        graph.add_edge("retrieve", "rerank_with_feedback")
        graph.add_edge("rerank_with_feedback", "generate_with_feedback")
        graph.add_edge("generate_with_feedback", "evaluate")
        graph.add_edge("evaluate", "memory_update")
        
        # Conditional routing
        graph.add_conditional_edges(
            "memory_update",
            self._should_continue,
            {
                "continue": "memory_retrieve",  # Loop back
                "stop": END
            }
        )
        
        return graph
    
    # ============================================================================
    # LangGraph Node Implementations for OPTIMIZE Mode
    # ============================================================================
    
    async def _memory_retrieve_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Node 1: Retrieve relevant memories for all three modules
        Uses hybrid semantic + judge retrieval
        """
        print(f"  [Memory Retrieve] Iteration {state['iteration'] + 1}")
        
        if not self.memory_store or not self.memory_store.memories:
            print("    No memories available yet (cold start)")
            return {
                'retrieved_memories': {},
                'feedback': {}
            }
        
        # Retrieve memories for each module
        retrieved_memories = {}
        feedback = {}
        
        for module in [ModuleType.QUERY_PLANNING, ModuleType.RERANKING, ModuleType.GENERATION]:
            memories, feedback_text = await self.memory_store.retrieve_with_reflection(
                query=state['query'],
                module=module,
                current_state=state,
                top_k=5
            )
            
            retrieved_memories[module.value] = memories
            feedback[module.value] = feedback_text
            
            if memories:
                print(f"    Retrieved {len(memories)} memories for {module.value}")
        
        # Show memory stats
        if self.memory_store.memories:
            stats = self.memory_store.get_stats()
            print(f"    Memory Stats: {stats['total_memories']} total, "
                  f"{stats['high_quality_memories']} high-quality")
        
        return {
            'retrieved_memories': retrieved_memories,
            'feedback': feedback
        }
    
    async def _reform_with_feedback_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Node 2: Query planning with memory-based feedback
        Passes feedback to LLMBase.query_planner()
        """
        print(f"  [Query Planning] With feedback")
        
        feedback = state.get('feedback', {}).get(ModuleType.QUERY_PLANNING.value, "")
        
        if feedback:
            print(f"    Using feedback: {feedback[:80]}...")
        
        response = await self.base_component.query_planner(
            query=state['query'],
            feedback=feedback
        )
        
        print(f"    Mode: {response.mode}, Queries: {len(response.query) if isinstance(response.query, list) else 1}")
        
        return {'generated_queries': response}
    
    async def _retrieve_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Node 3: Execute retrieval (parallel if decomposition)
        """
        print(f"  [Retrieval] k={state['retrieval_k']}")
        
        response = state['generated_queries']
        retrieved_chunks, contexts = await self._retrieve_parallel(
            response, state['query'], state['retrieval_k']
        )
        
        print(f"    Retrieved {len(contexts)} contexts")
        
        return {
            'retrieved_chunks': retrieved_chunks,
            'contexts': contexts
        }
    
    async def _rerank_with_feedback_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Node 4: Reranking with memory-based feedback
        Passes feedback to LLMBase.rerank_documents()
        """
        print(f"  [Reranking] With feedback")
        
        feedback = state.get('feedback', {}).get(ModuleType.RERANKING.value, "")
        
        if feedback:
            print(f"    Using feedback: {feedback[:80]}...")
        
        rerank_response = await self.base_component.rerank_documents(
            query=state['query'],
            documents=state['contexts'],
            feedback=feedback
        )
        
        reranked_docs = rerank_response.ranked_documents[:state['final_k']]
        print(f"    Reranked to top {len(reranked_docs)}")
        
        return {'reranked_docs': reranked_docs}
    
    async def _generate_with_feedback_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Node 5: Answer generation with memory-based feedback
        Passes feedback to LLMBase.generate_answer()
        """
        print(f"  [Generation] With feedback")
        
        feedback = state.get('feedback', {}).get(ModuleType.GENERATION.value, "")
        
        if feedback:
            print(f"    Using feedback: {feedback[:80]}...")
        
        context_text = "\n\n".join(state['reranked_docs'])
        
        gen_response = await self.base_component.generate_answer(
            query=state['query'],
            context=context_text,
            feedback=feedback
        )
        
        print(f"    Generated answer ({len(gen_response.answer)} chars)")
        
        return {
            'answer': gen_response.answer,
            'rationale': gen_response.rationale
        }
    
    async def _evaluate_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Node 6: Evaluate all three modules separately
        - Retrieval: context_precision + context_recall
        - Reranking: context_precision + context_recall (post-rerank)
        - Generation: faithfulness + answer_correctness
        """
        print(f"  [Evaluation] All modules")
        
        ground_truth = state.get('ground_truth')
        
        # Evaluate retrieval (pre-rerank contexts)
        retrieval_scores = await self.evaluator.evaluate_llm(
            state['query'], state['contexts'], ground_truth
        )
        retrieval_f1 = self._calculate_f1(
            retrieval_scores.get('context_precision', 0),
            retrieval_scores.get('context_recall', 0)
        )
        
        # Evaluate reranking (post-rerank contexts)
        reranking_scores = await self.evaluator.evaluate_llm(
            state['query'], state['reranked_docs'], ground_truth
        )
        reranking_f1 = self._calculate_f1(
            reranking_scores.get('context_precision', 0),
            reranking_scores.get('context_recall', 0)
        )
        
        # Evaluate generation
        generation_scores = await self.evaluator.evaluate_generation(
            state['query'], state['contexts'], state['answer'], ground_truth
        )
        generation_quality = (
            generation_scores.get('faithfulness', 0) + 
            generation_scores.get('answer_correctness', 0)
        ) / 2
        
        # Overall quality (weighted)
        overall_quality = (
            retrieval_f1 * 0.3 +
            reranking_f1 * 0.2 +
            generation_quality * 0.5
        )
        
        print(f"    Retrieval F1: {retrieval_f1:.3f}")
        print(f"    Reranking F1: {reranking_f1:.3f}")
        print(f"    Generation Quality: {generation_quality:.3f}")
        print(f"    Overall Quality: {overall_quality:.3f}")
        
        return {
            'retrieval_scores': retrieval_scores,
            'reranking_scores': reranking_scores,
            'generation_scores': generation_scores,
            'retrieval_f1': retrieval_f1,
            'reranking_f1': reranking_f1,
            'generation_quality': generation_quality,
            'overall_quality': overall_quality
        }
    
    async def _memory_update_node(self, state: RAGState) -> Dict[str, Any]:
        """
        Node 7: Update memory weights based on execution outcome
        Creates new memory and updates weights of retrieved memories
        """
        print(f"  [Memory Update]")
        
        # Check if we improved
        previous_best = max(
            [h['overall_quality'] for h in state['improvement_history']],
            default=0.0
        )
        current_quality = state['overall_quality']
        improved = current_quality > previous_best
        
        if improved:
            print(f"    ✓ Improved! {previous_best:.3f} → {current_quality:.3f}")
        else:
            print(f"    No improvement: {current_quality:.3f} (best: {previous_best:.3f})")
        
        # Create module-specific memories
        query_planning_mem = ModuleMemory(
            module=ModuleType.QUERY_PLANNING,
            input_data={'query': state['query']},
            output_data={'generated_queries': state['generated_queries']},
            score=state['retrieval_f1'],
            what_worked="Generated queries led to good retrieval" if state['retrieval_f1'] > 0.7 else "",
            what_failed="Queries didn't capture intent well" if state['retrieval_f1'] < 0.5 else ""
        )
        
        reranking_mem = ModuleMemory(
            module=ModuleType.RERANKING,
            input_data={'contexts': state['contexts']},
            output_data={'reranked_docs': state['reranked_docs']},
            score=state['reranking_f1'],
            what_worked="Reranking improved precision" if state['reranking_f1'] > state['retrieval_f1'] else "",
            what_failed="Reranking didn't help" if state['reranking_f1'] <= state['retrieval_f1'] else ""
        )
        
        generation_mem = ModuleMemory(
            module=ModuleType.GENERATION,
            input_data={'query': state['query'], 'contexts': state['reranked_docs']},
            output_data={'answer': state['answer']},
            score=state['generation_quality'],
            what_worked="Answer was faithful and correct" if state['generation_quality'] > 0.8 else "",
            what_failed="Answer lacked grounding or accuracy" if state['generation_quality'] < 0.6 else ""
        )
        
        # Create execution memory
        memory_id = f"mem_{state['iteration']}_{hash(state['query'])}"
        new_memory = ExecutionMemory(
            memory_id=memory_id,
            query=state['query'],
            contexts=state['contexts'],
            answer=state['answer'],
            query_planning_memory=query_planning_mem,
            reranking_memory=reranking_mem,
            generation_memory=generation_mem,
            retrieval_f1=state['retrieval_f1'],
            reranking_f1=state['reranking_f1'],
            generation_quality=state['generation_quality'],
            overall_quality=current_quality,
            weight=1.0 + (0.5 if improved else 0.0)
        )
        
        # Add to memory store
        if self.memory_store:
            self.memory_store.add_memory(new_memory)
            
            # Update weights of retrieved memories
            all_retrieved = []
            for mem_list in state.get('retrieved_memories', {}).values():
                all_retrieved.extend(mem_list)
            
            if all_retrieved:
                self.memory_store.update_memory_weights(
                    retrieved_memories=all_retrieved,
                    outcome={
                        'quality_improved': improved,
                        'retrieval_f1_delta': state['retrieval_f1'] - previous_best * 0.3,
                        'reranking_f1_delta': state['reranking_f1'] - previous_best * 0.2,
                        'generation_quality_delta': state['generation_quality'] - previous_best * 0.5
                    }
                )
        
        # Update improvement history
        improvement_entry = {
            'iteration': state['iteration'] + 1,
            'overall_quality': current_quality,
            'retrieval_f1': state['retrieval_f1'],
            'reranking_f1': state['reranking_f1'],
            'generation_quality': state['generation_quality'],
            'improved': improved
        }
        
        return {
            'iteration': state['iteration'] + 1,
            'improvement_history': [improvement_entry],
            'memory_store': self.memory_store
        }
    
    def _should_continue(self, state: RAGState) -> str:
        """
        Routing logic: Continue iterating or stop?
        
        Stop if:
        - Reached max iterations
        - Quality exceeds threshold (converged)
        - No improvement for 2 consecutive iterations
        """
        iteration = state['iteration']
        max_iterations = state['max_iterations']
        quality = state['overall_quality']
        threshold = state['quality_threshold']
        
        # Check max iterations
        if iteration >= max_iterations:
            print(f"  [Routing] STOP: Reached max iterations ({max_iterations})")
            return "stop"
        
        # Check convergence
        if quality >= threshold:
            print(f"  [Routing] STOP: Converged (quality {quality:.3f} >= {threshold})")
            return "stop"
        
        # Check stagnation (no improvement in last 2 iterations)
        history = state.get('improvement_history', [])
        if len(history) >= 2:
            last_two_improved = [h['improved'] for h in history[-2:]]
            if not any(last_two_improved):
                print(f"  [Routing] STOP: No improvement in last 2 iterations")
                return "stop"
        
        print(f"  [Routing] CONTINUE: Quality {quality:.3f} < {threshold}")
        return "continue"
    
    async def batch_query(
        self,
        dataset: List[Dict[str, str]],
        retrieval_k: int = 20,
        final_k: int = 10,
        max_concurrent: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process multiple queries concurrently"""
        print(f"🚀 Batch processing {len(dataset)} queries in {self.mode.value.upper()} mode")
        print(f"   Concurrency: {max_concurrent}")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        start_time = time.time()
        
        async def process_with_semaphore(item: Dict[str, str]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    result = await self.query(
                        query=item['query'],
                        ground_truth=item.get('ground_truth'),
                        retrieval_k=retrieval_k,
                        final_k=final_k,
                        **kwargs
                    )
                    return result
                except Exception as e:
                    print(f"❌ Error: {e}")
                    return {
                        'error': str(e),
                        'query': item['query'],
                        'ground_truth': item.get('ground_truth')
                    }
        
        tasks = [process_with_semaphore(item) for item in dataset]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'error': str(result),
                    'query': dataset[i]['query'],
                    'ground_truth': dataset[i].get('ground_truth')
                })
            else:
                processed_results.append(result)
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in processed_results if 'error' not in r)
        
        print(f"✅ Batch complete: {success_count}/{len(dataset)} successful in {total_time:.2f}s")
        print(f"📊 Throughput: {len(dataset)/total_time:.2f} queries/second")
        
        return processed_results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example: Using UnifiedRAGPipeline in different modes"""
    
    # Setup (same for all modes)
    documents = [
        {'doc_id': 'doc1', 'text': 'Revenue in 2023 was $10M...'},
        # ... more documents
    ]
    
    evaluation_queries = [
        {
            'query_id': 0,
            'query': 'What was the revenue in 2023?',
            'ground_truth': '$10M',
            'reference_doc_ids': ['doc1'],
            'reference_evidence_texts': [
                {'doc_id': 'doc1', 'evidence_text': 'Revenue in 2023 was $10M'}
            ]
        },
        # ... more queries
    ]
    
    # ===== FAST MODE (Production) =====
    print("\n" + "="*80)
    print("FAST MODE - Single-pass optimized")
    print("="*80)
    
    pipeline_fast = UnifiedRAGPipeline(
        mode=PipelineMode.FAST,
        enable_tracing=True
    )
    pipeline_fast.setup_pipeline(documents, evaluation_queries)
    
    result_fast = await pipeline_fast.query(
        query="What was the revenue in 2023?",
        ground_truth="$10M",
        retrieval_k=20,
        final_k=10
    )
    print(f"Answer: {result_fast['answer']}")
    print(f"Quality: {result_fast['overall_quality']:.3f}")
    
    # ===== ITERATIVE MODE (Experimentation) =====
    print("\n" + "="*80)
    print("ITERATIVE MODE - Multiple iterations with feedback")
    print("="*80)
    
    pipeline_iterative = UnifiedRAGPipeline(
        mode=PipelineMode.ITERATIVE,
        max_iterations=3,
        quality_threshold=0.85
    )
    pipeline_iterative.setup_pipeline(documents, evaluation_queries)
    
    result_iterative = await pipeline_iterative.query(
        query="What was the revenue in 2023?",
        ground_truth="$10M",
        retrieval_k=20,
        final_k=10
    )
    print(f"Answer: {result_iterative['answer']}")
    print(f"Quality: {result_iterative['overall_quality']:.3f}")
    print(f"Iterations: {result_iterative['total_iterations']}")
    print(f"Converged: {result_iterative['converged']}")
    
    # ===== OPTIMIZE MODE (Training) =====
    print("\n" + "="*80)
    print("OPTIMIZE MODE - Full prompt optimization")
    print("="*80)
    
    pipeline_optimize = UnifiedRAGPipeline(
        mode=PipelineMode.OPTIMIZE,
        max_iterations=5
    )
    pipeline_optimize.setup_pipeline(documents, evaluation_queries)
    
    result_optimize = await pipeline_optimize.query(
        query="What was the revenue in 2023?",
        ground_truth="$10M",
        retrieval_k=20,
        final_k=10
    )
    print(f"Answer: {result_optimize.get('answer', 'N/A')}")
    print(f"Quality: {result_optimize.get('overall_quality', 0):.3f}")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(example_usage())

