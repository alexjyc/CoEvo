"""
Self-Improving RAG Pipeline using LangGraph and RAGAS
Implements iterative refinement based on evaluation feedback
"""

import asyncio
from itertools import chain
from typing import Optional, List, Dict, Any, Annotated, Union, Tuple
from typing_extensions import TypedDict as ExtendedTypedDict
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


from LLMBase import LLMBase
from preprocessor import DocumentPreprocessor
from retrieval import Retriever, RetrievalMethod
from reranker import Reranker
from generation import ResponseGenerator
from evaluation import Evaluator
from prompt_optimization import PromptOptimizationCycle


class RAGState(ExtendedTypedDict):
    """State schema for the self-improving RAG pipeline"""
    # Input
    query: str
    ground_truth: Optional[str]
    
    # Query Planning
    sub_queries: Optional[List[str]]
    
    # Retrieval
    retrieved_chunks: Optional[List[Dict[str, Any]]]
    contexts: Optional[List[str]]
    
    # Reranking
    reranked_chunks: Optional[List[Dict[str, Any]]]
    final_contexts: Optional[List[str]]
    
    # Generation
    answer: Optional[str]
    answer_reference: Optional[str]
    generation_rationale: Optional[str]
    
    # Evaluation (RAGAS scores)
    faithfulness_score: Optional[float]
    answer_relevancy_score: Optional[float]
    context_utilization_score: Optional[float]
    context_precision_score: Optional[float]
    context_recall_score: Optional[float]
    retrieval_eval: Optional[Dict[str, float]]
    reranking_eval: Optional[Dict[str, float]]
    generation_eval: Optional[Dict[str, float]]
    overall_score: Optional[float]
    
    # Iteration control
    iteration: int
    max_iterations: int
    converged: bool
    improvement_history: Annotated[List[Dict[str, Any]], operator.add]
    prompt_history: Annotated[List[Dict[str, Any]], operator.add]
    
    # Configuration
    retrieval_k: int
    final_k: int
    convergence_threshold: float
    retrieval_round: int
    rerank_round: int
    generation_round: int


class SelfImprovingRAG:
    """
    Self-improving RAG Pipeline using LangGraph
    
    This pipeline iteratively refines its outputs based on RAGAS evaluation scores,
    adjusting retrieval parameters and query reformulation strategies.
    """
    
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 50,
        generation_model: str = "gpt-4o-mini",
        evaluator_model: str = "gpt-4o-mini",
        temperature: float = 0,
        retrieval_method: str = "hybrid",
        max_iterations: int = 3,
        convergence_threshold: float = 0.85,
        prompt_optimization_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the self-improving RAG pipeline
        
        Args:
            openai_api_key: OpenAI API key
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            generation_model: Model for generation
            evaluator_model: Model for RAGAS evaluation
            temperature: Generation temperature
            retrieval_method: "hybrid", "bm25", or "dense"
            max_iterations: Maximum refinement iterations
            convergence_threshold: Score threshold to stop iterations (0-1)
            prompt_optimization_config: Optional overrides for the RL prompt optimizer
        """
        
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Determine retrieval method
        if retrieval_method == "hybrid":
            self.retrieval_method = RetrievalMethod.HYBRID
        elif retrieval_method == "bm25":
            self.retrieval_method = RetrievalMethod.BM25_ONLY
        elif retrieval_method == "dense":
            self.retrieval_method = RetrievalMethod.DENSE_ONLY
        else:
            raise ValueError(f"Invalid retrieval method: {retrieval_method}")
        
        # Initialize components
        self.preprocessor = DocumentPreprocessor(chunk_size, chunk_overlap)
        self.retriever = None
        self.base_component = LLMBase(model_name=generation_model)
        self.reranker = Reranker(default_k=10)
        self.generator = ResponseGenerator(generation_model, temperature)
        self.evaluator = Evaluator(model=evaluator_model)
        optimizer_config = prompt_optimization_config or {}
        self.prompt_optimizer = PromptOptimizationCycle(**optimizer_config)
        
        self.is_ready = False
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
    
    def load_and_process_documents(self, documents: List[Dict[str, Any]], use_context: bool = True):
        """Load documents and create searchable index"""
        print("Loading and processing documents...")
        self.preprocessor.process_documents(documents, use_context=use_context)
        self.retriever = Retriever(self.preprocessor)
        self.is_ready = True
        print(f"Pipeline ready! Processed {len(documents)} documents.")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with feedback loops"""
        
        # Define the state graph
        graph = StateGraph(RAGState)
        
        # Add nodes
        graph.add_node("initialize", self._initialize_node)
        graph.add_node("reformulate_query", self._reformulate_query_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("evaluate_retrieval", self._evaluate_retrieval_node)
        graph.add_node("rerank", self._rerank_node)
        graph.add_node("evaluate_reranking", self._evaluate_reranking_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("evaluate_generation", self._evaluate_generation_node)
        
        # Add edges
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "reformulate_query")
        graph.add_edge("reformulate_query", "retrieve")
        graph.add_edge("retrieve", "evaluate_retrieval")
        graph.add_edge("rerank", "evaluate_reranking")
        graph.add_edge("generate", "evaluate_generation")

        graph.add_conditional_edges(
            "evaluate_retrieval",
            self._route_after_retrieval_eval,
            {
                "reformulate_query": "reformulate_query",
                "rerank": "rerank"
            }
        )
        graph.add_conditional_edges(
            "evaluate_reranking",
            self._route_after_reranking_eval,
            {
                "rerank": "rerank",
                "generate": "generate"
            }
        )
        graph.add_conditional_edges(
            "evaluate_generation",
            self._route_after_generation_eval,
            {
                "generate": "generate",
                END: END
            }
        )
        
        return graph
    
    def _initialize_node(self, state: RAGState) -> Dict[str, Any]:
        """Initialize the iteration state"""
        return {
            "iteration": 0,
            "converged": False,
            "improvement_history": [],
            "max_iterations": state.get("max_iterations", self.max_iterations),
            "convergence_threshold": state.get("convergence_threshold", self.convergence_threshold),
            "retrieval_round": 0,
            "rerank_round": 0,
            "generation_round": 0
        }
    
    def _query_planner_node(self, state: RAGState) -> Dict[str, Any]:
        """Reformulate the query, potentially adjusting based on previous feedback"""
        print(f"\n[Iteration {state['iteration']}] Reformulating query...")
        
        query = state["query"]
        print(f"  Original Query: {query}")
        
        response = self.base_component.query_planner(query)
        print(f" Query Sub-Queries: {response.sub_queries}")
        
        return {
            "sub_queries": response.sub_queries
        }
    
    async def _retrieve_node(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve documents, potentially adjusting k based on feedback"""
        print(f"[Iteration {state['iteration']}] Retrieving documents...")
        
        original_query = state["query"]
        sub_queries = state["sub_queries"]
        retrieval_k = state["retrieval_k"]
        final_query = [original_query] + sub_queries
        
        async def fetch(query_text):
            result = await asyncio.to_thread(
                self.retriever.retrieve_hybrid,
                query_text,
                retrieval_k,
            )

            chunks, details = self.reranker.optimize_fusion_params(
                query=query_text,
                bm25_results=result.get("bm25", []),
                dense_results=result.get("dense", []),
                top_k=retrieval_k,
            )
            
            return {
                "chunks": chunks,
                "contexts": [c["chunk_text"] for c in chunks],
                "weight_details": details,
            }

        sub_query_results = {}
        tasks = [fetch(fq) for fq in final_query]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for fq, output in zip(final_query, results):
            if isinstance(output, Exception):
                print(f"  ⚠ Retrieval failed for '{fq}': {output}")
                continue
            sub_query_results[fq] = output

        # TODO: find a optimal way to aggregate chunks from different sub-queries and next steps. 
        # Current thought, either dive deeper into the sub-queries if more context is needed or aggregate reranking.
        # How to decide which one to do? For ragas eval, calling generation this step with aggregated chunks good indicator?
        combined_chunks = list(chain.from_iterable(
            data["chunks"] for data in sub_query_results.values()
        ))
        combined_contexts = [chunk["chunk_text"] for chunk in combined_chunks]

        return {
            "sub_query_results": sub_query_results,
            "retrieved_chunks": combined_chunks,
            "contexts": combined_contexts,
            "retrieval_method": self.retrieval_method.value,
        }
    
    def _rerank_node(self, state: RAGState) -> Dict[str, Any]:
        """Rerank retrieved documents"""
        print(f"[Iteration {state['iteration']}] Reranking documents...")
        
        final_k = state["final_k"]
        retrieved_chunks = state.get("retrieved_chunks") or []
        
        processed_chunks = self.reranker.process_chunks(
            retrieved_chunks,
            k=final_k
        )
        
        final_contexts = [chunk['chunk_text'] for chunk in processed_chunks]
        reranking_rationale = None
        if final_contexts:
            prompt_text, prompt_record = self.prompt_optimizer.select_prompt(
                stage="reranking",
                iteration_id=state["iteration"],
                context={"query": state["query"], "num_candidates": len(final_contexts)}
            )
            try:
                llm_response = self.base_component.rerank_documents(
                    state["query"],
                    final_contexts,
                    prompt_override=prompt_text
                )
                reranking_rationale = llm_response.rationale
                prompt_record["rationale"] = llm_response.rationale
                ranked_texts = llm_response.ranked_documents
                ordered_chunks: List[Dict[str, Any]] = []
                remaining = processed_chunks.copy()
                for text in ranked_texts:
                    match = next((chunk for chunk in remaining if chunk['chunk_text'] == text), None)
                    if match:
                        ordered_chunks.append(match)
                        remaining.remove(match)
                ordered_chunks.extend(remaining)
                processed_chunks = ordered_chunks[:final_k]
                final_contexts = [chunk['chunk_text'] for chunk in processed_chunks]
            except Exception as err:
                reranking_rationale = f"Prompt-based reranker failed: {err}"
                prompt_record["error"] = str(err)
        
        return {
            "reranked_chunks": processed_chunks,
            "final_contexts": final_contexts,
            "reranking_rationale": reranking_rationale
        }
    
    def _generate_node(self, state: RAGState) -> Dict[str, Any]:
        """Generate answer from contexts"""
        print(f"[Iteration {state['iteration']}] Generating answer...")
        
        rerank_eval = state.get("reranking_eval") or {}
        prompt_text, prompt_record = self.prompt_optimizer.select_prompt(
            stage="generation",
            iteration_id=state["iteration"],
            context={
                "query": state["query"],
                "num_contexts": len(state.get("final_contexts") or []),
                "rerank_confidence": rerank_eval.get("reranking_confidence", 0.0)
            }
        )
        contexts_to_use = state.get("final_contexts") or []
        response = self.base_component.generate_answer(
            state["reformulated_query"],
            "\n\n".join(contexts_to_use),
            prompt_override=prompt_text
        )
        prompt_record["rationale"] = response.rationale

        return {
            "answer": response.answer,
            "answer_reference": response.reference,
            "generation_rationale": response.rationale
        }

    def _stage_memory(self, stage: str, state_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Stage-specific LangGraph memory for agentic rewards."""
        history = state_snapshot.get("improvement_history") or []
        return [entry for entry in history if entry.get("stage") == stage]

    def _finalize_prompt_stage(
        self,
        stage: str,
        state_snapshot: Dict[str, Any],
        metric_overrides: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Finalize RL updates for a specific stage with optional metric overrides."""
        snapshot = dict(state_snapshot)
        if metric_overrides:
            snapshot.update(metric_overrides)
        stage_memory = self._stage_memory(stage, snapshot)
        prompt_records = self.prompt_optimizer.finalize_iteration(snapshot, stages=[stage], memory=stage_memory)
        if prompt_records:
            print(f"  Prompt RL updates ({stage}):")
            for record in prompt_records:
                reward = record.get("reward") or 0.0
                normalized = record.get("normalized_reward") or 0.0
                print(f"    [{record['stage']}] {record['prompt_name']} → reward {reward:.3f}, norm {normalized:.3f}")
        improvements: List[Dict[str, Any]] = []
        for record in prompt_records or []:
            improvements.append({
                "stage": stage,
                "iteration": snapshot.get("iteration", 0),
                "prompt_name": record.get("prompt_name"),
                "reward": record.get("reward"),
                "normalized_reward": record.get("normalized_reward"),
                "metrics_snapshot": record.get("metrics_snapshot"),
                "context": record.get("context", {}),
            })
        return prompt_records or [], improvements
    
    async def _evaluate_retrieval_node(self, state: RAGState) -> Dict[str, Any]:
        """Evaluate retrieval quality before reranking."""
        print(f"[Iteration {state['iteration']}] Evaluating retrieval step...")
    
        retrieved_chunks = state.get("retrieved_chunks") or []
        
        if not retrieved_chunks:
            print("  No contexts retrieved, skipping retrieval evaluation.")
            round_idx = state.get("retrieval_round", 0) + 1
            return {"retrieval_eval": None, "retrieval_round": round_idx}
        
        # Use RAGAS to evaluate retrieval (context_precision & context_recall)
        contexts = [chunk['chunk_text'] for chunk in retrieved_chunks]
        scores = await self.evaluator.evaluate_metrics(
            metric_keys=['context_precision', 'context_recall'],
            query=state["query"],
            contexts=contexts,
            ground_truth=state.get("ground_truth")
        )
        
        print(f"  Retrieval RAGAS metrics: precision={scores.get('context_precision', 0.0):.3f}, "
              f"recall={scores.get('context_recall', 0.0):.3f}")
        
        round_idx = state.get("retrieval_round", 0) + 1
        
        prompt_records, improvements = self._finalize_prompt_stage(
            stage="reformulation",
            state_snapshot=state,
            metric_overrides={
                "retrieval_eval": scores,
                "context_precision_score": scores.get('context_precision'),
                "context_recall_score": scores.get('context_recall')
            }
        )
        
        return {
            "retrieval_eval": scores,
            "retrieval_round": round_idx,
            "prompt_history": prompt_records or [],
            "improvement_history": improvements or []
        }

    async def _evaluate_reranking_node(self, state: RAGState) -> Dict[str, Any]:
        """Evaluate reranking quality using score comparison (no LLM calls!)"""
        print(f"[Iteration {state['iteration']}] Evaluating reranking step...")
        
        reranked_chunks = state.get("reranked_chunks") or []
        if not reranked_chunks:
            print("  No reranked contexts, skipping reranking evaluation.")
            round_idx = state.get("rerank_round", 0) + 1
            return {"reranking_eval": None, "rerank_round": round_idx}
        
        contexts = [chunk['chunk_text'] for chunk in reranked_chunks]
        score = await self.evaluator.evaluate_reranking(
            query=state["query"],
            contexts=contexts,
            ground_truth=state.get("ground_truth")
        )
        
        round_idx = state.get("rerank_round", 0) + 1
        
        prompt_records, improvements = self._finalize_prompt_stage(
            stage="reranking",
            state_snapshot=state,
            metric_overrides={
                "reranking_eval": score,
                "context_precision_score": score.get('context_precision'),
            }
        )
        
        return {
            "reranking_eval": score,
            "rerank_round": round_idx,
            "prompt_history": prompt_records or [],
            "improvement_history": improvements or []
        }

    async def _evaluate_generation_node(self, state: RAGState) -> Dict[str, Any]:
        """Evaluate the generated answer using RAGAS generation metrics"""
        print(f"[Iteration {state['iteration']}] Evaluating generation output...")
        
        # Use RAGAS to evaluate generation (faithfulness, answer_relevancy, context_utilization)
        scores = await self.evaluator.evaluate_generation(
            query=state["query"],
            contexts=state.get("final_contexts") or [],
            answer=state.get("answer") or "",
            ground_truth=state.get("ground_truth")   
        )
        
        round_idx = state.get("generation_round", 0) + 1
        
        prompt_records, improvements = self._finalize_prompt_stage(
            stage="generation",
            state_snapshot=state,
            metric_overrides={
                "generation_eval": scores,
                "faithfulness_score": scores.get('faithfulness'),
                "answer_correctness_score": scores.get('answer_correctness'),
            }
        )
        
        return {
            "generation_eval": scores,
            "generation_round": round_idx,
            "prompt_history": prompt_records or [],
            "improvement_history": improvements or []
        }
    
    
    def _route_after_retrieval_eval(self, state: RAGState) -> str:
        """Decide whether to rerank or repeat reformulation based on RAGAS retrieval metrics"""
        # Thresholds for RAGAS retrieval metrics
        PRECISION_THRESHOLD = 0.60  # Context precision (relevant contexts ranked high)
        RECALL_THRESHOLD = 0.50     # Context recall (coverage of relevant info)
        COMBINED_THRESHOLD = 0.55   # Average of both

        rounds = state.get("retrieval_round", 0)
        max_rounds = state.get("max_iterations", self.max_iterations)
        metrics = state.get("retrieval_eval") or {}
        
        # Get RAGAS retrieval metrics
        context_precision = metrics.get("context_precision", 0.0)
        context_recall = metrics.get("context_recall", 0.0)
        combined_score = (context_precision + context_recall) / 2

        if rounds >= max_rounds:
            print(f"✓ Max retrieval rounds ({rounds}/{max_rounds}); moving to rerank")
            return "rerank"
        
        # Primary check: combined retrieval quality
        if combined_score >= COMBINED_THRESHOLD:
            print(f"✓ Retrieval acceptable (precision={context_precision:.3f}, "
                  f"recall={context_recall:.3f}, combined={combined_score:.3f}); continuing to rerank")
            return "rerank"
        
        # Check precision (relevant contexts ranked highly)
        if context_precision < PRECISION_THRESHOLD:
            print(f"↻ Context precision too low ({context_precision:.3f}); retrying reformulation")
            return "reformulate_query"
        
        # Check recall (coverage of relevant information)
        if context_recall < RECALL_THRESHOLD:
            print(f"↻ Context recall too low ({context_recall:.3f}); retrying reformulation")
            return "reformulate_query"
        
        # Marginal but proceed
        print(f"⚠️  Retrieval marginal (combined={combined_score:.3f}); proceeding to rerank")
        return "rerank"
    
    def _route_after_reranking_eval(self, state: RAGState) -> str:
        """Decide whether to regenerate contexts or proceed to answer generation."""
        # Thresholds for reranking quality (using score-based metrics)
        RERANKING_CONFIDENCE_THRESHOLD = 0.70
        CONCENTRATION_MIN = 0.50  # How much better are top results
        IMPROVEMENT_MIN = 0.05  # Minimum improvement from reranking

        rounds = state.get("rerank_round", 0)
        max_rounds = state.get("max_iterations", self.max_iterations)
        metrics = state.get("reranking_eval") or {}

        # Get reranking quality metrics
        reranking_confidence = metrics.get("reranking_confidence", 0.0)
        score_concentration = metrics.get("score_concentration", 0.0)
        score_improvement = metrics.get("score_improvement", 0.0)

        if rounds >= max_rounds:
            print(f"✓ Max reranking rounds ({rounds}/{max_rounds}); moving to generation")
            return "generate"
        
        # Primary check: overall reranking confidence
        if reranking_confidence >= RERANKING_CONFIDENCE_THRESHOLD:
            print(f"✓ Reranking acceptable (confidence={reranking_confidence:.3f}); continuing to generation")
            return "generate"
        
        # Check if reranking actually improved things
        if score_improvement < IMPROVEMENT_MIN:
            print(f"↻ Reranking didn't improve scores ({score_improvement:.3f}); retrying")
            return "rerank"
        
        # Check if top results are clearly better
        if score_concentration < CONCENTRATION_MIN:
            print(f"↻ Top results not clearly better ({score_concentration:.3f}); retrying")
            return "rerank"
        
        # Marginal but proceed
        print(f"⚠️  Reranking marginal (confidence={reranking_confidence:.3f}); proceeding to generation")
        return "generate"
    
    def _route_after_generation_eval(self, state: RAGState) -> str:
        """Decide whether to regenerate answer based on RAGAS generation metrics"""
        # Thresholds for RAGAS generation metrics
        FAITHFULNESS_THRESHOLD = 0.75      # Answer grounded in context
        ANSWER_RELEVANCY_THRESHOLD = 0.70  # Answer relevance to query
        OVERALL_THRESHOLD = 0.70           # Combined threshold

        rounds = state.get("generation_round", 0)
        max_rounds = state.get("max_iterations", self.max_iterations)
        overall = state.get("overall_score", 0.0)
        
        # Get individual RAGAS generation metrics
        faithfulness = state.get("faithfulness_score", 0.0)
        answer_relevancy = state.get("answer_relevancy_score", 0.0)

        if rounds >= max_rounds:
            print(f"✓ Generation reached max iterations ({rounds}/{max_rounds}); finishing")
            return END
        
        if overall >= OVERALL_THRESHOLD:
            print(f"✓ Generation met quality threshold (faithfulness={faithfulness:.3f}, "
                  f"relevancy={answer_relevancy:.3f}, overall={overall:.3f}); finishing")
            return END
        
        # Check if faithfulness is critically low
        if faithfulness < FAITHFULNESS_THRESHOLD:
            print(f"↻ Faithfulness too low ({faithfulness:.3f}); regenerating")
            return "generate"
        
        # Check if answer relevancy is low
        if answer_relevancy < ANSWER_RELEVANCY_THRESHOLD:
            print(f"↻ Answer relevancy too low ({answer_relevancy:.3f}); regenerating")
            return "generate"
        
        print(f"⚠️  Generation below threshold (overall={overall:.3f}); finishing anyway")
        return END
    
    
    def _build_initial_state(
        self,
        query: str,
        ground_truth: Optional[str],
        retrieval_k: int,
        final_k: int,
        max_iterations: Optional[int],
        convergence_threshold: Optional[float],
    ) -> RAGState:
        """Helper to construct a fresh state for each graph run."""
        return {
            "query": query,
            "ground_truth": ground_truth,
            "retrieval_k": retrieval_k,
            "final_k": final_k,
            "max_iterations": max_iterations or self.max_iterations,
            "convergence_threshold": convergence_threshold or self.convergence_threshold,
            "iteration": 0,
            "converged": False,
            "improvement_history": [],
            # Optional fields initialized as None
            "reformulated_query": None,
            "reformation_rationale": None,
            "retrieved_chunks": None,
            "contexts": None,
            "retrieval_method": None,
            "reranked_chunks": None,
            "final_contexts": None,
            "reranking_rationale": None,
            "answer": None,
            "answer_reference": None,
            "generation_rationale": None,
            "faithfulness_score": None,
            "answer_relevancy_score": None,
            "context_utilization_score": None,
            "context_precision_score": None,
            "context_recall_score": None,
            "overall_score": None,
            "retrieval_eval": None,
            "reranking_eval": None,
            "generation_eval": None,
            "prompt_history": [],
            "retrieval_round": 0,
            "rerank_round": 0,
            "generation_round": 0,
        }
    
    async def _run_compiled_graph(
        self,
        compiled_graph,
        initial_state: RAGState,
        thread_suffix: str = "",
    ) -> Dict[str, Any]:
        """Execute a compiled graph run and extract final outputs."""
        self.prompt_optimizer.reset_active_records()

        thread_id = f"query-{hash(initial_state['query'])}{thread_suffix}"
        config = {"configurable": {"thread_id": thread_id}}
        final_state = None

        print(f"\n{'='*80}")
        print(f"Processing query: {initial_state['query']}")
        print(f"{'='*80}")
        
        async for state_update in compiled_graph.astream(initial_state, config=config):
            final_state = state_update

        if final_state and isinstance(final_state, dict):
            last_node_key = list(final_state.keys())[-1]
            final_result = final_state[last_node_key]
            
            print(f"\n{'='*80}")
            print("FINAL RESULTS")
            print(f"{'='*80}")
            print(f"Iterations: {len(final_result.get('improvement_history', []))}")
            print(f"Final Answer: {final_result.get('answer', 'N/A')}")
            print(f"Overall Score: {final_result.get('overall_score', 0.0):.3f}")
            print(f"{'='*80}\n")
            
            return final_result
        
        return {}

    async def query(
        self,
        query: str,
        ground_truth: Optional[str] = None,
        retrieval_k: int = 20,
        final_k: int = 10,
        max_iterations: Optional[int] = None,
        convergence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the self-improving pipeline
        
        Args:
            query: User's question
            ground_truth: Optional ground truth answer for evaluation
            retrieval_k: Initial number of chunks to retrieve
            final_k: Initial number of chunks for generation
            max_iterations: Override default max iterations
            convergence_threshold: Override default convergence threshold
        
        Returns:
            Final state with answer and evaluation history
        """
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Load documents first.")
        
        initial_state = self._build_initial_state(
            query=query,
            ground_truth=ground_truth,
            retrieval_k=retrieval_k,
            final_k=final_k,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
        )
        
        checkpointer = MemorySaver()
        compiled_graph = self.graph.compile(checkpointer=checkpointer)
        
        return await self._run_compiled_graph(
            compiled_graph=compiled_graph,
            initial_state=initial_state,
            thread_suffix=""
        )
    
    def query_sync(
        self,
        query: str,
        ground_truth: Optional[str] = None,
        retrieval_k: int = 20,
        final_k: int = 10,
        max_iterations: Optional[int] = None,
        convergence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for query() - useful for notebooks and scripts
        
        Args:
            Same as query()
        
        Returns:
            Final state with answer and evaluation history
        """
        return asyncio.run(self.query(
            query=query,
            ground_truth=ground_truth,
            retrieval_k=retrieval_k,
            final_k=final_k,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold
        ))

    async def query_batch(
        self,
        queries: List[Union[str, Dict[str, Any]]],
        retrieval_k: int = 20,
        final_k: int = 10,
        max_iterations: Optional[int] = None,
        convergence_threshold: Optional[float] = None,
        concurrency: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries with optional limited concurrency.
        
        Args:
            queries: List of query strings or dicts with keys 'query' and optional 'ground_truth'.
            retrieval_k: Initial number of chunks to retrieve for each query.
            final_k: Initial number of chunks for generation for each query.
            max_iterations: Override default max iterations.
            convergence_threshold: Override default convergence threshold.
            concurrency: Max number of simultaneous graph runs (sharing the same prompt optimizer).
        
        Returns:
            List of final states in the same order as the input queries.
        """
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Load documents first.")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if not queries:
            return []

        checkpointer = MemorySaver()
        compiled_graph = self.graph.compile(checkpointer=checkpointer)
        semaphore = asyncio.Semaphore(concurrency)
        results: List[Dict[str, Any]] = [{} for _ in queries]

        async def _run_single(idx: int, item: Union[str, Dict[str, Any]]) -> None:
            async with semaphore:
                if isinstance(item, dict):
                    query_text = item.get("query")
                    ground_truth_val = item.get("ground_truth")
                else:
                    query_text = str(item)
                    ground_truth_val = None

                if not query_text:
                    results[idx] = {}
                    return

                initial_state = self._build_initial_state(
                    query=query_text,
                    ground_truth=ground_truth_val,
                    retrieval_k=retrieval_k,
                    final_k=final_k,
                    max_iterations=max_iterations,
                    convergence_threshold=convergence_threshold,
                )

                results[idx] = await self._run_compiled_graph(
                    compiled_graph=compiled_graph,
                    initial_state=initial_state,
                    thread_suffix=f"-batch-{idx}"
                )

        await asyncio.gather(*[
            _run_single(idx, item) for idx, item in enumerate(queries)
        ])

        return results

    def query_batch_sync(
        self,
        queries: List[Union[str, Dict[str, Any]]],
        retrieval_k: int = 20,
        final_k: int = 10,
        max_iterations: Optional[int] = None,
        convergence_threshold: Optional[float] = None,
        concurrency: int = 1
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for query_batch()."""
        return asyncio.run(self.query_batch(
            queries=queries,
            retrieval_k=retrieval_k,
            final_k=final_k,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            concurrency=concurrency
        ))
