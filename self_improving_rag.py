"""
Self-Improving RAG Pipeline using LangGraph and RAGAS
Implements iterative refinement based on evaluation feedback
"""

import asyncio
from typing import Optional, List, Dict, Any, Annotated
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
    
    # Reformation
    reformulated_query: Optional[str]
    reformation_rationale: Optional[str]
    
    # Retrieval
    retrieved_chunks: Optional[List[Dict[str, Any]]]
    contexts: Optional[List[str]]
    retrieval_method: Optional[str]
    
    # Reranking
    reranked_chunks: Optional[List[Dict[str, Any]]]
    final_contexts: Optional[List[str]]
    reranking_rationale: Optional[str]
    
    # Generation
    answer: Optional[str]
    answer_reference: Optional[str]
    generation_rationale: Optional[str]
    
    # Evaluation (RAGAS scores)
    faithfulness_score: Optional[float]
    answer_accuracy_score: Optional[float]
    answer_relevancy_score: Optional[float]
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
        generation_model: str = "gemini-2.5-flash",
        evaluator_model: str = "gemini-2.5-flash",
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
    
    def _reformulate_query_node(self, state: RAGState) -> Dict[str, Any]:
        """Reformulate the query, potentially adjusting based on previous feedback"""
        print(f"\n[Iteration {state['iteration']}] Reformulating query...")
        
        query = state["query"]
        print(f"  Original Query: {query}")
        prompt_text, prompt_record = self.prompt_optimizer.select_prompt(
            stage="reformulation",
            iteration_id=state["iteration"],
            context={"query": query}
        )
        
        # On subsequent iterations, use feedback to guide reformulation
        # latest_retrieval_eval = state.get("retrieval_eval") or {}
        # precision_signal = latest_retrieval_eval.get("context_precision")
        # recall_signal = latest_retrieval_eval.get("context_recall")
        # if precision_signal is not None and precision_signal < 0.7:
        #     print("  → Adjusting query to improve context precision")
        # elif recall_signal is not None and recall_signal < 0.7:
        #     print("  → Broadening query to improve recall")
        
        response = self.base_component.reformate_query(query, prompt_override=prompt_text)
        print(f"  Reformulated Query: {response.reformatted_query}")
        prompt_record["model_output"] = response.reformatted_query
        prompt_record["rationale"] = response.rationale
        
        return {
            "reformulated_query": response.reformatted_query,
            "reformation_rationale": response.rationale
        }
    
    def _retrieve_node(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve documents, potentially adjusting k based on feedback"""
        print(f"[Iteration {state['iteration']}] Retrieving documents...")
        
        effective_query = state["reformulated_query"]
        retrieval_k = state["retrieval_k"]
        
        # Adjust retrieval_k based on previous retrieval evaluation
        # precision_signal = (state.get("retrieval_eval") or {}).get("context_precision")
        # recall_signal = (state.get("retrieval_eval") or {}).get("context_recall")
        # if state.get("retrieval_round", 0) > 0:
        #     if precision_signal is not None and precision_signal < 0.7:
        #         retrieval_k = min(retrieval_k + 10, 100)
        #         print(f"  → Increasing retrieval_k to {retrieval_k} for better precision")
        #     elif recall_signal is not None and recall_signal < 0.7 and retrieval_k > 5:
        #         retrieval_k = max(retrieval_k - 5, 5)
        #         print(f"  → Decreasing retrieval_k to {retrieval_k} to focus on top matches")
        
        # Perform retrieval
        retrieval_result = self.retriever.retrieve(
            effective_query,
            retrieval_k,
            method=self.retrieval_method
        )
        
        # Handle hybrid vs single method results
        if isinstance(retrieval_result, dict) and self.retrieval_method == RetrievalMethod.HYBRID:
            retrieved_chunks = self.reranker.rank_fusion(
                bm25_results=retrieval_result.get('bm25', []),
                dense_results=retrieval_result.get('dense', []),
                k=retrieval_k
            )
        else:
            retrieved_chunks = retrieval_result
        
        contexts = [chunk['chunk_text'] for chunk in retrieved_chunks]
        
        return {
            "retrieved_chunks": retrieved_chunks,
            "contexts": contexts,
            "retrieval_method": self.retrieval_method.value
        }
    
    def _rerank_node(self, state: RAGState) -> Dict[str, Any]:
        """Rerank retrieved documents"""
        print(f"[Iteration {state['iteration']}] Reranking documents...")
        
        final_k = state["final_k"]
        retrieved_chunks = state.get("retrieved_chunks") or []
        
        # retrieval_eval = state.get("retrieval_eval") or {}
        # precision_signal = retrieval_eval.get("context_precision")
        # if precision_signal is not None and precision_signal < 0.65 and retrieved_chunks:
        #     final_k = min(final_k + 2, len(retrieved_chunks))
        #     print(f"  → Retrieval precision low ({precision_signal:.2f}); expanding final_k to {final_k}")
        
        # # Adjust final_k based on reranking feedback
        # rerank_eval = state.get("reranking_eval") or {}
        # precision_signal = rerank_eval.get("context_precision")
        # if state.get("rerank_round", 0) > 0 and precision_signal is not None and precision_signal < 0.75 and retrieved_chunks:
        #     final_k = min(final_k + 1, len(retrieved_chunks))
        #     print(f"  → Rerank precision low ({precision_signal:.2f}); expanding final_k to {final_k}")
        # last_generation_eval = state.get("generation_eval") or {}
        # if last_generation_eval.get("faithfulness") is not None and last_generation_eval["faithfulness"] < 0.8:
        #     final_k = max(final_k - 1, 3)
        #     print(f"  → Faithfulness low ({last_generation_eval['faithfulness']:.2f}); narrowing final_k to {final_k}")
        
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
    
    def _finalize_prompt_stage(
        self,
        stage: str,
        state_snapshot: Dict[str, Any],
        metric_overrides: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Finalize RL updates for a specific stage with optional metric overrides."""
        snapshot = dict(state_snapshot)
        if metric_overrides:
            snapshot.update(metric_overrides)
        prompt_records = self.prompt_optimizer.finalize_iteration(snapshot, stages=[stage])
        if prompt_records:
            print(f"  Prompt RL updates ({stage}):")
            for record in prompt_records:
                reward = record.get("reward") or 0.0
                normalized = record.get("normalized_reward") or 0.0
                print(f"    [{record['stage']}] {record['prompt_name']} → reward {reward:.3f}, norm {normalized:.3f}")
        return prompt_records
    
    async def _evaluate_retrieval_node(self, state: RAGState) -> Dict[str, Any]:
        """Evaluate retrieval quality before reranking."""
        print(f"[Iteration {state['iteration']}] Evaluating retrieval step...")
    
        retrieved_chunks = state.get("retrieved_chunks") or []
        
        if not retrieved_chunks:
            print("  No contexts retrieved, skipping retrieval evaluation.")
            round_idx = state.get("retrieval_round", 0) + 1
            return {"retrieval_eval": None, "retrieval_round": round_idx}
        
        # Compute retrieval quality from existing scores (instant, no API calls!)
        scores = self._compute_retrieval_quality(retrieved_chunks)
        print(f"  Retrieval metrics: confidence={scores['retrieval_confidence']:.3f}, "
              f"top={scores['top_score']:.3f}, avg={scores['avg_top_score']:.3f}, "
              f"agreement={scores['bm25_dense_agreement']:.3f}")
        
        round_idx = state.get("retrieval_round", 0) + 1
        stage_overall = scores['retrieval_confidence']  # Use confidence as overall score
        
        prompt_records = self._finalize_prompt_stage(
            stage="reformulation",
            state_snapshot=state,
            metric_overrides={
                "retrieval_eval": scores,
                "overall_score": stage_overall
            }
        )
        
        return {
            "retrieval_eval": scores,
            "retrieval_round": round_idx,
            "prompt_history": prompt_records or []
        }

    async def _evaluate_reranking_node(self, state: RAGState) -> Dict[str, Any]:
        """Evaluate reranking quality using score comparison (no LLM calls!)"""
        print(f"[Iteration {state['iteration']}] Evaluating reranking step...")
        
        reranked_chunks = state.get("reranked_chunks") or []
        if not reranked_chunks:
            print("  No reranked contexts, skipping reranking evaluation.")
            round_idx = state.get("rerank_round", 0) + 1
            return {"reranking_eval": None, "rerank_round": round_idx}
        
        # Compute reranking quality by comparing before/after (instant!)
        original_chunks = state.get("retrieved_chunks") or []
        scores = self._compute_reranking_quality(reranked_chunks, original_chunks)
        print(f"  Reranking metrics: confidence={scores['reranking_confidence']:.3f}, "
              f"improvement={scores['score_improvement']:.3f}, "
              f"concentration={scores['score_concentration']:.3f}")
        
        round_idx = state.get("rerank_round", 0) + 1
        stage_overall = scores['reranking_confidence']  # Use confidence as overall score
        
        prompt_records = self._finalize_prompt_stage(
            stage="reranking",
            state_snapshot=state,
            metric_overrides={
                "reranking_eval": scores,
                "overall_score": stage_overall
            }
        )
        
        return {
            "reranking_eval": scores,
            "rerank_round": round_idx,
            "prompt_history": prompt_records or []
        }

    async def _evaluate_generation_node(self, state: RAGState) -> Dict[str, Any]:
        """Evaluate the generated answer using the full RAGAS suite."""
        print(f"[Iteration {state['iteration']}] Evaluating generation output...")
        scores = await self.evaluator.evaluate_trace(
            query=state["query"],
            contexts=state.get("final_contexts") or [],
            answer=state.get("answer") or "",
            ground_truth=state.get("ground_truth")
        )
        overall = self._compute_overall_score(scores)
        round_idx = state.get("generation_round", 0) + 1
        history_entry = {
            "iteration": round_idx - 1,
            "faithfulness_score": scores.get('faithfulness'),
            "answer_accuracy_score": scores.get('answer_accuracy'),
            "context_precision_score": scores.get('context_precision'),
            "context_recall_score": scores.get('context_recall'),
            "overall_score": overall,
            "retrieval_eval": state.get("retrieval_eval"),
            "reranking_eval": state.get("reranking_eval"),
            "generation_eval": scores,
            "answer": state.get("answer"),
            "num_contexts": len(state.get("final_contexts", []))
        }
        prompt_records = self._finalize_prompt_stage(
            stage="generation",
            state_snapshot=state,
            metric_overrides={
                "generation_eval": scores,
                "faithfulness_score": scores.get('faithfulness'),
                "answer_accuracy_score": scores.get('answer_accuracy'),
                "context_precision_score": scores.get('context_precision'),
                "context_recall_score": scores.get('context_recall'),
                "overall_score": overall,
            }
        )
        return {
            "generation_eval": scores,
            "faithfulness_score": scores.get('faithfulness'),
            "answer_accuracy_score": scores.get('answer_accuracy'),
            "context_precision_score": scores.get('context_precision'),
            "context_recall_score": scores.get('context_recall'),
            "overall_score": overall,
            "generation_round": round_idx,
            "iteration": state.get("iteration", 0) + 1,
            "improvement_history": [history_entry],
            "prompt_history": prompt_records or []
        }
    
    def _compute_overall_score(self, scores: Dict[str, Optional[float]]) -> float:
        """Aggregate the provided metrics into a single score."""
        valid_scores = [score for score in scores.values() if isinstance(score, (int, float))]
        if not valid_scores:
            return 0.0
        return sum(valid_scores) / len(valid_scores)
    
    def _compute_retrieval_quality(self, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute retrieval quality from existing chunk scores (no LLM calls!)
        
        Returns metrics based on:
        - Score distribution and quality
        - BM25 vs Dense agreement
        - Score drop-off and concentration
        """
        if not chunks:
            return {
                "top_score": 0.0,
                "avg_top_score": 0.0,
                "bm25_dense_agreement": 0.5,
                "score_drop_off": 0.5,
                "retrieval_confidence": 0.0
            }
        
        # Extract and CLAMP scores to prevent negative values
        similarity_scores = [max(0.0, min(1.0, c.get('similarity_score', 0.0))) for c in chunks]
        bm25_scores = [max(0.0, c.get('bm25_score', 0.0)) for c in chunks]
        dense_scores = [max(0.0, c.get('dense_score', 0.0)) for c in chunks]
        
        # 1. Top score quality (already clamped)
        top_score = similarity_scores[0] if similarity_scores else 0.0
        
        # 2. Average score of top-k
        top_k = min(10, len(similarity_scores))
        avg_top_score = sum(similarity_scores[:top_k]) / top_k if top_k > 0 else 0.0
        
        # 3. Score drop-off
        if len(similarity_scores) > 3:
            top3_avg = sum(similarity_scores[:3]) / 3
            rest_avg = sum(similarity_scores[3:]) / len(similarity_scores[3:])
            if top3_avg > 1e-6:  # Safety check
                drop_off = (top3_avg - rest_avg) / top3_avg
                drop_off_score = max(0.0, min(1.0, drop_off))
            else:
                drop_off_score = 0.5
        else:
            drop_off_score = 0.5
        
        # 4. BM25-Dense agreement with SAFETY CHECKS
        agreement = 0.5  # Default neutral
        
        if len(bm25_scores) >= top_k and len(dense_scores) >= top_k:
            max_bm25 = max(bm25_scores[:top_k])
            max_dense = max(dense_scores[:top_k])
            
            # Only compute if we have valid scores (not near zero)
            if max_bm25 > 1e-6 and max_dense > 1e-6:
                norm_bm25 = [s / max_bm25 for s in bm25_scores[:top_k]]
                norm_dense = [s / max_dense for s in dense_scores[:top_k]]
                
                # Compute average absolute difference
                avg_diff = sum(abs(b - d) for b, d in zip(norm_bm25, norm_dense)) / len(norm_bm25)
                agreement = max(0.0, min(1.0, 1.0 - avg_diff))
        
        # 5. Overall confidence with FINAL CLAMPING
        retrieval_confidence = (
            top_score * 0.25 +
            avg_top_score * 0.30 +
            agreement * 0.25 +
            drop_off_score * 0.20
        )
        retrieval_confidence = max(0.0, min(1.0, retrieval_confidence))
        
        return {
            "top_score": round(top_score, 3),
            "avg_top_score": round(avg_top_score, 3),
            "bm25_dense_agreement": round(agreement, 3),
            "score_drop_off": round(drop_off_score, 3),
            "retrieval_confidence": round(retrieval_confidence, 3)
        }
    
    def _compute_reranking_quality(
        self, 
        reranked_chunks: List[Dict[str, Any]], 
        original_chunks: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute reranking quality by comparing before/after (no LLM calls!)
        
        Returns metrics based on:
        - Score improvement from reranking
        - Score concentration in top results
        - Rank changes
        """
        if not reranked_chunks:
            return {
                "top_reranked_score": 0.0,
                "avg_reranked_score": 0.0,
                "score_concentration": 0.5,
                "score_improvement": 0.5,
                "reranking_confidence": 0.0
            }
        
        # Extract scores after reranking
        reranked_scores = [
            max(0.0, min(1.0, c.get('cross_encoder_score', c.get('similarity_score', 0.0))))
            for c in reranked_chunks
        ]
        
        # 1. Top score after reranking
        top_reranked_score = reranked_scores[0] if reranked_scores else 0.0
        
        # 2. Average of top results
        top_k = min(5, len(reranked_scores))
        avg_reranked = sum(reranked_scores[:top_k]) / top_k if top_k > 0 else 0.0
        
        # 3. Score concentration (are top results much better than rest?)
        if len(reranked_scores) > top_k:
            rest_avg = sum(reranked_scores[top_k:]) / len(reranked_scores[top_k:])
            if avg_reranked > 1e-6:
                concentration = (avg_reranked - rest_avg) / avg_reranked
                concentration_score = max(0.0, min(1.0, concentration))
        else:
            concentration_score = 0.5
        
        # 4. Improvement from original retrieval
        if original_chunks:
            original_scores = [
                max(0.0, min(1.0, c.get('similarity_score', 0.0)))
                for c in original_chunks[:top_k]
            ]
            original_avg = sum(original_scores) / len(original_scores) if original_scores else 0.0
            
            if original_avg > 1e-6:
                improvement = (avg_reranked - original_avg) / original_avg
                # Map [-0.5, +0.5] to [0, 1]
                improvement_score = max(0.0, min(1.0, 0.5 + improvement))
        else:
            improvement_score = 0.5
        
        # 5. Overall reranking confidence
        reranking_confidence = (
            top_reranked_score * 0.30 +
            avg_reranked * 0.30 +
            concentration_score * 0.20 +
            improvement_score * 0.20
        )
        reranking_confidence = max(0.0, min(1.0, reranking_confidence))
        
        return {
            "top_reranked_score": round(top_reranked_score, 3),
            "avg_reranked_score": round(avg_reranked, 3),
            "score_concentration": round(concentration_score, 3),
            "score_improvement": round(improvement_score, 3),
            "reranking_confidence": round(reranking_confidence, 3)
        }
    
    def _route_after_retrieval_eval(self, state: RAGState) -> str:
        """Decide whether to rerank or repeat reformulation after retrieval eval."""
        # Thresholds for retrieval quality (using score-based metrics, not RAGAS)
        RETRIEVAL_CONFIDENCE_THRESHOLD = 0.65
        TOP_SCORE_MIN = 0.40
        AVG_SCORE_MIN = 0.30

        rounds = state.get("retrieval_round", 0)
        max_rounds = state.get("max_iterations", self.max_iterations)
        metrics = state.get("retrieval_eval") or {}
        
        # Get score-based retrieval metrics (not context precision/recall)
        retrieval_confidence = metrics.get("retrieval_confidence", 0.0)
        top_score = metrics.get("top_score", 0.0)
        avg_top_score = metrics.get("avg_top_score", 0.0)
        bm25_dense_agreement = metrics.get("bm25_dense_agreement", 0.5)

        if rounds >= max_rounds:
            print(f"✓ Max retrieval rounds ({rounds}/{max_rounds}); moving to rerank")
            return "rerank"
        
        # Primary check: overall retrieval confidence
        if retrieval_confidence >= RETRIEVAL_CONFIDENCE_THRESHOLD:
            print(f"✓ Retrieval acceptable (confidence={retrieval_confidence:.3f}); continuing to rerank")
            return "rerank"
        
        # Critical failure checks
        if top_score < TOP_SCORE_MIN:
            print(f"↻ Top score too low ({top_score:.3f}); retrying reformulation")
            return "reformulate_query"
        
        if avg_top_score < AVG_SCORE_MIN:
            print(f"↻ Average score too low ({avg_top_score:.3f}); retrying reformulation")
            return "reformulate_query"
        
        # If methods disagree, query might be ambiguous
        if bm25_dense_agreement < 0.4:
            print(f"↻ BM25/dense disagree ({bm25_dense_agreement:.3f}); retrying reformulation")
            return "reformulate_query"
        
        # Marginal but not terrible
        print(f"⚠️  Retrieval marginal (confidence={retrieval_confidence:.3f}); proceeding to rerank")
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
        """Decide whether to regenerate answer or finish the pipeline."""
        GENERATION_THRESHOLD = 0.80

        rounds = state.get("generation_round", 0)
        max_rounds = state.get("max_iterations", self.max_iterations)
        overall = state.get("overall_score", 0.0)

        if overall >= GENERATION_THRESHOLD:
            print(f"✓ Generation met quality threshold ({overall:.3f} ≥ {GENERATION_THRESHOLD:.3f}); finishing")
            return END
        if rounds >= max_rounds:
            print("✓ Generation reached max iterations; exiting with latest answer")
            return END
        print(f"↻ Generation below threshold ({overall:.3f} < {GENERATION_THRESHOLD:.3f}); regenerating")
        return "generate"
    
    
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
        
        self.prompt_optimizer.reset_active_records()
        
        # Compile graph with checkpointer for memory
        checkpointer = MemorySaver()
        compiled_graph = self.graph.compile(checkpointer=checkpointer)
        
        # Initial state
        initial_state: RAGState = {
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
        
        # Run the graph with async streaming
        config = {"configurable": {"thread_id": f"query-{hash(query)}"}}
        final_state = None
        
        print(f"\n{'='*80}")
        print(f"Processing query: {query}")
        print(f"{'='*80}")
        
        # Use astream() for async execution
        async for state_update in compiled_graph.astream(initial_state, config=config):
            final_state = state_update
        
        # Extract final results
        if final_state and isinstance(final_state, dict):
            # Get the last node's output
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
