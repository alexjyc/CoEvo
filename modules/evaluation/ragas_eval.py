"""
RAGAS Evaluation Module

Provides module-specific evaluation using RAGAS metrics.
Each module has its own set of target metrics.

Module 1 (Query Planner + Retrieval): context_precision, context_recall (LLM-based)
Module 2 (Reranker): rerank_precision, rerank_recall, improvement metrics (compares vs pre-rerank)
Module 3 (Generator): faithfulness, answer_correctness
"""

import os
import asyncio
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from ragas import SingleTurnSample
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    Faithfulness,
    AnswerCorrectness,  # More comprehensive - factual accuracy + semantic similarity
    AnswerSimilarity,   # Required for AnswerCorrectness
    IDBasedContextPrecision,
    IDBasedContextRecall,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings

from modules.base import (
    ModuleType,
    ModuleEvaluator,
    EvaluationResult,
    Module,
    ModuleInput,
    ModuleOutput,
    QueryPlannerOutput,
    RetrievalOutput,
    RerankerOutput,
    GeneratorOutput,
)


class RAGASEvaluator(ModuleEvaluator):
    """
    RAGAS-based evaluator for RAG pipeline modules.

    Provides module-specific metrics:
    - Query Planner + Retrieval: context_precision, context_recall (measures retrieval quality)
    - Reranker: rerank_precision, rerank_recall, precision_improvement, recall_improvement
                (measures post-rerank quality AND improvement over pre-rerank)
    - Generator: faithfulness, answer_correctness
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Initialize RAGAS metrics with evaluator LLM using LangchainLLMWrapper"""
        # Use LangchainLLMWrapper as recommended by RAGAS documentation
        chat_llm = ChatOpenAI(
            model=self.model,
            temperature=0,
            request_timeout=60,
            max_retries=3,
        )
        self.evaluator_llm = LangchainLLMWrapper(chat_llm)

        # Initialize embeddings for AnswerSimilarity (used by AnswerCorrectness)
        langchain_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

        # Initialize AnswerSimilarity with embeddings (required for AnswerCorrectness)
        answer_similarity = AnswerSimilarity(embeddings=self.evaluator_embeddings)

        # Initialize metrics
        self.metrics = {
            # ID-based metrics for retrieval evaluation (when ground truth IDs available)
            'id_context_precision': IDBasedContextPrecision(),
            'id_context_recall': IDBasedContextRecall(),

            # LLM-based metrics (when only reference answer available)
            'context_precision': LLMContextPrecisionWithReference(llm=self.evaluator_llm),
            'context_recall': LLMContextRecall(llm=self.evaluator_llm),
            'faithfulness': Faithfulness(llm=self.evaluator_llm),
            # AnswerCorrectness = factual accuracy + semantic similarity (more comprehensive)
            'answer_correctness': AnswerCorrectness(
                llm=self.evaluator_llm,
                answer_similarity=answer_similarity,
            ),
        }

    async def _call_metric(
        self,
        metric_name: str,
        max_retries: int = 5,
        **kwargs
    ) -> float:
        """
        Call a RAGAS metric with automatic retry for connection errors.
        
        Uses SingleTurnSample as recommended by RAGAS documentation.
        
        Args:
            metric_name: Name of the metric in self.metrics
            max_retries: Maximum retry attempts
            **kwargs: Arguments to build SingleTurnSample
            
        Returns:
            Metric score as float, or 0.0 on failure
        """
        metric = self.metrics[metric_name]
        
        # Build SingleTurnSample from kwargs
        sample = SingleTurnSample(**kwargs)
        
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                result = await metric.single_turn_ascore(sample)
                return float(result) if result is not None else 0.0
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if retryable error
                is_retryable = any(err in error_str for err in [
                    'connection', 'timeout', 'rate limit', 'server error',
                    '503', '502', '504', '429', 'reset by peer', 'broken pipe'
                ])
                
                if attempt < max_retries and is_retryable:
                    delay = min(2.0 * (2 ** attempt), 60.0)
                    print(f"  [RETRY] {metric_name} attempt {attempt + 1}/{max_retries}, waiting {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    break
        
        # Log final failure
        print(f"[RAGAS] {metric_name} failed after {max_retries} retries: {str(last_exception)[:100]}")
        return 0.0

    async def evaluate(
        self,
        module: Module,
        input_data: ModuleInput,
        output_data: ModuleOutput,
        ground_truth: Optional[Any] = None
    ) -> EvaluationResult:
        """
        Evaluate module output using appropriate RAGAS metrics.

        Routes to module-specific evaluation based on module type.
        """
        if module.module_type == ModuleType.QUERY_PLANNER:
            metrics = await self._evaluate_query_planner(input_data, output_data, ground_truth)
        elif module.module_type == ModuleType.RETRIEVAL:
            metrics = await self._evaluate_retrieval(input_data, output_data, ground_truth)
        elif module.module_type == ModuleType.RERANKER:
            metrics = await self._evaluate_reranker(input_data, output_data, ground_truth)
        elif module.module_type == ModuleType.GENERATOR:
            metrics = await self._evaluate_generator(input_data, output_data, ground_truth)
        else:
            metrics = {}

        return EvaluationResult(
            module_type=module.module_type,
            metrics=metrics,
            input_data=input_data.to_dict(),
            output_data=output_data.to_dict(),
        )

    async def _evaluate_query_planner(
        self,
        input_data: ModuleInput,
        output_data: QueryPlannerOutput,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Evaluate query planner output.

        Note: Query planner is evaluated indirectly via retrieval results.
        Direct evaluation would require assessing query decomposition quality.
        """
        # Query planner is typically evaluated via downstream retrieval quality
        # Here we just track basic metrics
        return {
            "num_queries": len(output_data.queries),
            "is_decomposition": 1.0 if output_data.mode == "decomposition" else 0.0,
        }

    async def _evaluate_retrieval(
        self,
        input_data: ModuleInput,
        output_data: RetrievalOutput,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Evaluate retrieval output at chunk level (primary) with LLM fallback.

        Evaluation modes (in priority order):
        1. Chunk-level (PRIMARY): Compare retrieved chunk_indices to relevant_chunk_indices
           - This is the correct level since retrieval operates on chunks
        2. LLM-based (FALLBACK): When only reference answer available

        Args:
            ground_truth: Dict with:
                - 'relevant_chunk_indices' (List[int]) for chunk-level eval (preferred)
                - 'query' (str) and 'reference' (str) for LLM-based eval
        """
        scores = {}

        if not ground_truth:
            return scores

        # Mode 1: Chunk-level evaluation (PRIMARY - correct level for RAG retrieval)
        relevant_chunk_indices = ground_truth.get('relevant_chunk_indices', [])
        retrieved_chunk_indices = output_data.chunk_indices if output_data.chunk_indices else []
        
        if relevant_chunk_indices and retrieved_chunk_indices:
            # Calculate chunk-level precision and recall
            retrieved_set = set(retrieved_chunk_indices)
            relevant_set = set(relevant_chunk_indices)
            
            hits = retrieved_set & relevant_set
            
            chunk_precision = len(hits) / len(retrieved_set) if retrieved_set else 0.0
            chunk_recall = len(hits) / len(relevant_set) if relevant_set else 0.0
            chunk_f1 = (2 * chunk_precision * chunk_recall / (chunk_precision + chunk_recall) 
                        if (chunk_precision + chunk_recall) > 0 else 0.0)
            
            scores['retrieval_precision'] = chunk_precision
            scores['retrieval_recall'] = chunk_recall
            scores['retrieval_f1'] = chunk_f1
            scores['chunks_retrieved'] = len(retrieved_set)
            scores['chunks_relevant'] = len(relevant_set)
            scores['chunks_hit'] = len(hits)

        # Also try RAGAS ID-based metrics for chunk evaluation
        if relevant_chunk_indices and retrieved_chunk_indices:
            try:
                sample = SingleTurnSample(
                    retrieved_context_ids=retrieved_chunk_indices,
                    reference_context_ids=relevant_chunk_indices
                )

                precision = await self.metrics['id_context_precision'].single_turn_ascore(sample)
                recall = await self.metrics['id_context_recall'].single_turn_ascore(sample)

                # Store RAGAS scores separately if different calculation method
                scores['ragas_precision'] = float(precision) if precision else 0.0
                scores['ragas_recall'] = float(recall) if recall else 0.0

            except Exception as e:
                print(f"Error in RAGAS chunk-level evaluation: {e}")

        # Mode 2: LLM-based evaluation (FALLBACK - when only reference answer available)
        if 'query' in ground_truth and 'reference' in ground_truth:
            query = ground_truth['query']
            reference = ground_truth['reference']
            
            # Get document texts for LLM evaluation
            doc_texts = output_data.document_texts if output_data.document_texts else []
            if not doc_texts:
                doc_texts = [d.get('chunk_text', '') for d in output_data.documents if d]
            
            if doc_texts:
                try:
                    # Ensure all doc_texts are strings
                    doc_texts = [str(d) for d in doc_texts if d]
                    if not doc_texts:
                        doc_texts = ["No context available."]
                    
                    scores['context_precision'] = await self._call_metric(
                        'context_precision',
                        user_input=str(query),
                        retrieved_contexts=doc_texts,
                        reference=str(reference)
                    )
                    scores['context_recall'] = await self._call_metric(
                        'context_recall',
                        user_input=str(query),
                        retrieved_contexts=doc_texts,
                        reference=str(reference)
                    )

                    # Calculate F1
                    if scores['context_precision'] + scores['context_recall'] > 0:
                        scores['context_f1'] = 2 * (
                            scores['context_precision'] * scores['context_recall']
                        ) / (scores['context_precision'] + scores['context_recall'])
                    else:
                        scores['context_f1'] = 0.0

                except Exception as e:
                    print(f"Error in LLM-based retrieval evaluation: {e}")
                    if 'context_precision' not in scores:
                        scores.update({'context_precision': 0.0, 'context_recall': 0.0, 'context_f1': 0.0})

        return scores

    async def _evaluate_reranker(
        self,
        input_data: ModuleInput,
        output_data: RerankerOutput,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Evaluate reranker output using LLM-based context metrics.
        
        Measures both absolute quality AND improvement over pre-rerank results.

        Args:
            ground_truth: Dict with:
                - 'query' (str): The original query
                - 'reference' (str): Ground truth answer
                - 'pre_rerank_documents' (List[str], optional): Documents before reranking
                - 'pre_rerank_precision' (float, optional): Pre-computed precision before rerank
                - 'pre_rerank_recall' (float, optional): Pre-computed recall before rerank
        """
        scores = {}

        if not ground_truth:
            return scores

        query = ground_truth.get('query', '')
        reference = ground_truth.get('reference', '')

        if not query or not reference:
            return scores
        
        # Ensure reference is a string (not float or other type)
        reference = str(reference) if reference else ""
        query = str(query) if query else ""
        
        # Get ranked documents and ensure it's a list of strings
        ranked_docs = getattr(output_data, 'ranked_documents', [])
        if not ranked_docs:
            return {'context_precision': 0.0, 'context_recall': 0.0, 'context_f1': 0.0}
        
        # Convert all docs to strings and filter empty
        ranked_docs = [str(doc) for doc in ranked_docs if doc]
        if not ranked_docs:
            return {'context_precision': 0.0, 'context_recall': 0.0, 'context_f1': 0.0}

        try:
            # Evaluate context precision after reranking (with retry)
            scores['rerank_precision'] = await self._call_metric(
                'context_precision',
                user_input=str(query),
                retrieved_contexts=[str(d) for d in ranked_docs],
                reference=str(reference)
            )

            # Evaluate context recall after reranking (with retry)
            scores['rerank_recall'] = await self._call_metric(
                'context_recall',
                user_input=str(query),
                retrieved_contexts=[str(d) for d in ranked_docs],
                reference=str(reference)
            )

            # Calculate F1
            if scores['rerank_precision'] + scores['rerank_recall'] > 0:
                scores['rerank_f1'] = 2 * (
                    scores['rerank_precision'] * scores['rerank_recall']
                ) / (scores['rerank_precision'] + scores['rerank_recall'])
            else:
                scores['rerank_f1'] = 0.0

            # Calculate improvement over pre-rerank (if pre-rerank scores provided)
            pre_precision = ground_truth.get('pre_rerank_precision')
            pre_recall = ground_truth.get('pre_rerank_recall')

            # If pre-rerank documents provided but not scores, compute them
            pre_rerank_docs = ground_truth.get('pre_rerank_documents')
            if pre_rerank_docs and pre_precision is None:
                pre_docs = [str(d) for d in pre_rerank_docs if d]
                if pre_docs:
                    pre_precision = await self._call_metric(
                        'context_precision',
                        user_input=str(query),
                        retrieved_contexts=pre_docs,
                        reference=str(reference)
                    )

            if pre_rerank_docs and pre_recall is None:
                pre_docs = [str(d) for d in pre_rerank_docs if d]
                if pre_docs:
                    pre_recall = await self._call_metric(
                        'context_recall',
                        user_input=str(query),
                        retrieved_contexts=pre_docs,
                        reference=str(reference)
                    )

            # Calculate improvement metrics
            if pre_precision is not None:
                scores['pre_rerank_precision'] = float(pre_precision)
                scores['precision_improvement'] = scores['rerank_precision'] - scores['pre_rerank_precision']
                scores['precision_improvement_pct'] = (
                    (scores['precision_improvement'] / scores['pre_rerank_precision'] * 100)
                    if scores['pre_rerank_precision'] > 0 else 0.0
                )

            if pre_recall is not None:
                scores['pre_rerank_recall'] = float(pre_recall)
                scores['recall_improvement'] = scores['rerank_recall'] - scores['pre_rerank_recall']
                scores['recall_improvement_pct'] = (
                    (scores['recall_improvement'] / scores['pre_rerank_recall'] * 100)
                    if scores['pre_rerank_recall'] > 0 else 0.0
                )

            # Calculate overall F1 improvement
            if pre_precision is not None and pre_recall is not None:
                pre_f1 = (
                    2 * pre_precision * pre_recall / (pre_precision + pre_recall)
                    if (pre_precision + pre_recall) > 0 else 0.0
                )
                scores['pre_rerank_f1'] = pre_f1
                scores['f1_improvement'] = scores['rerank_f1'] - pre_f1
                scores['f1_improvement_pct'] = (
                    (scores['f1_improvement'] / pre_f1 * 100) if pre_f1 > 0 else 0.0
                )

        except Exception as e:
            print(f"Error in reranker evaluation: {e}")
            scores = {'rerank_precision': 0.0, 'rerank_recall': 0.0, 'rerank_f1': 0.0}

        return scores

    async def _evaluate_generator(
        self,
        input_data: ModuleInput,
        output_data: GeneratorOutput,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Evaluate generator output using faithfulness and answer_correctness.

        Args:
            ground_truth: Dict with 'query', 'contexts', 'reference' (ground truth answer)
        """
        scores = {'faithfulness': 0.0, 'answer_correctness': 0.0, 'generation_quality': 0.0}

        if not ground_truth:
            return scores

        query = ground_truth.get('query', '')
        contexts = ground_truth.get('contexts', [])
        reference = ground_truth.get('reference', '')
        
        # Ensure all inputs are proper types
        query = str(query) if query else ""
        reference = str(reference) if reference else ""
        
        # Ensure contexts is a list of non-empty strings
        if not contexts:
            contexts = []
        contexts = [str(c) for c in contexts if c]
        if not contexts:
            contexts = ["No context provided."]
        
        # Get the answer from output_data
        answer = getattr(output_data, 'answer', '')
        answer = str(answer) if answer else "No answer generated."

        # Faithfulness: Is the answer grounded in contexts? (with retry)
        scores['faithfulness'] = await self._call_metric(
            'faithfulness',
            user_input=str(query),
            response=str(answer),
            retrieved_contexts=[str(c) for c in contexts]
        )

        # Answer Correctness: Does the answer match ground truth? (with retry)
        if reference:
            scores['answer_correctness'] = await self._call_metric(
                'answer_correctness',
                user_input=str(query),
                response=str(answer),
                reference=str(reference)
            )

        # Calculate generation quality
        scores['generation_quality'] = (
            scores.get('faithfulness', 0) + scores.get('answer_correctness', 0)
        ) / 2

        return scores

    async def compute_retrieval_scores(
        self,
        query: str,
        documents: List[str],
        reference: str
    ) -> Dict[str, float]:
        """
        Compute context precision and recall for retrieved documents.
        
        Useful for getting pre-rerank scores to pass to reranker evaluation.
        
        Args:
            query: The original query
            documents: List of retrieved document texts
            reference: Ground truth answer
            
        Returns:
            Dict with 'context_precision' and 'context_recall'
        """
        scores = {'context_precision': 0.0, 'context_recall': 0.0, 'context_f1': 0.0}
        
        # Validate inputs
        if not documents or not reference:
            return scores
        
        # Ensure all documents are non-empty strings
        documents = [str(d) for d in documents if d]
        if not documents:
            return scores
        
        reference = str(reference)
        
        scores['context_precision'] = await self._call_metric(
            'context_precision',
            user_input=str(query),
            retrieved_contexts=documents,
            reference=reference
        )
        
        scores['context_recall'] = await self._call_metric(
            'context_recall',
            user_input=str(query),
            retrieved_contexts=documents,
            reference=reference
        )
        
        # Calculate F1
        if scores['context_precision'] + scores['context_recall'] > 0:
            scores['context_f1'] = 2 * (
                scores['context_precision'] * scores['context_recall']
            ) / (scores['context_precision'] + scores['context_recall'])
        else:
            scores['context_f1'] = 0.0
            
        return scores

    async def evaluate_end_to_end(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        ground_truth: Optional[str] = None,
        retrieved_chunk_indices: Optional[List[int]] = None,
        relevant_chunk_indices: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate complete RAG pipeline output.

        Returns all metrics for retrieval and generation.
        
        Args:
            query: The user query
            contexts: Retrieved context texts
            answer: Generated answer
            ground_truth: Expected answer
            retrieved_chunk_indices: Chunk indices that were retrieved
            relevant_chunk_indices: Ground truth relevant chunk indices
        """
        scores = {}
        
        # Validate inputs - ensure contexts is a non-empty list of strings
        if not contexts or not isinstance(contexts, (list, tuple)):
            contexts = ["No context retrieved."]
        else:
            # Convert all items to strings, filtering empty/None values
            validated_contexts = []
            for c in contexts:
                if c is not None:
                    c_str = str(c) if not isinstance(c, str) else c
                    if c_str.strip():  # Only add non-empty strings
                        validated_contexts.append(c_str)
            contexts = validated_contexts if validated_contexts else ["No context retrieved."]
            
        # Ensure query is a string
        if not isinstance(query, str):
            query = str(query) if query else "No query provided."
            
        # Ensure answer and ground_truth are strings
        if not isinstance(answer, str):
            answer = str(answer) if answer else "No answer generated."
        if not answer.strip():
            answer = "No answer generated."
            
        if ground_truth is not None and not isinstance(ground_truth, str):
            ground_truth = str(ground_truth) if ground_truth else None
        if ground_truth and not ground_truth.strip():
            ground_truth = None

        # Chunk-level retrieval metrics (PRIMARY - correct level for RAG retrieval)
        if retrieved_chunk_indices and relevant_chunk_indices:
            try:
                sample = SingleTurnSample(
                    retrieved_context_ids=retrieved_chunk_indices,
                    reference_context_ids=relevant_chunk_indices
                )
                precision = await self.metrics['id_context_precision'].single_turn_ascore(sample)
                recall = await self.metrics['id_context_recall'].single_turn_ascore(sample)

                scores['retrieval_precision'] = float(precision) if precision else 0.0
                scores['retrieval_recall'] = float(recall) if recall else 0.0
            except Exception as e:
                print(f"Error in retrieval evaluation: {e}")
                scores['retrieval_precision'] = 0.0
                scores['retrieval_recall'] = 0.0

        # Context quality (LLM-based) - only if we have real contexts and ground truth
        if ground_truth and len(contexts) > 0 and contexts[0] != "No context retrieved.":
            scores['context_precision'] = await self._call_metric(
                'context_precision',
                user_input=query,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
            scores['context_recall'] = await self._call_metric(
                'context_recall',
                user_input=query,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        else:
            scores['context_precision'] = 0.0
            scores['context_recall'] = 0.0

        # Generation metrics - only if we have real contexts
        if len(contexts) > 0 and contexts[0] != "No context retrieved.":
            scores['faithfulness'] = await self._call_metric(
                'faithfulness',
                user_input=query,
                response=answer,
                retrieved_contexts=contexts,
            )
        else:
            scores['faithfulness'] = 0.0

        if ground_truth:
            scores['answer_correctness'] = await self._call_metric(
                'answer_correctness',
                user_input=query,
                response=answer,
                reference=ground_truth,
            )
        else:
            scores['answer_correctness'] = 0.0

        # Calculate aggregate scores
        if 'retrieval_precision' in scores and 'retrieval_recall' in scores:
            p, r = scores['retrieval_precision'], scores['retrieval_recall']
            scores['retrieval_f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        if 'context_precision' in scores and 'context_recall' in scores:
            p, r = scores['context_precision'], scores['context_recall']
            scores['context_f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        scores['generation_quality'] = (
            scores.get('faithfulness', 0) + scores.get('answer_correctness', 0)
        ) / 2

        # Overall quality (weighted)
        scores['overall_quality'] = (
            scores.get('retrieval_f1', scores.get('context_f1', 0)) * 0.3 +
            scores.get('context_f1', 0) * 0.2 +
            scores['generation_quality'] * 0.5
        )

        return scores
