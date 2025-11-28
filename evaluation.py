import os
from typing import List, Dict, Any, Optional
from langfuse import Langfuse

# RAGAS imports (v0.2 collections API)
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas import SingleTurnSample
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    Faithfulness
)
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall
from openai import AsyncOpenAI

class Evaluator:
    """
    RAG evaluator using RAGAS v0.2 metrics with llm_factory
    
    Metrics by stage:
    - Retrieval: context_precision, context_recall
    - Generation: faithfulness, answer_relevancy, context_utilization
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize RAG evaluator with RAGAS v0.2 collections API
        
        Args:
            model: Model to use for evaluation ("gpt-5-nano" "gpt-4o-mini" or "gemini-2.5-flash")
        """
        # Initialize Langfuse client
        self.langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST")
        )
        
        # Initialize RAGAS evaluator LLM using llm_factory
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self._setup_ragas_llm()
        self._setup_ragas_metrics()
    
    def _setup_ragas_llm(self):
        """Setup evaluator LLM using RAGAS llm_factory"""
        if self.model == "gpt-5-nano":
            self.evaluator_llm = llm_factory(
                model="gpt-5-nano",
                client=self.client
            )
            self.evaluator_embeddings = embedding_factory(
                model="text-embedding-3-small",
                client=self.client
            )
        elif self.model == "gpt-4o-mini":
            # Use llm_factory for OpenAI
            self.evaluator_llm = llm_factory(
                model="gpt-4o-mini",
                client=self.client
            )
            # Embeddings for embedding-based metrics
            self.evaluator_embeddings = embedding_factory(
                model="text-embedding-3-small",
                client=self.client
            )
        elif self.model == "gemini-2.5-flash":
            # Use llm_factory for Gemini
            self.evaluator_llm = llm_factory(
                model="gemini-2.5-flash",
                client=self.client
            )
            # Fallback to OpenAI embeddings for now
            self.evaluator_embeddings = embedding_factory(
                model="text-embedding-3-small",
                client=self.client
            )
        else:
            raise ValueError(f"Unsupported model: {self.model}")
    
    def _setup_ragas_metrics(self):
        """Initialize RAGAS metrics with evaluator LLM"""
        # Retrieval stage metrics (context_precision can work without reference)
        self.metrics = {
            # Retrieval metrics - evaluate quality of retrieved contexts
            'id_based_context_precision': IDBasedContextPrecision(),
            'id_based_context_recall': IDBasedContextRecall(),

            # Reranking metrics - evaluate quality of reranked contexts
            'context_precision': ContextPrecision(llm=self.evaluator_llm),
            'context_recall': ContextRecall(llm=self.evaluator_llm),
            
            # Generation metrics - evaluate quality of generated answer
            'faithfulness': Faithfulness(llm=self.evaluator_llm),
            'answer_correctness': AnswerCorrectness(
                llm=self.evaluator_llm,
                embeddings=self.evaluator_embeddings
            )
        }

    async def evaluate_retrieval(self, retrieved_contexts: List[str], reference_contexts: List[str]) -> Dict[str, float]:
        """
        Evaluate retrieval quality using RAGAS metrics
        """
        keys = ['id_based_context_precision', 'id_based_context_recall']
        print(f"  Evaluating RAGAS metrics: {keys}")
        scores: Dict[str, float] = {}
        
        for key in keys:
            metric = self.metrics.get(key)
            if not metric:
                print(f"    ⚠️  Metric '{key}' not found, skipping")
                continue
            
            try:
                sample = SingleTurnSample(
                    retrieved_context_ids=retrieved_contexts,
                    reference_context_ids=reference_contexts
                )
                score = await metric.single_turn_ascore(sample)
                scores[key] = float(score) if score is not None else 0.0
                print(f"    ✓ {key}: {scores[key]:.3f}")
                
            except Exception as e:
                print(f"    ✗ Error evaluating {key}: {e}")
                scores[key] = 0.0
        
        return scores

    async def evaluate_reranking(self, query: str, contexts: List[str], ground_truth: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate reranking quality using RAGAS metrics
        """
        keys = ['context_precision']
        print(f"  Evaluating RAGAS metrics: {keys}")
        scores: Dict[str, float] = {}
        
        for key in keys:
            metric = self.metrics.get(key)
            if not metric:
                print(f"    ⚠️  Metric '{key}' not found, skipping")
                continue
            
            try:
                score = await metric.ascore(
                    user_input=query,
                    retrieved_contexts=contexts,
                    reference=ground_truth
                )
                scores[key] = float(score) if score is not None else 0.0
                print(f"    ✓ {key}: {scores[key]:.3f}")
                
            except Exception as e:
                print(f"    ✗ Error evaluating {key}: {e}")
                scores[key] = 0.0
        
        return scores

    async def evaluate_generation(self, query: str, contexts: List[str], answer: str, ground_truth: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate generation quality using RAGAS metrics
        """
        keys = ['faithfulness', 'answer_correctness']
        print(f"  Evaluating RAGAS metrics: {keys}")
        scores: Dict[str, float] = {}
        for key in keys:
            metric = self.metrics.get(key)
            if not metric:
                print(f"    ⚠️  Metric '{key}' not found, skipping")
                continue
            
            try:
                score = await metric.ascore(
                    user_input=query,
                    retrieved_contexts=contexts,
                    response=answer,
                    reference=ground_truth
                )
                scores[key] = float(score) if score is not None else 0.0
                print(f"    ✓ {key}: {scores[key]:.3f}")
                
            except Exception as e:
                print(f"    ✗ Error evaluating {key}: {e}")
                scores[key] = 0.0
        
        return scores

    async def evaluate_trace(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a complete RAG trace using all RAGAS metrics
        
        This evaluates both retrieval and generation stages together.
        
        Args:
            query: User query
            contexts: Retrieved contexts
            answer: Generated answer
            ground_truth: Optional reference answer
        
        Returns:
            Dict with all metric scores
        """
        print("  Evaluating complete trace with RAGAS...")
        return await self.evaluate_metrics(
            metric_keys=None,  # Evaluate all metrics
            query=query,
            contexts=contexts,
            answer=answer,
            ground_truth=ground_truth
        )
