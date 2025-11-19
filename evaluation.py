import os
from typing import List, Dict, Any, Optional
from langfuse import Langfuse
# from langfuse.decorators import observe, langfuse_context
# from langfuse.client import StatefulTraceClient

# RAGAS imports
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    AnswerAccuracy
)
from ragas.run_config import RunConfig
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas.dataset_schema import SingleTurnSample

# LangChain wrappers for RAGAS
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper



class Evaluator:
    """
    Langfuse-integrated RAG evaluator using RAGAS metrics
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize Langfuse RAG evaluator
        
        Args:
            openai_api_key: OpenAI API key
            langfuse_public_key: Langfuse public key (or from env)
            langfuse_secret_key: Langfuse secret key (or from env)
            langfuse_host: Langfuse host URL
        """
        # Initialize Langfuse client
        self.langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST")
        )
        
        # Initialize RAGAS metrics
        if model == "gpt-4o-mini":
            self.llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o-mini", temperature=0)
            self.embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'), model="text-embedding-3-small")
        elif model == "gemini-2.5-flash":
            self.llm = ChatGoogleGenerativeAI(google_api_key=os.getenv('GEMINI_API_KEY'), model="gemini-2.5-flash", temperature=0)
            # Note: Replace with appropriate Gemini embeddings class when available
            self.embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'), model="text-embedding-3-small")
        self._setup_ragas_metrics()
    
    def _setup_ragas_metrics(self):    
        # Wrap for RAGAS
        self.ragas_llm = LangchainLLMWrapper(self.llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        # Define metrics
        self.metrics = {
            'faithfulness': Faithfulness(),
            'answer_accuracy': AnswerAccuracy(),
            'context_precision': LLMContextPrecisionWithoutReference(),
            'context_recall': LLMContextRecall(),
        }
        
        self._init_ragas_metrics()
    
    def _init_ragas_metrics(self):
        """Initialize RAGAS metrics with LLM and embeddings"""
        for metric in self.metrics.values():
            if isinstance(metric, MetricWithLLM):
                metric.llm = self.ragas_llm
            if isinstance(metric, MetricWithEmbeddings):
                metric.embeddings = self.ragas_embeddings
            
            run_config = RunConfig()
            metric.init(run_config)
    
    async def evaluate_metrics(
        self,
        metric_keys: Optional[List[str]],
        query: str,
        contexts: List[str],
        answer: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate a configurable subset of metrics for the current trace."""
        keys = metric_keys or list(self.metrics.keys())
        print(f"Evaluating metrics: {keys}, query: {query}, contexts: {contexts}, answer: {answer}, ground_truth: {ground_truth}")
        scores: Dict[str, Any] = {}
        for key in keys:
            metric = self.metrics.get(key)
            if not metric:
                continue
            sample = SingleTurnSample(
                user_input=query,
                retrieved_contexts=contexts,
                response=answer,
                reference=ground_truth
            )
            scores[key] = await metric.single_turn_ascore(sample)
        return scores

    async def evaluate_trace(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG interaction using the full set of RAGAS metrics
        """
        print("Evaluating single trace with RAGAS...")
        try:
            return await self.evaluate_metrics(
                metric_keys=None,
                query=query,
                contexts=contexts,
                answer=answer,
                ground_truth=ground_truth
            )
        except Exception as e:
            print(f"Error during RAGAS evaluation: {e}")
            return {}
