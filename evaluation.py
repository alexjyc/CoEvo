import os
import asyncio
from typing import List, Dict, Any, Optional
from langfuse import Langfuse
# from langfuse.decorators import observe, langfuse_context
# from langfuse.client import StatefulTraceClient

# RAGAS imports
from ragas.metrics import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerAccuracy,
    ContextRelevance
)
from ragas.run_config import RunConfig
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample

# LangChain wrappers for RAGAS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import pandas as pd
from datetime import datetime
import numpy as np


class Evaluator:
    """
    Langfuse-integrated RAG evaluator using RAGAS metrics
    """
    
    def __init__(self, openai_api_key: str):
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
        self.openai_api_key = openai_api_key
        self._setup_ragas_metrics()
    
    def _setup_ragas_metrics(self):
        """Setup RAGAS metrics with OpenAI models"""
        # Initialize LLM and embeddings for RAGAS
        llm = ChatOpenAI(openai_api_key=self.openai_api_key, model="gpt-4o-mini")
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key, model="text-embedding-3-small")
        
        # Wrap for RAGAS
        self.ragas_llm = LangchainLLMWrapper(llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
        
        # Define metrics
        self.metrics = {
            'faithfulness': Faithfulness(),
            'answer_accuracy': AnswerAccuracy(),
            'context_precision': ContextPrecision(),
            'context_recall': ContextRecall(),
            'context_relevance': ContextRelevance()
        }
        
        # Initialize metrics
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
    
    async def evaluate_trace(self, 
                                   query: str,
                                   contexts: List[str],
                                   answer: str,
                                   ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single RAG interaction using RAGAS metrics
        
        Args:
            question: The user question
            contexts: Retrieved contexts
            answer: Generated answer
            ground_truth: Ground truth answer (optional)
            
        Returns:
            Dictionary of RAGAS scores
        """
        print(f"Evaluating single trace with RAGAS...")
        try:
            scores = {}
            for m in self.metrics.values():
                sample = SingleTurnSample(
                    user_input=query,
                    retrieved_contexts=contexts,
                    response=answer,
                    reference=ground_truth
                )
                scores[m.name] = await m.single_turn_ascore(sample)
            
            return scores
        except Exception as e:
            print(f"Error during RAGAS evaluation: {e}")
            return {}