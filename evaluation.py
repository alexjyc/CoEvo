"""
LLM-as-Judge Pipeline for RAG Output Assessment
Evaluates generated responses against ground truth with boolean pass/fail and confidence scoring
"""

import json
from typing import Dict, Any, Optional, List
import openai
from openai import OpenAI
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field


class JudgmentCriteria(Enum):
    """Different criteria for LLM judgment"""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    OVERALL = "overall"

class JudgmentOutput(BaseModel):
    """Structured output model for LLM judgment"""
    confidence_score: float = Field(
        description="Confidence score from 0.0 to 1.0 indicating how well the generated answer matches the ground truth",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of the evaluation and why the confidence score was assigned"
    )
    key_differences: List[str] = Field(
        description="List of key differences between generated answer and ground truth",
        default_factory=list
    )
    final_assessment: bool = Field(
        description="Final assessment of the generated response"
    )

class LLMJudge:
    """
    LLM-as-judge for assessing generated responses against ground truth
    Returns boolean pass/fail with confidence scores
    """
    
    def __init__(self, 
                 openai_api_key: str,
                 judge_model: str = "gpt-4o-mini",
                 temperature: float = 0.1,
                 confidence_threshold: float = 0.7):
        """
        Initialize the LLM judge
        
        Args:
            openai_api_key: OpenAI API key
            judge_model: Model to use for judging
            confidence_threshold: Minimum confidence to pass (0.0-1.0)
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model = judge_model
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
    
    def judge_response(self, 
                      generated_response: str,
                      ground_truth: str,
                      query: Optional[str] = None,
                      context: Optional[str] = None):
        """
        Judge a generated response against ground truth
        
        Args:
            generated_response: The generated response to evaluate
            ground_truth: The ground truth/expected answer
            query: Original query (optional, for context)
            context: Retrieved context (optional, for context)
            
        Returns:
            JSON object with judgment result
        """
        
        # Create comprehensive judgment prompt
        prompt = self._create_judgment_prompt(
            generated_response, ground_truth, query, context
        )
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format=JudgmentOutput,
            )
            
            judgment_data = response.choices[0].message.parsed
            
            return judgment_data
            
        except Exception as e:
            return {}
    
    def _create_judgment_prompt(self, 
                               generated_response: str,
                               ground_truth: str,
                               query: Optional[str] = None,
                               context: Optional[str] = None) -> str:
        """Create the judgment prompt for the LLM"""
        
        base_prompt = f"""You are an expert evaluator tasked with assessing the quality of a generated response against a ground truth answer.

Generated Response: {generated_response}

Ground Truth: {ground_truth}"""

        if query:
            base_prompt += f"\n\nOriginal Query: {query}"
        
        if context:
            base_prompt += f"\n\nContext Used: {context}"

        evaluation_prompt = f"""{base_prompt}

Please evaluate the generated response against the ground truth across these criteria:

1. **Accuracy**: Are the facts and information correct?
2. **Completeness**: Does it cover the key points from the ground truth?
3. **Relevance**: Does it appropriately address what was asked?
4. **Overall Quality**: Taking everything into account

For each criterion, provide a score from 0.0 to 1.0 (1.0 = perfect).

Your response must be a valid JSON object with this exact structure:
{{
    "accuracy_score": <float 0.0-1.0>,
    "completeness_score": <float 0.0-1.0>,
    "relevance_score": <float 0.0-1.0>,
    "overall_quality_score": <float 0.0-1.0>,
    "overall_confidence": <float 0.0-1.0>,
    "reasoning": "<explanation of your evaluation>",
    "key_strengths": ["<strength 1>", "<strength 2>"],
    "key_weaknesses": ["<weakness 1>", "<weakness 2>"],
    "recommendation": "<pass/fail recommendation>",
    "criteria_scores": {{
        "accuracy": <float 0.0-1.0>,
        "completeness": <float 0.0-1.0>,
        "relevance": <float 0.0-1.0>,
        "overall": <float 0.0-1.0>
    }}
}}

Provide only the JSON response, no additional text."""

        return evaluation_prompt
    
    def _parse_judgment_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse the JSON response from the LLM judge"""
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                return json.loads(response_text)
                
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract key information with regex or simple parsing
            return None