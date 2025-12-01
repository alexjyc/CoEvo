"""
Memory System for RAG Optimization
Implements hybrid semantic + judge-based memory retrieval with weighted sampling

Design:
- ExecutionMemory: Stores successful/failed patterns for each module
- MemoryStore: Manages memory lifecycle and retrieval
- Judge: LLM-based reflection for quality feedback generation
- Weight Update: TD-learning inspired reward system

Author: Following Google/OpenAI clean code guidelines
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np

from LLMBase import LLMBase
from pydantic import BaseModel, Field


# ============================================================================
# Module Types
# ============================================================================

class ModuleType(Enum):
    """Three optimization modules in RAG pipeline"""
    QUERY_PLANNING = "query_planning"
    RERANKING = "reranking"
    GENERATION = "generation"


# ============================================================================
# Memory Data Structures
# ============================================================================

@dataclass
class ModuleMemory:
    """Memory for a single module's execution"""
    module: ModuleType
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    score: float  # Module-specific score (F1 or quality)
    what_worked: str = ""
    what_failed: str = ""


@dataclass
class ExecutionMemory:
    """
    Complete memory of a single RAG execution
    Stores patterns, outcomes, and lessons learned
    """
    # Identifiers
    memory_id: str
    query: str
    timestamp: float = field(default_factory=time.time)
    
    # Execution data
    contexts: List[str] = field(default_factory=list)
    answer: str = ""
    
    # Module-specific memories
    query_planning_memory: Optional[ModuleMemory] = None
    reranking_memory: Optional[ModuleMemory] = None
    generation_memory: Optional[ModuleMemory] = None
    
    # Overall metrics
    retrieval_f1: float = 0.0
    reranking_f1: float = 0.0
    generation_quality: float = 0.0
    overall_quality: float = 0.0
    
    # Memory metadata (for weighting)
    weight: float = 1.0
    access_count: int = 0
    success_count: int = 0  # Times this memory led to improvement
    
    # Judge assessments
    judge_assessments: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_module_memory(self, module: ModuleType) -> Optional[ModuleMemory]:
        """Get memory for specific module"""
        if module == ModuleType.QUERY_PLANNING:
            return self.query_planning_memory
        elif module == ModuleType.RERANKING:
            return self.reranking_memory
        elif module == ModuleType.GENERATION:
            return self.generation_memory
        return None
    
    def get_module_score(self, module: ModuleType) -> float:
        """Get score for specific module"""
        if module == ModuleType.QUERY_PLANNING:
            return self.retrieval_f1
        elif module == ModuleType.RERANKING:
            return self.reranking_f1
        elif module == ModuleType.GENERATION:
            return self.generation_quality
        return 0.0


# ============================================================================
# Judge Output Schema
# ============================================================================

class MemoryRelevance(BaseModel):
    """Judge's assessment of a single memory's relevance"""
    memory_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    what_to_apply: str = Field(description="Specific technique/strategy to use")
    why_relevant: str = Field(description="Brief reasoning for selection")


class JudgeReflection(BaseModel):
    """Judge's complete reflection on memories"""
    selected_memories: List[MemoryRelevance]
    module_feedback: Dict[str, str] = Field(
        description="Stage-specific feedback: {module_name: feedback_text}"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Overall reasoning for selections")


# ============================================================================
# Memory Store
# ============================================================================

class MemoryStore:
    """
    Manages memory storage, retrieval, and weight updates
    
    Strategy:
    - Stage 1: Semantic retrieval (fast filter)
    - Stage 2: Judge reflection (quality + feedback)
    - Weight updates: TD-learning inspired
    """
    
    def __init__(
        self,
        judge_model: str = "gpt-4o-mini",
        use_judge_threshold: int = 10,
        max_memories: int = 100
    ):
        """
        Initialize memory store
        
        Args:
            judge_model: LLM for judge-based reflection
            use_judge_threshold: Min memories before using judge
            max_memories: Max memories to keep (prune low-weight)
        """
        self.memories: List[ExecutionMemory] = []
        self.judge = LLMBase(model_name=judge_model)
        self.use_judge_threshold = use_judge_threshold
        self.max_memories = max_memories
        
        # Embedding cache for semantic retrieval
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def add_memory(self, memory: ExecutionMemory) -> None:
        """Add new memory to store"""
        self.memories.append(memory)
        
        # Prune if exceeds max
        if len(self.memories) > self.max_memories:
            self._prune_memories()
    
    async def retrieve_with_reflection(
        self,
        query: str,
        module: ModuleType,
        current_state: Dict[str, Any],
        top_k: int = 5
    ) -> Tuple[List[ExecutionMemory], str]:
        """
        Retrieve relevant memories and generate module-specific feedback
        
        Returns:
            (selected_memories, feedback_text)
        """
        if not self.memories:
            return [], ""
        
        # Stage 1: Semantic retrieval (fast filter)
        semantic_candidates = self._semantic_retrieval(
            query=query,
            module=module,
            top_k=20  # Get top 20 for judge
        )
        
        if not semantic_candidates:
            return [], ""
        
        # Stage 2: Judge reflection (if enough memories)
        if len(self.memories) >= self.use_judge_threshold:
            judge_result = await self._judge_reflection(
                query=query,
                module=module,
                candidates=semantic_candidates,
                current_state=current_state,
                top_k=top_k
            )
            
            # Update access counts for retrieved memories
            for mem_relevance in judge_result.selected_memories:
                memory = self._find_memory(mem_relevance.memory_id)
                if memory:
                    memory.access_count += 1
                    memory.judge_assessments.append({
                        'timestamp': time.time(),
                        'module': module.value,
                        'relevance': mem_relevance.relevance_score,
                        'feedback': mem_relevance.what_to_apply
                    })
            
            selected_memories = [
                self._find_memory(m.memory_id) 
                for m in judge_result.selected_memories
            ]
            selected_memories = [m for m in selected_memories if m is not None]
            
            feedback_text = judge_result.module_feedback.get(module.value, "")
            
            return selected_memories, feedback_text
        
        else:
            # Early phase: Simple semantic retrieval
            top_memories = semantic_candidates[:top_k]
            feedback_text = self._simple_feedback(top_memories, module)
            
            for mem in top_memories:
                mem.access_count += 1
            
            return top_memories, feedback_text
    
    def update_memory_weights(
        self,
        retrieved_memories: List[ExecutionMemory],
        outcome: Dict[str, Any]
    ) -> None:
        """
        Update weights based on execution outcome
        
        Args:
            retrieved_memories: Memories that were retrieved
            outcome: {
                'quality_improved': bool,
                'retrieval_f1_delta': float,
                'reranking_f1_delta': float,
                'generation_quality_delta': float
            }
        """
        improved = outcome.get('quality_improved', False)
        
        for memory in retrieved_memories:
            # Base weight update
            if improved:
                memory.success_count += 1
                memory.weight += 0.1  # Success bonus
            
            # Recurrence bonus (cap at +1.0)
            recurrence_bonus = min(0.05 * memory.access_count, 1.0)
            memory.weight += recurrence_bonus
            
            # Quality bonus
            quality_bonus = 0.2 * memory.overall_quality
            memory.weight += quality_bonus
            
            # Temporal decay
            age_hours = (time.time() - memory.timestamp) / 3600
            temporal_decay = 0.99 ** age_hours
            memory.weight *= temporal_decay
            
            # Diversity penalty (prevent over-reliance on one pattern)
            if memory.access_count > 10:
                success_rate = memory.success_count / memory.access_count
                if success_rate < 0.4:
                    memory.weight *= 0.8  # This pattern isn't helping
    
    def _semantic_retrieval(
        self,
        query: str,
        module: ModuleType,
        top_k: int
    ) -> List[ExecutionMemory]:
        """
        Fast semantic retrieval using query embeddings
        
        Returns memories weighted by:
        - Semantic similarity to query
        - Module-specific score
        - Current weight
        - Recency
        """
        query_emb = self._get_embedding(query)
        
        scored_memories = []
        for memory in self.memories:
            # Semantic similarity
            memory_emb = self._get_embedding(memory.query)
            similarity = self._cosine_similarity(query_emb, memory_emb)
            
            # Module-specific score
            module_score = memory.get_module_score(module)
            
            # Recency factor
            age_hours = (time.time() - memory.timestamp) / 3600
            recency_factor = 0.99 ** age_hours
            
            # Combined score
            combined_score = (
                similarity * 0.4 +
                module_score * 0.3 +
                memory.weight * 0.2 +
                recency_factor * 0.1
            )
            
            scored_memories.append((memory, combined_score))
        
        # Sort by score and return top-k
        sorted_memories = sorted(
            scored_memories,
            key=lambda x: x[1],
            reverse=True
        )
        
        return [mem for mem, _ in sorted_memories[:top_k]]
    
    async def _judge_reflection(
        self,
        query: str,
        module: ModuleType,
        candidates: List[ExecutionMemory],
        current_state: Dict[str, Any],
        top_k: int
    ) -> JudgeReflection:
        """
        Judge evaluates memories and generates module-specific feedback
        
        This is the KEY innovation: Judge doesn't just rank,
        it REFLECTS and generates actionable strategies
        """
        # Format candidates for judge
        formatted_candidates = []
        for i, mem in enumerate(candidates):
            module_mem = mem.get_module_memory(module)
            module_score = mem.get_module_score(module)
            
            formatted_candidates.append(f"""
Memory {i+1} [ID: {mem.memory_id}]:
- Query: {mem.query}
- Module Score ({module.value}): {module_score:.3f}
- Overall Quality: {mem.overall_quality:.3f}
- Success Rate: {mem.success_count}/{mem.access_count}
- What Worked: {module_mem.what_worked if module_mem else 'N/A'}
- What Failed: {module_mem.what_failed if module_mem else 'N/A'}
""")
        
        judge_prompt = f"""You are a meta-learning judge for RAG optimization.

Current Query: {query}
Current Module: {module.value}
Current Iteration: {current_state.get('iteration', 0)}
Current Module Score: {current_state.get(f'{module.value}_score', 'N/A')}

Past Executions (ordered by weighted relevance):
{chr(10).join(formatted_candidates)}

Your Task:
1. Select the top {top_k} most relevant past executions for this module
2. Extract specific, actionable lessons for {module.value}
3. Generate concrete feedback that can improve performance

Module-Specific Guidance:
- query_planning: Focus on decomposition vs reformulation, query types
- reranking: Focus on document prioritization strategies
- generation: Focus on answer structure, grounding, conciseness

Return structured output with:
- selected_memories: List of {{memory_id, relevance_score, what_to_apply, why_relevant}}
- module_feedback: Dict with {module.value} key and actionable feedback
- confidence: How confident are you these lessons will help? (0-1)
- reasoning: Brief explanation of your selections
"""
        
        # Get structured response from judge
        structured_judge = self.judge.llm.with_structured_output(JudgeReflection)
        
        response = await structured_judge.ainvoke([
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": f"Analyze and provide top {top_k} recommendations."}
        ])
        
        return response
    
    def _simple_feedback(
        self,
        memories: List[ExecutionMemory],
        module: ModuleType
    ) -> str:
        """
        Generate simple feedback from high-quality memories
        Used when not enough memories for judge
        """
        feedback_parts = []
        
        for mem in memories:
            module_mem = mem.get_module_memory(module)
            if module_mem and mem.get_module_score(module) > 0.7:
                if module_mem.what_worked:
                    feedback_parts.append(f"- {module_mem.what_worked}")
        
        if feedback_parts:
            return f"Successful patterns for {module.value}:\n" + "\n".join(feedback_parts[:3])
        
        return ""
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get cached embedding for text (mock implementation)"""
        # TODO: Replace with actual embedding model
        # For now, use simple hash-based embedding
        if text not in self._embedding_cache:
            # Simple mock: hash text and convert to embedding
            hash_val = hashlib.md5(text.encode()).hexdigest()
            # Convert hex to numbers and normalize
            embedding = np.array([int(hash_val[i:i+2], 16) for i in range(0, 32, 2)])
            embedding = embedding / np.linalg.norm(embedding)
            self._embedding_cache[text] = embedding
        
        return self._embedding_cache[text]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def _find_memory(self, memory_id: str) -> Optional[ExecutionMemory]:
        """Find memory by ID"""
        for mem in self.memories:
            if mem.memory_id == memory_id:
                return mem
        return None
    
    def _prune_memories(self) -> None:
        """Prune low-weight memories to stay under max_memories"""
        # Sort by weight and keep top memories
        self.memories = sorted(
            self.memories,
            key=lambda m: m.weight,
            reverse=True
        )[:self.max_memories]
        
        print(f"  🗑️ Pruned memories to {len(self.memories)} (kept highest weights)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        if not self.memories:
            return {
                'total_memories': 0,
                'avg_weight': 0.0,
                'avg_success_rate': 0.0
            }
        
        total_accesses = sum(m.access_count for m in self.memories)
        total_successes = sum(m.success_count for m in self.memories)
        
        return {
            'total_memories': len(self.memories),
            'avg_weight': np.mean([m.weight for m in self.memories]),
            'avg_access_count': np.mean([m.access_count for m in self.memories]),
            'avg_success_rate': total_successes / total_accesses if total_accesses > 0 else 0.0,
            'high_quality_memories': sum(1 for m in self.memories if m.overall_quality > 0.8)
        }

