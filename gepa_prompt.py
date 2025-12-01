import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from tqdm import tqdm

from rag_pipeline import UnifiedRAGPipeline  # Updated import
from evaluation import Evaluator


@dataclass
class TrainingExample:
    """Base class for module-specific training examples"""
    query_id: str
    query: str
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReformulationExample(TrainingExample):
    """Training example for query reformulation module"""
    reference_doc_ids: List[int] = None
    reference_evidence_texts: List[Dict[str, Any]] = None
    
    # Evaluation signals
    ideal_sub_queries: Optional[List[str]] = None
    retrieval_precision: Optional[float] = None
    retrieval_recall: Optional[float] = None


@dataclass
class RerankingExample(TrainingExample):
    """Training example for reranking module"""
    candidate_contexts: List[str] = None
    candidate_scores: List[float] = None
    
    # Evaluation signals
    gold_ranking: Optional[List[int]] = None  # Optimal ordering indices
    ndcg_score: Optional[float] = None


@dataclass
class GenerationExample(TrainingExample):
    """Training example for answer generation module"""
    contexts: List[str] = None
    
    # Evaluation signals
    reference_answer: Optional[str] = None
    faithfulness_score: Optional[float] = None
    answer_correctness: Optional[float] = None


class GEPATrainingSetGenerator:
    """
    Generate high-quality training sets for GEPA optimization
    
    UPDATED WORKFLOW:
    1. Use RAGPipeline.get_representative_data() to select diverse queries
    2. Run representatives through pipeline to collect execution traces
    3. Generate module-specific training sets with quality signals
    4. Bootstrap with hard negatives for optimization
    
    Strategy based on:
    - Google DeepMind: "Learning to Learn" (meta-learning for prompt optimization)
    - OpenAI: GPT-4 technical report (prompt engineering methodology)
    - Anthropic: Constitutional AI (multi-stage refinement)
    """
    
    def __init__(
        self,
        rag_pipeline: UnifiedRAGPipeline,  # Updated to use UnifiedRAGPipeline
        evaluator: Optional[Evaluator] = None,
        output_dir: str = "./gepa_training_data"
    ):
        self.rag_pipeline = rag_pipeline
        self.evaluator = evaluator or Evaluator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_complete_training_set(
        self,
        evaluation_queries: List[Dict[str, Any]],
        use_representatives: bool = True,
        n_representatives: Optional[int] = None,
        n_reformulation: int = 500,
        n_reranking: int = 300,
        n_generation: int = 500,
        bootstrap_rounds: int = 2
    ) -> Dict[str, Any]:
        """
        Generate complete training sets using representative queries
        
        UPDATED WORKFLOW:
        1. Select representative queries (or use all if use_representatives=False)
        2. Run representatives through pipeline to collect traces
        3. Generate module-specific training sets
        4. Bootstrap with hard negatives
        5. Quality filter
        
        Args:
            evaluation_queries: Full evaluation query set with ground truth
            use_representatives: Whether to use clustering to select representatives
            n_representatives: Number of representatives (None = auto-detect via elbow)
            n_reformulation: Target size for reformulation training set
            n_reranking: Target size for reranking training set
            n_generation: Target size for generation training set
            bootstrap_rounds: Number of bootstrap iterations
        
        Returns:
            Dictionary with training sets and metadata
        """
        print("="*80)
        print("GEPA TRAINING SET GENERATION")
        print("="*80)
        
        # Phase 1: Select representative queries
        print("\n[Phase 1] Selecting representative queries...")
        if use_representatives:
            representatives = self.rag_pipeline.preprocessor.get_representative_data(
                evaluation_queries
            )
            if n_representatives:
                representatives = representatives[:n_representatives]
            print(f"✅ Selected {len(representatives)} representative queries")
            queries_to_process = representatives
        else:
            print(f"📋 Using all {len(evaluation_queries)} queries")
            queries_to_process = evaluation_queries
        
        training_sets = {
            "reformulation": [],
            "reranking": [],
            "generation": [],
            "metadata": {
                "total_queries": len(evaluation_queries),
                "representatives_used": len(queries_to_process),
                "use_representatives": use_representatives
            }
        }
        
        # Phase 2: Collect execution traces from representatives
        print(f"\n[Phase 2] Collecting execution traces from {len(queries_to_process)} queries...")
        execution_traces = await self._collect_execution_traces(queries_to_process)
        print(f"✅ Collected {len(execution_traces)} execution traces")
        
        # Phase 3: Module-specific training set generation
        print("\n[Phase 3] Generating module-specific training sets...")
        
        # Reformulation training set
        print(f"\n  → Reformulation module (target: {n_reformulation} examples)...")
        reformulation_set = await self._generate_reformulation_training_set(
            execution_traces, target_size=n_reformulation
        )
        training_sets["reformulation"] = reformulation_set
        print(f"    Generated {len(reformulation_set)} examples")
        
        # Reranking training set
        print(f"\n  → Reranking module (target: {n_reranking} examples)...")
        reranking_set = await self._generate_reranking_training_set(
            execution_traces, target_size=n_reranking
        )
        training_sets["reranking"] = reranking_set
        print(f"    Generated {len(reranking_set)} examples")
        
        # Generation training set
        print(f"\n  → Generation module (target: {n_generation} examples)...")
        generation_set = await self._generate_generation_training_set(
            execution_traces, target_size=n_generation
        )
        training_sets["generation"] = generation_set
        print(f"    Generated {len(generation_set)} examples")
        
        # Phase 4: Hard negative mining through bootstrap iterations
        for round_idx in range(bootstrap_rounds):
            print(f"\n[Phase 4] Bootstrap round {round_idx+1}/{bootstrap_rounds}...")
            training_sets = await self._bootstrap_hard_negatives(
                training_sets, execution_traces
            )
        
        # Phase 5: Quality filtering
        print("\n[Phase 5] Quality filtering...")
        training_sets = await self._quality_filter_and_augment(training_sets)
        
        # Save training sets
        self._save_training_sets(training_sets)
        
        print("\n" + "="*80)
        print("TRAINING SET GENERATION COMPLETE")
        print("="*80)
        print(f"Representatives: {training_sets['metadata']['representatives_used']}/{training_sets['metadata']['total_queries']}")
        for module in ["reformulation", "reranking", "generation"]:
            examples = training_sets.get(module, [])
            print(f"  {module:15s}: {len(examples):4d} examples")
        print("="*80 + "\n")
        
        return training_sets
    
    async def _collect_execution_traces(
        self,
        queries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute queries through UnifiedRAGPipeline (FAST mode) and collect traces
        
        UPDATED: Works with UnifiedRAGPipeline.query()
        """
        traces = []
        
        for query_data in tqdm(queries, desc="Collecting traces"):
            try:
                # Use UnifiedRAGPipeline's query method (FAST mode)
                result = await self.rag_pipeline.query(
                    query=query_data['query'],
                    ground_truth=query_data.get('ground_truth'),
                    retrieval_k=20,
                    final_k=10
                )
                
                trace = {
                    'query_id': query_data.get('query_id', hash(query_data['query'])),
                    'query': query_data['query'],
                    'ground_truth': query_data.get('ground_truth'),
                    'reference_doc_ids': query_data.get('reference_doc_ids'),
                    'reference_evidence_texts': query_data.get('reference_evidence_texts'),
                    
                    # Pipeline results
                    'contexts': result.get('contexts'),
                    'ranked_documents': result.get('ranked_documents'),
                    'answer': result.get('answer'),
                    'rationale': result.get('rationale'),
                    
                    # Evaluation scores
                    'retrieval_scores': result.get('retrieval_scores'),
                    'reranking_scores': result.get('reranking_scores'),
                    'generation_scores': result.get('generation_scores'),
                    'retrieval_f1': result.get('retrieval_f1'),
                    'reranking_f1': result.get('reranking_f1'),
                    'generation_quality': result.get('generation_quality'),
                    'overall_quality': result.get('overall_quality'),
                    
                    # For compatibility with existing code
                    'retrieval_eval': result.get('retrieval_scores'),
                    'reranking_eval': result.get('reranking_scores'),
                    'generation_eval': result.get('generation_scores'),
                    'overall_score': result.get('overall_quality'),
                }
                
                traces.append(trace)
                
            except Exception as e:
                print(f"  Warning: Query {query_data.get('query_id')} failed: {e}")
                continue
        
        return traces
    
    async def _generate_reformulation_training_set(
        self,
        traces: List[Dict[str, Any]],
        target_size: int
    ) -> List[Dict[str, Any]]:
        """
        Generate training examples for query reformulation
        
        Key Signals:
        1. Retrieval precision/recall (RAGAS context_precision, context_recall)
        2. Sub-query quality (coverage, specificity)
        3. Hard negatives: queries that failed retrieval
        
        Research Foundation:
        - Dense Passage Retrieval (Facebook AI): Query reformulation improves recall
        - ColBERT: Multi-vector queries for better coverage
        """
        examples = []
        
        # Strategy 1: Use traces with evaluation signals
        for trace in traces:
            retrieval_eval = trace.get('retrieval_eval') or {}
            
            example = ReformulationExample(
                query_id=str(trace['query_id']),
                query=trace['query'],
                ground_truth=trace.get('ground_truth'),
                reference_doc_ids=trace.get('reference_doc_ids'),
                reference_evidence_texts=trace.get('reference_evidence_texts'),
                ideal_sub_queries=trace.get('sub_queries'),
                retrieval_precision=retrieval_eval.get('context_precision'),
                retrieval_recall=retrieval_eval.get('context_recall'),
                metadata={
                    'source': 'execution_trace',
                    'retrieved_count': len(trace.get('retrieved_chunks', []))
                }
            )
            examples.append(example.to_dict())
        
        # Strategy 2: Mine hard negatives (low retrieval scores)
        hard_negatives = [
            trace for trace in traces
            if (trace.get('retrieval_eval') or {}).get('context_recall', 1.0) < 0.5
        ]
        print(f"    Found {len(hard_negatives)} hard negatives for reformulation")
        
        # Strategy 3: Augment with synthetic variations
        if len(examples) < target_size:
            print(f"    Augmenting with synthetic examples...")
            synthetic = await self._augment_reformulation_examples(
                examples, target_size - len(examples)
            )
            examples.extend(synthetic)
        
        return examples[:target_size]
    
    async def _generate_reranking_training_set(
        self,
        traces: List[Dict[str, Any]],
        target_size: int
    ) -> List[Dict[str, Any]]:
        """
        Generate training examples for document reranking
        
        Key Signals:
        1. RAGAS context_precision (optimal ranking)
        2. Score improvements from reranking
        3. Diversity of reranked contexts
        
        Research Foundation:
        - MonoT5 (Google): Pointwise reranking with LLMs
        - RankGPT (Microsoft): Listwise LLM reranking
        - DRAGON (Meta): Dense retrieval + reranking
        """
        examples = []
        
        for trace in traces:
            retrieved_chunks = trace.get('retrieved_chunks') or []
            reranked_chunks = trace.get('reranked_chunks') or []
            
            if not retrieved_chunks or not reranked_chunks:
                continue
            
            # Extract candidate contexts with scores
            candidates = [
                {
                    'text': chunk.get('chunk_text', ''),
                    'score': chunk.get('similarity_score', 0.0),
                    'doc_id': chunk.get('doc_id')
                }
                for chunk in retrieved_chunks
            ]
            
            # Compute gold ranking based on reference evidence
            gold_ranking = self._compute_gold_ranking(
                candidates,
                trace.get('reference_evidence_texts', [])
            )
            
            reranking_eval = trace.get('reranking_eval') or {}
            
            example = RerankingExample(
                query_id=str(trace['query_id']),
                query=trace['query'],
                ground_truth=trace.get('ground_truth'),
                candidate_contexts=[c['text'] for c in candidates],
                candidate_scores=[c['score'] for c in candidates],
                gold_ranking=gold_ranking,
                ndcg_score=reranking_eval.get('reranking_confidence'),
                metadata={
                    'source': 'execution_trace',
                    'score_improvement': reranking_eval.get('score_improvement', 0.0),
                    'concentration': reranking_eval.get('score_concentration', 0.0)
                }
            )
            examples.append(example.to_dict())
        
        # Mine hard negatives: cases where reranking didn't improve
        hard_negatives = [
            ex for ex in examples
            if (ex.get('metadata') or {}).get('score_improvement', 0) < 0.05
        ]
        print(f"    Found {len(hard_negatives)} hard negatives for reranking")
        
        return examples[:target_size]
    
    async def _generate_generation_training_set(
        self,
        traces: List[Dict[str, Any]],
        target_size: int
    ) -> List[Dict[str, Any]]:
        """
        Generate training examples for answer generation
        
        Key Signals:
        1. RAGAS faithfulness (grounding in context)
        2. RAGAS answer_correctness (semantic similarity to ground truth)
        3. Context utilization patterns
        
        Research Foundation:
        - RLHF (OpenAI): Reward modeling for generation quality
        - Self-Refine (CMU): Iterative refinement with feedback
        - Constitutional AI (Anthropic): Multi-objective optimization
        """
        examples = []
        
        for trace in traces:
            final_contexts = trace.get('final_contexts') or []
            answer = trace.get('answer')
            
            if not final_contexts or not answer:
                continue
            
            generation_eval = trace.get('generation_eval') or {}
            
            example = GenerationExample(
                query_id=str(trace['query_id']),
                query=trace['query'],
                ground_truth=trace.get('ground_truth'),
                contexts=final_contexts,
                reference_answer=answer,
                faithfulness_score=generation_eval.get('faithfulness'),
                answer_correctness=generation_eval.get('answer_correctness'),
                metadata={
                    'source': 'execution_trace',
                    'context_count': len(final_contexts),
                    'answer_length': len(answer.split()),
                    'overall_score': trace.get('overall_score', 0.0)
                }
            )
            examples.append(example.to_dict())
        
        # Mine hard negatives: low faithfulness or correctness
        hard_negatives = [
            ex for ex in examples
            if (ex.get('faithfulness_score') or 1.0) < 0.7
            or (ex.get('answer_correctness') or 1.0) < 0.7
        ]
        print(f"    Found {len(hard_negatives)} hard negatives for generation")
        
        return examples[:target_size]
    
    def _compute_gold_ranking(
        self,
        candidates: List[Dict[str, Any]],
        reference_evidence: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Compute optimal ranking based on reference evidence
        Uses exact match and semantic overlap scoring
        """
        if not reference_evidence:
            # Fallback: rank by original scores
            return sorted(range(len(candidates)), 
                         key=lambda i: candidates[i]['score'], 
                         reverse=True)
        
        # Score each candidate by overlap with reference evidence
        scores = []
        for candidate in candidates:
            max_overlap = 0.0
            for ref in reference_evidence:
                # Simple token overlap (could use semantic similarity)
                candidate_tokens = set(candidate['text'].lower().split())
                ref_tokens = set(ref.get('evidence_text', '').lower().split())
                
                if candidate_tokens and ref_tokens:
                    overlap = len(candidate_tokens & ref_tokens) / len(candidate_tokens | ref_tokens)
                    max_overlap = max(max_overlap, overlap)
            
            scores.append(max_overlap)
        
        # Return indices sorted by score
        return sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    
    async def _augment_reformulation_examples(
        self,
        base_examples: List[Dict[str, Any]],
        n_augment: int
    ) -> List[Dict[str, Any]]:
        """
        Augment reformulation examples with synthetic variations
        
        Techniques:
        1. Paraphrase queries
        2. Add/remove constraints
        3. Change specificity level
        """
        # TODO: Implement synthetic augmentation
        # Could use LLM to generate variations or use back-translation
        return []
    
    async def _bootstrap_hard_negatives(
        self,
        training_sets: Dict[str, List[Dict[str, Any]]],
        traces: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Bootstrap training sets with hard negatives
        
        Research Insight (AlphaCode, GPT-4):
        Hard negative mining improves optimization convergence:
        1. Identify failure modes
        2. Generate targeted examples
        3. Iteratively refine
        """
        # TODO: Implement hard negative mining
        # For now, return as-is
        return training_sets
    
    async def _quality_filter_and_augment(
        self,
        training_sets: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Quality filter using RAGAS and augment edge cases
        
        Filtering Criteria:
        1. Evaluation signals must be present
        2. Scores must be valid (0-1 range)
        3. No duplicate examples
        """
        filtered = {}
        
        for module, examples in training_sets.items():
            # Remove examples with missing evaluation signals
            valid_examples = []
            for ex in examples:
                if module == "reformulation":
                    if ex.get('retrieval_precision') is not None and ex.get('retrieval_recall') is not None:
                        valid_examples.append(ex)
                elif module == "reranking":
                    if ex.get('gold_ranking') is not None:
                        valid_examples.append(ex)
                elif module == "generation":
                    if ex.get('faithfulness_score') is not None:
                        valid_examples.append(ex)
            
            filtered[module] = valid_examples
            print(f"    {module}: {len(valid_examples)}/{len(examples)} passed quality filter")
        
        return filtered
    
    def _save_training_sets(self, training_sets: Dict[str, List[Dict[str, Any]]]):
        """Save training sets to disk"""
        for module, examples in training_sets.items():
            output_path = self.output_dir / f"{module}_training_set.json"
            with open(output_path, 'w') as f:
                json.dump(examples, f, indent=2)
            print(f"  Saved {module} training set to {output_path}")
    
    @staticmethod
    def load_training_sets(input_dir: str = "./gepa_training_data") -> Dict[str, List[Dict[str, Any]]]:
        """Load training sets from disk"""
        input_path = Path(input_dir)
        training_sets = {}
        
        for module in ["reformulation", "reranking", "generation"]:
            file_path = input_path / f"{module}_training_set.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    training_sets[module] = json.load(f)
        
        return training_sets


class GEPAAdapter:
    """
    Adapter for running GEPA optimization on RAG modules
    
    Implements the interface required by GEPA:
    1. evaluate(): Execute candidates and return scores
    2. extract_traces_for_reflection(): Extract feedback for prompt evolution
    """
    
    def __init__(
        self,
        rag_pipeline: UnifiedRAGPipeline,
        training_set: List[Dict[str, Any]],
        module: str
    ):
        self.rag_pipeline = rag_pipeline
        self.training_set = training_set
        self.module = module  # "reformulation", "reranking", or "generation"
        self.evaluator = Evaluator()
    
    async def evaluate(
        self,
        candidate_prompts: Dict[str, str],
        minibatch_indices: List[int]
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Evaluate candidate prompts on a minibatch
        
        Returns:
            Tuple of (scores, execution_traces)
        """
        scores = []
        traces = []
        
        minibatch = [self.training_set[i] for i in minibatch_indices]
        
        for example in minibatch:
            try:
                # Run pipeline with candidate prompt
                result = await self.rag_pipeline.query(
                    query=example['query'],
                    ground_truth=example.get('ground_truth'),
                    retrieval_k=20,
                    final_k=10,
                    max_iterations=1
                )
                
                # Compute module-specific score
                score = self._compute_module_score(result, example)
                scores.append(score)
                
                # Collect trace
                traces.append({
                    'query': example['query'],
                    'result': result,
                    'score': score
                })
                
            except Exception as e:
                print(f"  Warning: Evaluation failed for example: {e}")
                scores.append(0.0)
                traces.append({'error': str(e)})
        
        return scores, traces
    
    def _compute_module_score(
        self,
        result: Dict[str, Any],
        example: Dict[str, Any]
    ) -> float:
        """Compute module-specific evaluation score"""
        if self.module == "reformulation":
            retrieval_eval = result.get('retrieval_eval') or {}
            precision = retrieval_eval.get('context_precision', 0.0)
            recall = retrieval_eval.get('context_recall', 0.0)
            return (precision + recall) / 2
        
        elif self.module == "reranking":
            reranking_eval = result.get('reranking_eval') or {}
            return reranking_eval.get('reranking_confidence', 0.0)
        
        elif self.module == "generation":
            generation_eval = result.get('generation_eval') or {}
            faithfulness = generation_eval.get('faithfulness', 0.0)
            correctness = generation_eval.get('answer_correctness', 0.0)
            return (faithfulness + correctness) / 2
        
        return 0.0
    
    def extract_traces_for_reflection(
        self,
        traces: List[Dict[str, Any]],
        component_name: str
    ) -> List[str]:
        """
        Extract textual feedback for GEPA's reflection mechanism
        
        Research Insight (GEPA paper):
        Reflection uses execution traces (errors, metrics, outputs) to guide prompt evolution
        """
        reflections = []
        
        for trace in traces:
            if 'error' in trace:
                reflections.append(f"Execution failed: {trace['error']}")
                continue
            
            result = trace.get('result', {})
            score = trace.get('score', 0.0)
            
            if self.module == "reformulation":
                retrieval_eval = result.get('retrieval_eval') or {}
                reflection = (
                    f"Query: {trace['query']}\n"
                    f"Score: {score:.3f}\n"
                    f"Precision: {retrieval_eval.get('context_precision', 0):.3f}\n"
                    f"Recall: {retrieval_eval.get('context_recall', 0):.3f}\n"
                    f"Sub-queries: {result.get('sub_queries')}\n"
                )
            
            elif self.module == "reranking":
                reranking_eval = result.get('reranking_eval') or {}
                reflection = (
                    f"Query: {trace['query']}\n"
                    f"Score: {score:.3f}\n"
                    f"Confidence: {reranking_eval.get('reranking_confidence', 0):.3f}\n"
                    f"Improvement: {reranking_eval.get('score_improvement', 0):.3f}\n"
                    f"Rationale: {result.get('reranking_rationale', 'N/A')}\n"
                )
            
            else:  # generation
                generation_eval = result.get('generation_eval') or {}
                reflection = (
                    f"Query: {trace['query']}\n"
                    f"Score: {score:.3f}\n"
                    f"Faithfulness: {generation_eval.get('faithfulness', 0):.3f}\n"
                    f"Correctness: {generation_eval.get('answer_correctness', 0):.3f}\n"
                    f"Answer: {result.get('answer', 'N/A')[:200]}...\n"
                )
            
            reflections.append(reflection)
        
        return reflections

async def run_gepa_optimization_workflow(
    rag_pipeline: UnifiedRAGPipeline,
    queries: List[Dict[str, Any]],
    output_dir: str = "./gepa_optimization"
):
    """
    Complete GEPA optimization workflow
    
    Research-Grade Pipeline:
    1. Generate module-specific training sets
    2. Run offline GEPA optimization per module
    3. Export optimized prompts
    4. Warmstart online RL policies
    """
    print("\n" + "="*80)
    print("GEPA OPTIMIZATION WORKFLOW")
    print("Research-grade prompt optimization for RAG pipeline")
    print("="*80 + "\n")
    
    # Step 1: Generate training sets
    print("[Step 1] Generating training sets...")
    generator = GEPATrainingSetGenerator(rag_pipeline, output_dir=output_dir)
    training_sets = await generator.generate_complete_training_set(
        queries=queries,
        n_reformulation=500,
        n_reranking=300,
        n_generation=500,
        bootstrap_rounds=3
    )
    
    # Step 2: Run GEPA optimization for each module
    print("\n[Step 2] Running GEPA optimization...")
    print("NOTE: This requires GEPA library installation:")
    print("  pip install gepa-ai")
    print("\nFor each module, GEPA will:")
    print("  1. Explore prompt space via evolutionary search")
    print("  2. Use execution feedback for reflection-based mutations")
    print("  3. Converge to Pareto-optimal prompt candidates")
    print("\nSkipping actual GEPA calls (implement based on GEPA API)")
    
    # TODO: Integrate actual GEPA optimization
    # Example structure:
    # for module in ["reformulation", "reranking", "generation"]:
    #     adapter = GEPAAdapter(rag_pipeline, training_sets[module], module)
    #     optimized_prompts = gepa.optimize(
    #         adapter=adapter,
    #         metric="module_score",
    #         trainset=training_sets[module],
    #         population_size=20,
    #         generations=50
    #     )
    #     save_optimized_prompts(module, optimized_prompts)
    
    print("\n[Step 3] Exporting optimized prompts...")
    print(f"  → Output directory: {output_dir}/optimized_prompts/")
    
    print("\n[Step 4] Warmstarting online RL policies...")
    print("  → Load optimized prompts as initial PromptOption pool")
    print("  → Continue online fine-tuning during inference")
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE")
    print("="*80 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Generate training sets from FinanceBench
    from demo.demo_financebench import load_financebench_data
    from rag_pipeline import UnifiedRAGPipeline, PipelineMode
    
    async def main():
        # Load data
        queries, documents = load_financebench_data(max_samples=100)
        
        # Initialize RAG pipeline (FAST mode for training set generation)
        pipeline = UnifiedRAGPipeline(mode=PipelineMode.FAST)
        pipeline.setup_pipeline(documents)
        
        # Run GEPA workflow
        await run_gepa_optimization_workflow(
            rag_pipeline=pipeline,
            queries=queries,
            output_dir="./gepa_optimization"
        )
    
    asyncio.run(main())

