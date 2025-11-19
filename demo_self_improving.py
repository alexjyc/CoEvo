"""
Demo: Self-Improving RAG Pipeline with FinanceBench Dataset
Demonstrates iterative refinement using LangGraph and RAGAS evaluation
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset

from self_improving_rag import SelfImprovingRAG


def load_financebench_data(split: str = "train", max_samples: int = 5):
    """
    Load FinanceBench dataset from HuggingFace
    
    Args:
        split: Dataset split to load
        max_samples: Maximum number of samples to process
    
    Returns:
        Tuple of (documents, queries)
    """
    print(f"Loading FinanceBench dataset (split={split}, max={max_samples})...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset("PatronusAI/financebench", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Extract documents (unique)
    documents = []
    seen_docs = set()
    
    for item in dataset:
        doc_name = item.get('doc_name', 'unknown')
        doc_text = item.get('evidence', '')  # FinanceBench has 'evidence' field
        
        if doc_text and doc_name not in seen_docs:
            documents.append({
                'text': doc_text,
                'metadata': {
                    'doc_name': doc_name,
                    'source': 'financebench'
                }
            })
            seen_docs.add(doc_name)
    
    # Extract queries
    queries = []
    for item in dataset:
        queries.append({
            'query': item.get('question', ''),
            'ground_truth': item.get('answer', ''),
            'doc_name': item.get('doc_name', ''),
            'company': item.get('company', ''),
            'year': item.get('year', '')
        })
    
    print(f"Loaded {len(documents)} unique documents and {len(queries)} queries")
    return documents, queries


def demo_single_query(pipeline: SelfImprovingRAG, query_data: dict):
    """
    Demonstrate self-improving pipeline on a single query
    
    Args:
        pipeline: Initialized SelfImprovingRAG instance
        query_data: Query dictionary with 'query' and 'ground_truth'
    """
    print(f"\n{'='*100}")
    print(f"DEMO: Single Query with Self-Improvement")
    print(f"{'='*100}")
    print(f"Query: {query_data['query']}")
    print(f"Company: {query_data.get('company', 'N/A')}")
    print(f"Year: {query_data.get('year', 'N/A')}")
    print(f"{'='*100}\n")
    
    # Run self-improving query
    result = pipeline.query(
        query=query_data['query'],
        ground_truth=query_data['ground_truth'],
        retrieval_k=30,
        final_k=10,
        max_iterations=3,
        convergence_threshold=0.85
    )
    
    # Display results
    print(f"\n{'='*100}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*100}")
    
    if result.get('improvement_history'):
        print("\nImprovement History:")
        for i, hist in enumerate(result['improvement_history']):
            print(f"\n  Iteration {i}:")
            print(f"    Overall Score: {hist.get('overall_score', 0.0):.3f}")
            print(f"    Faithfulness:  {hist.get('faithfulness_score', 0.0):.3f}")
            print(f"    Relevancy:     {hist.get('answer_relevancy_score', 0.0):.3f}")
            print(f"    Precision:     {hist.get('context_precision_score', 0.0):.3f}")
            if hist.get('context_recall_score') is not None:
                print(f"    Recall:        {hist.get('context_recall_score', 0.0):.3f}")
            print(f"    Contexts Used: {hist.get('num_contexts', 0)}")
    
    print(f"\nFinal Answer:")
    print(f"  {result.get('answer', 'N/A')}")
    
    print(f"\nGround Truth:")
    print(f"  {query_data.get('ground_truth', 'N/A')}")
    
    print(f"\n{'='*100}\n")
    
    return result


def demo_batch_comparison(pipeline: SelfImprovingRAG, queries: list, num_queries: int = 3):
    """
    Compare regular vs self-improving pipeline on multiple queries
    
    Args:
        pipeline: Initialized SelfImprovingRAG instance
        queries: List of query dictionaries
        num_queries: Number of queries to process
    """
    print(f"\n{'='*100}")
    print(f"DEMO: Batch Comparison (Regular vs Self-Improving)")
    print(f"{'='*100}\n")
    
    results = []
    
    for i, query_data in enumerate(queries[:num_queries]):
        print(f"\n[{i+1}/{num_queries}] Processing: {query_data['query'][:80]}...")
        
        # Run with self-improvement
        result = pipeline.query(
            query=query_data['query'],
            ground_truth=query_data['ground_truth'],
            retrieval_k=20,
            final_k=8,
            max_iterations=3,
            convergence_threshold=0.85
        )
        
        results.append({
            'query': query_data['query'],
            'ground_truth': query_data['ground_truth'],
            'iterations': len(result.get('improvement_history', [])),
            'initial_score': result['improvement_history'][0]['overall_score'] if result.get('improvement_history') else 0.0,
            'final_score': result.get('overall_score', 0.0),
            'improvement': (result.get('overall_score', 0.0) - 
                          (result['improvement_history'][0]['overall_score'] if result.get('improvement_history') else 0.0)),
            'converged': result.get('overall_score', 0.0) >= 0.85
        })
    
    # Summary statistics
    print(f"\n{'='*100}")
    print(f"BATCH RESULTS SUMMARY")
    print(f"{'='*100}\n")
    
    total_improvement = sum(r['improvement'] for r in results)
    avg_improvement = total_improvement / len(results) if results else 0.0
    converged_count = sum(1 for r in results if r['converged'])
    
    print(f"Queries Processed: {len(results)}")
    print(f"Average Initial Score: {sum(r['initial_score'] for r in results) / len(results):.3f}")
    print(f"Average Final Score: {sum(r['final_score'] for r in results) / len(results):.3f}")
    print(f"Average Improvement: {avg_improvement:+.3f}")
    print(f"Converged (≥0.85): {converged_count}/{len(results)} ({converged_count/len(results)*100:.1f}%)")
    
    print(f"\nDetailed Results:")
    for i, r in enumerate(results):
        print(f"\n  Query {i+1}:")
        print(f"    {r['query'][:80]}...")
        print(f"    Initial Score: {r['initial_score']:.3f}")
        print(f"    Final Score:   {r['final_score']:.3f}")
        print(f"    Improvement:   {r['improvement']:+.3f}")
        print(f"    Iterations:    {r['iterations']}")
        print(f"    Converged:     {'✓' if r['converged'] else '✗'}")
    
    print(f"\n{'='*100}\n")
    
    return results


def demo_parameter_tuning(pipeline: SelfImprovingRAG, query_data: dict):
    """
    Demonstrate how different parameters affect the self-improvement process
    
    Args:
        pipeline: Initialized SelfImprovingRAG instance
        query_data: Query dictionary
    """
    print(f"\n{'='*100}")
    print(f"DEMO: Parameter Tuning Comparison")
    print(f"{'='*100}\n")
    
    configs = [
        {"name": "Conservative", "retrieval_k": 15, "final_k": 5, "max_iterations": 2},
        {"name": "Balanced", "retrieval_k": 20, "final_k": 8, "max_iterations": 3},
        {"name": "Aggressive", "retrieval_k": 30, "final_k": 12, "max_iterations": 4},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        print(f"  retrieval_k={config['retrieval_k']}, final_k={config['final_k']}, max_iterations={config['max_iterations']}")
        
        result = pipeline.query(
            query=query_data['query'],
            ground_truth=query_data['ground_truth'],
            retrieval_k=config['retrieval_k'],
            final_k=config['final_k'],
            max_iterations=config['max_iterations'],
            convergence_threshold=0.85
        )
        
        results.append({
            'config': config['name'],
            'final_score': result.get('overall_score', 0.0),
            'iterations': len(result.get('improvement_history', [])),
            'answer': result.get('answer', 'N/A')
        })
    
    # Compare results
    print(f"\n{'='*100}")
    print(f"PARAMETER COMPARISON RESULTS")
    print(f"{'='*100}\n")
    
    for r in results:
        print(f"\n{r['config']} Configuration:")
        print(f"  Final Score: {r['final_score']:.3f}")
        print(f"  Iterations:  {r['iterations']}")
        print(f"  Answer:      {r['answer'][:100]}...")
    
    best_config = max(results, key=lambda x: x['final_score'])
    print(f"\n🏆 Best Configuration: {best_config['config']} (Score: {best_config['final_score']:.3f})")
    print(f"\n{'='*100}\n")
    
    return results


def main():
    """Main demo function"""
    load_dotenv()
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    # Initialize pipeline
    print("Initializing Self-Improving RAG Pipeline...")
    pipeline = SelfImprovingRAG(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        generation_model="gpt-4o-mini",
        evaluator_model="gpt-4o-mini",
        retrieval_method="hybrid",
        max_iterations=3,
        convergence_threshold=0.85
    )
    
    # Load data
    documents, queries = load_financebench_data(split="train", max_samples=10)
    
    # Process documents
    pipeline.load_and_process_documents(documents, use_context=True)
    
    # Demo 1: Single query with detailed output
    if queries:
        print("\n" + "="*100)
        print("DEMO 1: Single Query Self-Improvement")
        print("="*100)
        demo_single_query(pipeline, queries[0])
    
    # Demo 2: Batch comparison
    if len(queries) >= 3:
        print("\n" + "="*100)
        print("DEMO 2: Batch Processing")
        print("="*100)
        demo_batch_comparison(pipeline, queries, num_queries=3)
    
    # Demo 3: Parameter tuning
    if queries:
        print("\n" + "="*100)
        print("DEMO 3: Parameter Tuning")
        print("="*100)
        demo_parameter_tuning(pipeline, queries[0])
    
    print("\n✅ All demos completed!")


if __name__ == "__main__":
    main()

