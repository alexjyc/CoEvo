"""
Demo script for RAG Pipeline using FinanceBench Dataset
Demonstrates the complete workflow from data loading to evaluation
"""

import os
from datasets import load_dataset
from typing import List, Dict, Any
import pandas as pd

from rag_pipeline import create_pipeline_from_config


def load_financebench_data(split: str = "train", max_samples: int = 50) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load FinanceBench dataset and prepare documents and queries
    
    Args:
        split: Dataset split to load ('train', 'test', etc.)
        max_samples: Maximum number of samples to load for demo
        
    Returns:
        Tuple of (documents, evaluation_queries)
    """
    print(f"Loading FinanceBench dataset ({split} split)...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset("PatronusAI/financebench", split=split)
    
    # Limit samples for demo purposes
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    print(f"Loaded {len(dataset)} samples from FinanceBench")
    
    # Extract unique documents from the dataset
    documents = []
    
    evaluation_queries = []
    
    for item in dataset:
        # Extract document text (assuming 'evidence' contains the document text)
        if 'evidence' in item and item['evidence']:
            evidence = item['evidence']
            for doc_text in evidence:  # Avoid duplicates
                documents.append(
                    {
                        'text': doc_text['evidence_text_full_page'],
                    }
                )
        
        # Extract query and answer for evaluation
        if 'question' in item and 'answer' in item:
            evaluation_queries.append({
                'query': item['question'],
                'ground_truth': item['answer']
            })

    
    print(f"Extracted {len(documents)} unique documents and {len(evaluation_queries)} evaluation queries")
    
    return documents, evaluation_queries


def run_basic_demo():
    """
    Run a basic demonstration of the RAG pipeline
    """
    print("=== RAG Pipeline Demo with FinanceBench Dataset ===\n")
    
    # Load data
    documents, evaluation_queries = load_financebench_data(max_samples=20)
    
    # Create and setup pipeline
    print("Setting up RAG pipeline...")
    pipeline = create_pipeline_from_config()
    
    # Process documents
    pipeline.load_and_process_documents(documents)
    
    # Save index for future use
    pipeline.save_index("financebench_index.faiss", "financebench_metadata.pkl")
    
    # Demo with a few sample queries
    print("\n=== Running Sample Queries ===")
    
    sample_queries = evaluation_queries[:3]  # Take first 3 queries
    
    for i, query_data in enumerate(sample_queries, 1):
        print(f"\n--- Sample Query {i} ---")
        print(f"Question: {query_data['query']}")
        print(f"Expected Answer: {query_data['ground_truth']}")
        
        # Process query through pipeline
        result = pipeline.query(
            query=query_data['query'],
            retrieval_k=8,
            final_k=3,
            use_financial_prompt=True,
            max_tokens=300
        )
        
        print(f"Generated Answer: {result['response']}")
        print(f"Retrieved {result['retrieval_stats']['initial_retrieved']} chunks, used {result['retrieval_stats']['final_processed']}")
        print(f"Average similarity: {result['retrieval_stats']['avg_similarity']:.3f}")


def run_evaluation_demo():
    """
    Run evaluation demonstration
    """
    print("\n=== RAG Pipeline Evaluation Demo ===\n")
    
    # Load data
    documents, evaluation_queries = load_financebench_data(max_samples=10)
    
    # Create and setup pipeline
    pipeline = create_pipeline_from_config()
    
    # Try to load existing index, otherwise create new one
    try:
        pipeline.load_from_saved_index("financebench_index.faiss", "financebench_metadata.pkl")
        print("Loaded existing index")
    except:
        print("Creating new index...")
        pipeline.load_and_process_documents(documents)
        pipeline.save_index("financebench_index.faiss", "financebench_metadata.pkl")
    
    # Run evaluation on subset of queries
    eval_llm_gen_data = []
    for query_data in evaluation_queries[:5]:
        result = pipeline.query(query=query_data['query'])
        eval_llm_gen_data.append({
            'query': query_data['query'],
            'ground_truth': query_data['ground_truth'],
            'generated_response': result['response'],
            'context': result['processed_chunks']
        })
    
    print(f"Running evaluation on {len(eval_llm_gen_data)} queries...")
    batch_results = pipeline.batch_judge(eval_llm_gen_data)
    
    # Display results
    print("\n=== Evaluation Results ===")    
    print("\nIndividual Query Results:")
    for i, result in enumerate(batch_results):
        final_eval = result.get('final_assessment', False)
        confidence = result.get('confidence_score', 0.0)
        status = "✅" if final_eval else "❌"

        print(f"   Result: {'PASS' if final_eval else 'FAIL'} (Confidence: {confidence:.3f})")
    

def interactive_demo():
    """
    Interactive demo allowing user to ask custom questions
    """
    print("\n=== Interactive RAG Demo ===")
    print("Ask financial questions and see how the RAG pipeline responds!")
    print("Type 'quit' to exit.\n")
    
    # Setup pipeline
    try:
        pipeline = create_pipeline_from_config()
        pipeline.load_from_saved_index("financebench_index.faiss", "financebench_metadata.pkl")
        print("Loaded existing FinanceBench index")
    except:
        print("No existing index found. Loading sample data...")
        documents, _ = load_financebench_data(max_samples=20)
        pipeline = create_pipeline_from_config()
        pipeline.load_and_process_documents(documents)
        pipeline.save_index("financebench_index.faiss", "financebench_metadata.pkl")
    
    print("Pipeline ready! Ask your questions:\n")
    
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Thanks for using the RAG pipeline demo!")
            break
        
        if not query:
            print("Please enter a question or 'quit' to exit.")
            continue
        
        try:
            result = pipeline.query(
                query=query,
                retrieval_k=8,
                final_k=3,
                use_financial_prompt=True,
                max_tokens=400
            )
            
            print(f"\nAnswer: {result['response']}")
            print(f"(Based on {result['retrieval_stats']['final_processed']} relevant document chunks)")
            print("-" * 80)
            
        except Exception as e:
            print(f"Error processing query: {e}")
            print("-" * 80)


def main():
    """
    Main demo function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Pipeline Demo with FinanceBench")
    parser.add_argument("--demo", choices=["basic", "eval", "interactive"], 
                       default="basic", help="Type of demo to run")
    
    args = parser.parse_args()
    
    # Check if OpenAI API key is set
    # if not os.getenv('OPENAI_API_KEY'):
    #     print("ERROR: OPENAI_API_KEY environment variable not set!")
    #     print("Please set your OpenAI API key before running the demo.")
    #     print("You can create a .env file with: OPENAI_API_KEY=your_key_here")
    #     return
    
    if args.demo == "basic":
        run_basic_demo()
    elif args.demo == "eval":
        run_evaluation_demo()
    elif args.demo == "interactive":
        interactive_demo()
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
