"""
Comprehensive demo of Langfuse integration with RAGAS evaluation
Shows tracing, individual evaluation, batch evaluation, and historical analysis
"""

import os
import asyncio
from dotenv import load_dotenv
from rag_pipeline import create_pipeline_from_config
from retrieval import RetrievalMethod

# Load environment variables
load_dotenv()


# def demo_single_query_with_evaluation():
#     """Demonstrate single query with RAGAS evaluation"""
#     print("=== Single Query with RAGAS Evaluation Demo ===\n")
    
#     # Sample financial documents
#     documents = [
#         {
#             "text": """Apple Inc. reported record quarterly revenue of $123.9 billion for Q1 2024, 
#                       up 2% year over year. iPhone revenue was $69.7 billion, up 6% from the 
#                       prior year quarter. Services revenue reached a new all-time high of 
#                       $23.1 billion, up 11% year over year."""
#         },
#         {
#             "text": """Microsoft Corporation announced revenue of $62.0 billion for Q2 2024, 
#                       representing an 18% increase year-over-year. Azure and other cloud services 
#                       revenue grew 31% year-over-year. Office Commercial products and services 
#                       revenue increased 15% year-over-year."""
#         },
#         {
#             "text": """Tesla's revenue for Q4 2023 was $25.2 billion, up 3% year-over-year. 
#                       Vehicle deliveries reached 484,507 units in Q4 2023. Energy generation and 
#                       storage revenue increased 54% year-over-year to $1.4 billion."""
#         }
#     ]
    
#     # Create and setup pipeline
#     pipeline = create_pipeline_from_config()
#     pipeline.load_and_process_documents(documents)
    
#     # Test query with ground truth
#     query = "What was Apple's total revenue in Q1 2024?"
#     ground_truth = "Apple Inc. reported record quarterly revenue of $123.9 billion for Q1 2024, up 2% year over year."
    
#     print(f"Query: {query}")
#     print(f"Ground Truth: {ground_truth}\n")
    
#     # Run query with RAGAS evaluation
#     result = pipeline.query_with_evaluation(
#         query=query,
#         ground_truth=ground_truth,
#         user_id="demo_user",
#         session_id="demo_session_1",
#         retrieval_k=8,
#         final_k=3,
#     )
    
#     # Display results
#     print("=== Results ===")
#     print(f"Generated Answer: {result['response']}")
#     print(f"Retrieval Method: {result['retrieval_stats']['retrieval_method']}")
#     print(f"Chunks Used: {result['retrieval_stats']['final_processed']}")
#     print(f"Avg Similarity: {result['retrieval_stats']['avg_similarity']:.3f}")
    
#     if 'ragas_evaluation' in result:
#         ragas_scores = result['ragas_evaluation']['scores']
#         print(f"\n📊 RAGAS Evaluation Scores:")
        
#         for metric, score in ragas_scores.items():
#             print(f"  • {metric}: {score:.3f}")
        
#         print(f"\n🔗 Langfuse Trace ID: {result['ragas_evaluation']['trace_id']}")
    
#     print("\n" + "="*80)


async def demo_batch_evaluation():
    """Demonstrate batch evaluation with RAGAS"""
    print("\n=== Batch RAGAS Evaluation Demo ===\n")
    
    # Create pipeline
    pipeline = create_pipeline_from_config()
    
    # Use existing saved index if available
    try:
        pipeline.load_from_saved_index("financebench_index.faiss", "financebench_metadata.pkl")
        print("✅ Loaded existing FinanceBench index")
    except:
        print("⚠️  No existing index found. Please run basic setup first.")
        return
    
    # Batch evaluation data
    evaluation_data = [
        {
            "query": "What was Apple's Q1 2024 revenue?",
            "ground_truth": "Apple reported $123.9 billion in Q1 2024 revenue",
        },
        {
            "query": "How much did Microsoft's revenue grow?",
            "ground_truth": "Microsoft's revenue grew 18% year-over-year",
        },
        {
            "query": "What was Tesla's Q4 2023 revenue?", 
            "ground_truth": "Tesla's Q4 2023 revenue was $25.2 billion",
        },
        {
            "query": "Which company had the highest cloud growth?",
            "ground_truth": "Microsoft had high cloud growth with Azure growing 31%",
        }
    ]
    
    print(f"Running batch evaluation on {len(evaluation_data)} queries...\n")
    
    # Run batch evaluation
    # for data in evaluation_data:
    #     result = await pipeline.query_with_eval(
    #         data['query'],
    #         data['ground_truth'],
    #         retrieval_k=10,
    #         final_k=5,
    #     )
    #     print('result', result)

    batch_result = await pipeline.batch_query(evaluation_data)
    print('batch_result', batch_result)
   





async def main():
    """Run all Langfuse demos"""
    
    # Configuration check
    # demo_langfuse_configuration()
    
    try:
        # Run evaluation demos
        await demo_batch_evaluation()
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("\n🔧 Setup Requirements:")
        print("   1. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
        print("   2. Set OPENAI_API_KEY")
        print("   3. Install: pip install langfuse ragas langchain-openai")
        print("   4. Create account at https://langfuse.com")


if __name__ == "__main__":
    asyncio.run(main())