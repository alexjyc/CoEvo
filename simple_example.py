"""
Simple Example of RAG Pipeline Usage
Demonstrates basic functionality with minimal setup
"""

from rag_pipeline import create_pipeline_from_config


def simple_example():
    """
    Simple example showing basic RAG pipeline usage
    """
    print("=== Simple RAG Pipeline Example ===\n")
    
    # Sample financial documents
    documents = [
        {
            "text": """Apple Inc. reported record quarterly revenue of $123.9 billion for Q1 2024, 
                      up 2% year over year. iPhone revenue was $69.7 billion, up 6% from the 
                      prior year quarter. Services revenue reached a new all-time high of 
                      $23.1 billion, up 11% year over year."""
        },
        {
            "text": """Microsoft Corporation announced revenue of $62.0 billion for Q2 2024, 
                      representing a 18% increase year-over-year. Productivity and Business 
                      Processes revenue was $19.0 billion and increased 13%. More Personal 
                      Computing revenue was $16.9 billion and increased 19%."""
        },
        {
            "text": """Tesla's total revenue for Q4 2023 was $25.2 billion, up 3% year-over-year. 
                      Automotive revenue was $21.6 billion in Q4, up 1% year-over-year. 
                      Energy generation and storage revenue increased 54% year-over-year to 
                      $1.4 billion in Q4."""
        }
    ]
    
    print("1. Creating RAG pipeline...")
    try:
        # Create pipeline (requires OPENAI_API_KEY in environment)
        pipeline = create_pipeline_from_config()
        
        print("2. Processing documents...")
        pipeline.load_and_process_documents(documents)
        
        print("3. Running sample queries...\n")
        
        # Sample queries
        queries = [
            "What was Apple's revenue in Q1 2024?",
            "How much did Microsoft's revenue increase?",
            "Which company had the highest revenue growth in energy?",
            "Compare the revenue performance of Apple and Microsoft"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"Query {i}: {query}")
            
            result = pipeline.query(
                query=query,
                retrieval_k=5,
                final_k=3,
                use_financial_prompt=True,
                max_tokens=150
            )
            
            print(f"Answer: {result['response']}")
            print(f"Sources: {result['retrieval_stats']['final_processed']} document chunks")
            print(f"Confidence: {result['retrieval_stats']['avg_similarity']:.3f}")
            print("-" * 60)
        
        print("\n4. Pipeline info:")
        info = pipeline.get_pipeline_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Installed all requirements: pip install -r requirements.txt")
        print("3. Valid OpenAI API key with sufficient credits")


if __name__ == "__main__":
    simple_example()
