"""
Quick Start: Self-Improving RAG Pipeline
A minimal example to get started quickly
"""

import os
from dotenv import load_dotenv
from self_improving_rag import create_self_improving_pipeline


def quickstart():
    """Minimal example of self-improving RAG"""
    
    # Load environment
    load_dotenv()
    
    # Create pipeline
    print("🚀 Creating self-improving RAG pipeline...")
    pipeline = create_self_improving_pipeline(
        generation_model="gpt-4o-mini",
        retrieval_method="hybrid",
        max_iterations=3,
        convergence_threshold=0.85
    )
    
    # Sample documents
    documents = [
        {
            "text": """
            Apple Inc. reported total revenue of $394.3 billion for fiscal year 2023.
            The company's net income was $97.0 billion, representing a 24.6% profit margin.
            iPhone sales accounted for 52% of total revenue at $205.5 billion.
            Services revenue reached a record $85.2 billion, growing 9.1% year-over-year.
            The company returned $99.2 billion to shareholders through dividends and share buybacks.
            """,
            "metadata": {"company": "Apple", "year": "2023", "source": "Annual Report"}
        },
        {
            "text": """
            Microsoft Corporation's fiscal 2023 revenue was $211.9 billion, up 7% year-over-year.
            Operating income increased to $88.5 billion with an operating margin of 41.8%.
            Cloud revenue (Azure, Office 365, Dynamics 365) reached $111.6 billion.
            LinkedIn revenue surpassed $15 billion for the first time.
            Gaming revenue, including Xbox and Activision Blizzard, totaled $21.5 billion.
            """,
            "metadata": {"company": "Microsoft", "year": "2023", "source": "Annual Report"}
        },
        {
            "text": """
            Tesla's 2023 financial results showed total revenue of $96.8 billion, up 19% from 2022.
            The company delivered 1.81 million vehicles globally in 2023.
            Automotive revenue was $82.4 billion with an automotive gross margin of 18.2%.
            Energy generation and storage revenue reached $6.0 billion, growing 54% year-over-year.
            Net income for 2023 was $15.0 billion, yielding a 15.5% net profit margin.
            """,
            "metadata": {"company": "Tesla", "year": "2023", "source": "Annual Report"}
        }
    ]
    
    # Load documents
    print(f"📚 Loading {len(documents)} documents...")
    pipeline.load_and_process_documents(documents, use_context=True)
    
    # Example queries
    queries = [
        {
            "query": "What was Apple's total revenue in 2023?",
            "ground_truth": "Apple Inc. reported total revenue of $394.3 billion for fiscal year 2023."
        },
        {
            "query": "How much did Microsoft's cloud services generate in revenue?",
            "ground_truth": "Microsoft's cloud revenue reached $111.6 billion in fiscal 2023."
        },
        {
            "query": "What was Tesla's vehicle delivery count in 2023?",
            "ground_truth": "Tesla delivered 1.81 million vehicles globally in 2023."
        }
    ]
    
    # Process each query
    print(f"\n{'='*100}")
    print("PROCESSING QUERIES WITH SELF-IMPROVEMENT")
    print(f"{'='*100}\n")
    
    for i, q in enumerate(queries, 1):
        print(f"\n[Query {i}/{len(queries)}]")
        print(f"Question: {q['query']}")
        print("-" * 100)
        
        result = pipeline.query(
            query=q['query'],
            ground_truth=q['ground_truth'],
            retrieval_k=10,
            final_k=5,
            max_iterations=3
        )
        
        # Display results
        print(f"\n✅ Final Answer: {result.get('answer', 'N/A')}")
        print(f"📊 Overall Score: {result.get('overall_score', 0.0):.3f}")
        print(f"🔄 Iterations: {len(result.get('improvement_history', []))}")
        
        if result.get('improvement_history'):
            hist = result['improvement_history']
            if len(hist) > 1:
                improvement = hist[-1]['overall_score'] - hist[0]['overall_score']
                print(f"📈 Improvement: {improvement:+.3f} ({hist[0]['overall_score']:.3f} → {hist[-1]['overall_score']:.3f})")
        
        print("-" * 100)
    
    print(f"\n{'='*100}")
    print("✨ Done! The pipeline automatically improved responses through iteration.")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    quickstart()

