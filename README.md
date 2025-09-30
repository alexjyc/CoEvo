# RAG Pipeline Optimization Project

A comprehensive Retrieval-Augmented Generation (RAG) pipeline implementation using OpenAI API, designed for financial document analysis with the FinanceBench dataset.

## Project Overview

This project implements a complete RAG pipeline following a modular architecture with five core components:

1. **Preprocessor**: Document chunking, embedding generation, and FAISS index creation
2. **Retrieval Module**: Basic similarity search using FAISS
3. **Reranker**: Top-k selection and deduplication
4. **Generation Module**: LLM integration for response generation
5. **Evaluation Module**: Comprehensive metric calculation

## Features

- ✅ Document chunking with configurable overlap
- ✅ OpenAI embedding generation (text-embedding-3-small)
- ✅ FAISS vector database for efficient similarity search
- ✅ Advanced reranking (deduplication, length filtering, source diversification)
- ✅ OpenAI GPT integration for response generation
- ✅ Comprehensive evaluation metrics (BLEU, ROUGE, retrieval metrics)
- ✅ FinanceBench dataset integration
- ✅ Financial domain-specific prompting
- ✅ Batch processing and evaluation
- ✅ Interactive demo interface

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd rag-optimization
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up OpenAI API key**:
```bash
# Create a .env file
cp .env.example .env
# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

## Quick Start

### Basic Usage

```python
from rag_pipeline import create_pipeline_from_config

# Create pipeline
pipeline = create_pipeline_from_config()

# Load documents (example)
documents = [
    {"text": "Company XYZ reported revenue of $100M in Q1 2023..."},
    {"text": "The financial statements show a 15% increase in profits..."},
    # ... more documents
]

# Process documents
pipeline.load_and_process_documents(documents)

# Query the pipeline
result = pipeline.query(
    query="What was the revenue for Q1 2023?",
    retrieval_k=10,
    final_k=5,
    use_financial_prompt=True
)

print(result['response'])
```

### Using FinanceBench Dataset

```python
# Run the demo script
python demo_financebench.py --demo basic

# Or for evaluation
python demo_financebench.py --demo evaluation

# Or for interactive mode
python demo_financebench.py --demo interactive
```

## Project Structure

```
rag-optimization/
├── preprocessor.py          # Document processing and embedding generation
├── retrieval.py            # Similarity search and context retrieval
├── reranker.py             # Top-k selection and filtering
├── generation.py           # LLM integration for response generation
├── evaluation.py           # Comprehensive evaluation metrics
├── rag_pipeline.py         # Main pipeline orchestrator
├── demo_financebench.py    # FinanceBench dataset demo
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## Core Components

### 1. Preprocessor
- **Text Chunking**: Splits documents into overlapping chunks
- **Embedding Generation**: Uses OpenAI's text-embedding-3-small model
- **FAISS Index**: Creates efficient vector search index
- **Persistence**: Save/load functionality for indexes

### 2. Retrieval Module
- **Similarity Search**: FAISS-based vector similarity
- **Query Embedding**: Converts queries to embeddings
- **Context Formatting**: Prepares retrieved chunks for generation
- **Statistics**: Retrieval quality metrics

### 3. Reranker
- **Top-k Selection**: Selects most relevant chunks
- **Deduplication**: Removes similar/duplicate content
- **Length Filtering**: Filters chunks by quality criteria
- **Source Diversification**: Ensures variety in source documents

### 4. Generation Module
- **OpenAI Integration**: GPT-4 integration for response generation
- **Custom Prompting**: Financial domain-specific prompts
- **Token Management**: Configurable output length
- **Batch Processing**: Multiple query handling

### 5. Evaluation Module
- **Retrieval Metrics**: Precision@K, Recall, MRR, F1-score
- **Generation Metrics**: BLEU, ROUGE-1, exact match
- **Pipeline Metrics**: End-to-end performance assessment
- **Batch Evaluation**: Aggregate statistics computation

## Configuration Options

The pipeline supports various configuration options:

```python
pipeline = RAGPipeline(
    openai_api_key="your-key",
    chunk_size=500,           # Characters per chunk
    chunk_overlap=50,         # Overlap between chunks
    generation_model="gpt-4", # OpenAI model
    temperature=0.1           # Generation randomness
)
```

Query parameters:
- `retrieval_k`: Number of chunks to retrieve initially
- `final_k`: Final number of chunks for generation
- `remove_duplicates`: Enable deduplication
- `filter_length`: Enable length filtering
- `diversify`: Enable source diversification
- `use_financial_prompt`: Use domain-specific prompting

## Evaluation

The pipeline includes comprehensive evaluation capabilities:

```python
# Single query evaluation
result = pipeline.evaluate_query(
    query="What was the company's revenue?",
    ground_truth="The revenue was $100M"
)

# Batch evaluation
evaluation_data = [
    {"query": "Question 1?", "ground_truth": "Answer 1"},
    {"query": "Question 2?", "ground_truth": "Answer 2"},
]
batch_results = pipeline.batch_evaluate(evaluation_data)
```

### Evaluation Metrics

- **Retrieval Metrics**:
  - Precision@K, Recall, F1-score
  - Mean Reciprocal Rank (MRR)
  - Average similarity scores

- **Generation Metrics**:
  - BLEU score (text similarity)
  - ROUGE-1 (word overlap)
  - Lexical diversity
  - Response length statistics

## FinanceBench Integration

The project specifically integrates with the FinanceBench dataset:

- **Dataset**: [PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench)
- **Domain**: Financial document analysis
- **Format**: Question-answering with evidence documents
- **Evaluation**: Ground truth answers for metric calculation

## Demo Modes

### 1. Basic Demo
```bash
python demo_financebench.py --demo basic
```
- Loads sample data
- Demonstrates core pipeline functionality
- Shows sample queries and responses

### 2. Evaluation Demo
```bash
python demo_financebench.py --demo evaluation
```
- Runs quantitative evaluation
- Displays metrics and statistics
- Compares against ground truth answers

### 3. Interactive Demo
```bash
python demo_financebench.py --demo interactive
```
- Interactive question-answering interface
- Real-time pipeline demonstration
- Custom query support

## Performance Considerations

- **Chunking**: Balance between context and precision
- **Embedding**: Uses efficient text-embedding-3-small model
- **Index**: FAISS provides fast similarity search
- **Generation**: Configurable token limits for cost control
- **Caching**: Index persistence for repeated use

## Future Enhancements

The modular architecture supports future improvements:

- **Reranking**: Cross-encoder models for better relevance
- **Hybrid Retrieval**: Combine semantic and keyword search
- **Advanced Chunking**: Semantic-aware text splitting
- **Multi-modal**: Support for tables and figures
- **Optimization**: Hyperparameter tuning and model selection

## API Reference

### RAGPipeline Class

Main pipeline orchestrator with the following key methods:

- `load_and_process_documents(documents)`: Process and index documents
- `query(query, **kwargs)`: Process a single query
- `evaluate_query(query, ground_truth, **kwargs)`: Query with evaluation
- `batch_evaluate(evaluation_data, **kwargs)`: Batch processing
- `save_index(index_path, metadata_path)`: Persist the index
- `load_from_saved_index(index_path, metadata_path)`: Load saved index

### Environment Variables

Set these in your `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
CHUNK_SIZE=500                    # Optional: chunk size
CHUNK_OVERLAP=50                  # Optional: chunk overlap
GENERATION_MODEL=gpt-4o-mini           # Optional: OpenAI model
TEMPERATURE=0.1                  # Optional: generation temperature
```

## Contributing

1. Follow the existing code structure and documentation style
2. Add comments for each function as specified in project requirements
3. Write clean, modular code avoiding unnecessary complexity
4. Test changes with the demo scripts before submitting

## License

This project is for educational and research purposes.

## Support

For issues and questions:
1. Check the demo scripts for usage examples
2. Review the evaluation metrics for performance assessment
3. Ensure proper OpenAI API key configuration
4. Verify all dependencies are installed correctly

