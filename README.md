# RAG Pipeline Optimization with GEPA

A research framework for optimizing Retrieval-Augmented Generation (RAG) pipelines using **Generative Evolutionary Prompt Adjustment (GEPA)**.

## 📝 Abstract

Retrieval-Augmented Generation (RAG) systems rely on complex interactions between multiple components (Query Planner, Retriever, Reranker, Generator). Optimizing these components individually often leads to sub-optimal end-to-end performance. We introduce a **Staged Evolutionary Optimization** approach that iteratively refines prompts for each module, ensuring downstream components adapt to the improved signal distributions of upstream modules. Our framework provides robust evaluation, statistical significance testing, and reproducibility for high-stakes domains like financial document analysis.

## ✨ Key Features

- **Iterative Staged Optimization**: Optimizes components in topological order (Query Planner &rarr; Reranker &rarr; Generator) to maximize holistic performance.
- **Modular RAG Architecture**:
  - **Query Planner**: Decomposes complex queries.
  - **Reranker**: Cross-encoder based filtering and deduplication.
  - **Generator**: Context-aware response generation.
- **Robust Evaluation Engine**:
  - Strict Train/Validation/Test splits to prevent data leakage.
  - Comprehensive metrics: Precision/Recall, BLEU/ROUGE, and custom RAGAS-based scores.
  - Statistical significance testing (Confidence Intervals, p-values).

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API Key (for GPT-4/GPT-3.5)

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   Create a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=your_sk_...
   ```

### Running Experiments

To reproduce the full research experiment with train/val/test splits and staged optimization:

```bash
python run_research_experiment.py \
    --experiment_name "output_001" \
    --n_queries 100 \
    --model "gpt-4-turbo"
```

To run a standalone optimization pass:

```bash
python run_optimization.py \
    --data_path data/train/ \
    --output_dir gepa_runs/
```

## 📂 Project Structure

```text
rag-optimization/
├── modules/                 # Core RAG Component Implementations
│   ├── evaluation/          # Metrics and RAGAS integration
│   ├── generator/           # LLM Response Generation
│   ├── query_planner/       # Query decomposition and strategy
│   ├── reranker/            # Context filtering and ranking
│   ├── base.py              # Base abstractions
│   └── pipeline.py          # End-to-end pipeline orchestrator
├── gepa_adapters/           # GEPA Optimization Interfaces
│   ├── generator_adapter.py
│   ├── query_planner_adapter.py
│   └── reranker_adapter.py
├── run_research_experiment.py # Main entry point for research exp
├── run_optimization.py      # Optimization runner
└── requirements.txt         # Dependencies
```

## 📊 Methodology

Our approach optimizes the RAG pipeline in three distinct stages:

1.  **Query Planner Optimization**: Evolves decomposition strategies to maximize retrieval recall.
2.  **Reranker Optimization**: Tunes filtering logic using the optimized queries from Stage 1, focusing on precision and context window utilization.
3.  **Generator Optimization**: Refines response synthesis prompts conditioned on the high-quality context from Stage 2.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.