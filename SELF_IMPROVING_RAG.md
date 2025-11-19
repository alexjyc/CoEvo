# Self-Improving RAG Pipeline

A sophisticated RAG (Retrieval-Augmented Generation) pipeline that uses **LangGraph** for workflow orchestration and **RAGAS** for evaluation-driven iterative refinement.

## 🌟 Key Features

### 1. **Iterative Self-Improvement**
- Evaluates each module using RAGAS metrics
- Automatically refines retrieval, reranking, and generation based on scores
- Continues until convergence or max iterations reached

### 2. **LangGraph Workflow**
```
START → Initialize → Reformulate Query → Retrieve → Eval Retrieval ↺
                                                      ↓
                                                    Rerank → Eval Rerank ↺
                                                                  ↓
                                                             Generate → Eval Generation ↺ → END
```

### 3. **RAGAS Evaluation Metrics**
- **Faithfulness**: Ensures answers are grounded in retrieved context (no hallucinations)
- **Answer Relevancy**: Measures how well the answer addresses the query
- **Context Precision**: Evaluates relevance of retrieved documents
- **Context Recall**: Checks coverage of necessary information (if ground truth available)

### 4. **Modular Evaluation Cycle**
- Retrieval is immediately evaluated to tune reranking depth
- Reranking quality is scored before answer generation to guide prompt strategy
- Final generation uses full RAGAS scoring to compute the iteration's `overall_score`
- Each module receives the latest metrics so it can adapt without waiting for the next loop
- Only the final generation evaluation is surfaced as the pipeline result; intermediate metrics remain internal control signals

### 5. **Adaptive Refinement**
The pipeline adjusts parameters between iterations based on evaluation scores:

| Low Score In | Adaptive Response |
|--------------|-------------------|
| Context Precision | Increase `retrieval_k` to get more candidates |
| Faithfulness | Decrease `final_k` to use fewer, more relevant chunks |
| Answer Relevancy | Refine query reformulation strategy |

## 📚 Architecture

### State Schema
```python
class RAGState:
    # Input
    query: str
    ground_truth: Optional[str]
    
    # Reformation
    reformulated_query: str
    reformation_rationale: str
    
    # Retrieval
    retrieved_chunks: List[Dict]
    contexts: List[str]
    
    # Reranking
    reranked_chunks: List[Dict]
    final_contexts: List[str]
    
    # Generation
    answer: str
    answer_reference: str
    
    # Evaluation (RAGAS)
    faithfulness_score: float
    answer_relevancy_score: float
    context_precision_score: float
    context_recall_score: float
    retrieval_eval: Dict[str, float]
    reranking_eval: Dict[str, float]
    generation_eval: Dict[str, float]
    overall_score: float
    
    # Iteration control
    iteration: int
    converged: bool
    improvement_history: List[Dict]
    retrieval_round: int
    rerank_round: int
    generation_round: int
```

### Graph Nodes

1. **Initialize** - Set up iteration and stage counters
2. **Reformulate Query** - Enhance query (adapts using retrieval metrics)
3. **Retrieve** - Get relevant documents (adjusts `k` with retrieval feedback)
4. **Eval Retrieval** - Scores contexts and either loops back to reformulation or proceeds
5. **Rerank** - Order documents by relevance (tunes `final_k` using rerank/generation signals)
6. **Eval Rerank** - Scores refined contexts and either loops back to rerank or moves forward
7. **Generate** - Create answer from contexts (conditioned on rerank scores)
8. **Eval Generation** - Runs full RAGAS metrics and either regenerates or finishes

### Convergence Criteria

Each module repeats until it either meets the `convergence_threshold` or exhausts `max_iterations`. Retrieval and reranking loops branch back to their respective optimizers, while generation’s evaluation decides whether to regenerate or finish. The pipeline surfaces the final metric bundle only after the generation evaluator allows it to exit.

## 🚀 Usage

### Basic Example

```python
from self_improving_rag import SelfImprovingRAG

# Initialize pipeline
pipeline = SelfImprovingRAG(
    openai_api_key="your-api-key",
    max_iterations=3,
    convergence_threshold=0.85
)

# Load documents
documents = [
    {"text": "Document content...", "metadata": {"source": "..."}}
]
pipeline.load_and_process_documents(documents)

# Process query with self-improvement
result = pipeline.query(
    query="What are the key financial metrics?",
    ground_truth="Optional ground truth for recall evaluation",
    retrieval_k=20,
    final_k=10
)

# Access results
print(f"Final Answer: {result['answer']}")
print(f"Overall Score: {result['overall_score']:.3f}")
print(f"Iterations: {len(result['improvement_history'])}")
```

### Advanced: Parameter Tuning

```python
# Conservative approach (faster, fewer documents)
result = pipeline.query(
    query="...",
    retrieval_k=15,
    final_k=5,
    max_iterations=2
)

# Aggressive approach (slower, more thorough)
result = pipeline.query(
    query="...",
    retrieval_k=40,
    final_k=15,
    max_iterations=5,
    convergence_threshold=0.90
)
```

### Analyzing Improvement History

```python
result = pipeline.query(query="...", ground_truth="...")

# View improvement over iterations
for i, hist in enumerate(result['improvement_history']):
    print(f"\nIteration {i}:")
    print(f"  Overall Score: {hist['overall_score']:.3f}")
    print(f"  Faithfulness: {hist['faithfulness_score']:.3f}")
    print(f"  Relevancy: {hist['answer_relevancy_score']:.3f}")
    print(f"  Precision: {hist['context_precision_score']:.3f}")
    print(f"  Contexts Used: {hist['num_contexts']}")
    print(f"  Retrieval Eval: {hist['retrieval_eval']}")
    print(f"  Rerank Eval: {hist['reranking_eval']}")
    print(f"  Generation Eval: {hist['generation_eval']}")
```

## 📊 Demo Scripts

### 1. Single Query Demo
```bash
python demo_self_improving.py
```

Demonstrates:
- Detailed iteration-by-iteration improvement
- Score evolution over iterations
- Parameter adaptation in action

### 2. Batch Comparison
Shows comparison between:
- Initial iteration scores
- Final converged scores
- Improvement metrics

### 3. Parameter Tuning
Compares different configurations:
- Conservative (fast, fewer resources)
- Balanced (recommended)
- Aggressive (thorough, more resources)

## 🔧 Configuration Options

### Pipeline Initialization

```python
SelfImprovingRAG(
    openai_api_key: str,                # Required
    chunk_size: int = 600,              # Document chunk size
    chunk_overlap: int = 50,            # Overlap between chunks
    generation_model: str = "gpt-4o-mini",  # Model for generation
    evaluator_model: str = "gpt-4o-mini",   # Model for RAGAS evaluation
    temperature: float = 0,             # Generation temperature
    retrieval_method: str = "hybrid",   # "hybrid", "bm25", or "dense"
    max_iterations: int = 3,            # Maximum refinement iterations
    convergence_threshold: float = 0.85 # Stop when score ≥ threshold
)
```

### Query Options

```python
pipeline.query(
    query: str,                         # Required: user question
    ground_truth: Optional[str] = None, # For context_recall metric
    retrieval_k: int = 20,              # Initial retrieval candidates
    final_k: int = 10,                  # Final chunks for generation
    max_iterations: Optional[int] = None,         # Override default
    convergence_threshold: Optional[float] = None # Override default
)
```

## 📈 Performance Considerations

### Iteration Cost
Each iteration involves:
- 1 query reformulation (LLM call)
- 1 retrieval operation (vector/BM25 search)
- 1 reranking operation
- 1 generation (LLM call)
- 4 evaluation calls (RAGAS metrics via LLM)

**Typical iteration time**: 10-20 seconds with OpenAI API

### Optimization Tips

1. **Use cheaper models for evaluation**
   ```python
   evaluator_model="gpt-4o-mini"  # Faster and cheaper
   ```

2. **Start with fewer iterations**
   ```python
   max_iterations=2  # Often sufficient
   ```

3. **Adjust convergence threshold**
   ```python
   convergence_threshold=0.80  # Lower threshold = fewer iterations
   ```

4. **Batch processing** (coming soon)
   - Process multiple queries in parallel
   - Share document index across queries

## 🎯 When to Use This Pipeline

### ✅ Good Use Cases
- **High-stakes QA**: Medical, legal, financial domains where accuracy matters
- **Complex queries**: Multi-hop reasoning, nuanced questions
- **Quality over speed**: When response quality justifies iteration cost
- **Development/testing**: Understanding what makes good RAG responses

### ❌ Not Recommended For
- **Real-time chat**: Iterations add latency
- **Simple factual lookups**: Single-pass RAG is sufficient
- **Cost-sensitive applications**: Multiple LLM calls per query

## 🔍 Monitoring and Debugging

### LangGraph Visualization
```python
# View the graph structure
from langgraph.graph import StateGraph
print(pipeline.graph.get_graph().draw_ascii())
```

### Integration with Langfuse
The pipeline is compatible with Langfuse tracing (coming soon):
```python
from langfuse import Langfuse

langfuse = Langfuse()
# Trace each iteration separately
```

### Score Interpretation

| Metric | Good (≥0.8) | Fair (0.6-0.8) | Poor (<0.6) |
|--------|-------------|----------------|-------------|
| **Faithfulness** | No hallucinations | Minor inaccuracies | Significant errors |
| **Answer Relevancy** | Directly answers query | Partially relevant | Off-topic |
| **Context Precision** | All retrieved docs relevant | Some irrelevant docs | Mostly irrelevant |
| **Context Recall** | Complete coverage | Partial coverage | Missing key info |

## 🛠️ Extending the Pipeline

### Custom Evaluation Metrics
```python
def custom_metric(state: RAGState) -> float:
    # Your custom logic
    return score

# Add to _evaluate_generation_node or wrap Evaluator.evaluate_metrics
```

### Custom Adaptation Logic
```python
def _route_after_generation_eval(self, state: RAGState) -> str:
    # Custom logic to decide whether to regenerate or stop
    if state['overall_score'] < 0.9 and state['generation_round'] < 5:
        return "generate"
    return END
```

### Different LLM Providers
```python
from langchain_anthropic import ChatAnthropic

# Use Claude for generation
pipeline = SelfImprovingRAG(
    openai_api_key="...",
    generation_model="claude-3-sonnet"  # Requires langchain-anthropic
)
```

## 📖 Further Reading

- **LangGraph Documentation**: https://deepcon.ai/context/cmho6un9p0001l104763q594i
- **RAGAS Documentation**: https://deepcon.ai/context/cmho6uwuz0003l104qualu3s5
- **Iterative Refinement Pattern**: https://deepcon.ai/context/cmho6v28q0001l204w4u4me0i

## 🤝 Contributing

Ideas for contributions:
- [ ] Add more RAGAS metrics (e.g., aspect_critique)
- [ ] Implement parallel batch processing
- [ ] Add visualization of improvement trajectory
- [ ] Support for local LLMs (Ollama/Gemini)
- [ ] Integration with more monitoring platforms

## 📝 License

[Your License Here]

---

**Built with**: LangChain, LangGraph, RAGAS, OpenAI
