# System Architecture Diagram

## Complete Memory-Augmented RAG Optimization System

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     UNIFIED RAG PIPELINE (rag_pipeline.py)                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                        PIPELINE MODES                                │    ║
║  ├─────────────────────────────────────────────────────────────────────┤    ║
║  │                                                                       │    ║
║  │  ╔═════════════════════════════════════════════════════════════╗   │    ║
║  │  ║  FAST MODE (Production)                                      ║   │    ║
║  │  ║  ─────────────────────                                       ║   │    ║
║  │  ║  Single-pass optimized query                                 ║   │    ║
║  │  ║  • No memory system                                          ║   │    ║
║  │  ║  • Langfuse tracing                                          ║   │    ║
║  │  ║  • ~4.6s per query                                           ║   │    ║
║  │  ╚═════════════════════════════════════════════════════════════╝   │    ║
║  │                                                                       │    ║
║  │  ╔═════════════════════════════════════════════════════════════╗   │    ║
║  │  ║  ITERATIVE MODE (Experimentation)                            ║   │    ║
║  │  ║  ────────────────────────────────                            ║   │    ║
║  │  ║  Multiple iterations with adaptive parameters                ║   │    ║
║  │  ║  • No memory system                                          ║   │    ║
║  │  ║  • Quality-based convergence                                 ║   │    ║
║  │  ║  • Parameter adjustment (retrieval_k, final_k)               ║   │    ║
║  │  ╚═════════════════════════════════════════════════════════════╝   │    ║
║  │                                                                       │    ║
║  │  ╔═════════════════════════════════════════════════════════════╗   │    ║
║  │  ║  OPTIMIZE MODE (Training) ⭐ NEW!                            ║   │    ║
║  │  ║  ────────────────────────────────                            ║   │    ║
║  │  ║  Full LangGraph with memory-weighted feedback                ║   │    ║
║  │  ║  • Memory system (semantic + judge)                          ║   │    ║
║  │  ║  • Module-specific optimization                              ║   │    ║
║  │  ║  • Weight updates (TD-learning)                              ║   │    ║
║  │  ╚═════════════════════════════════════════════════════════════╝   │    ║
║  │                                                                       │    ║
║  └───────────────────────────────────────────────────────────────────────┘    ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════════════╗
║              OPTIMIZE MODE: LangGraph Flow (7 Nodes)                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║   START                                                                       ║
║     │                                                                         ║
║     ▼                                                                         ║
║   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓         ║
║   ┃  Node 1: memory_retrieve_node                                ┃         ║
║   ┃  ──────────────────────────────                              ┃         ║
║   ┃  • Retrieve relevant memories for all 3 modules              ┃         ║
║   ┃  • Hybrid: Semantic filter → Judge reflection                ┃         ║
║   ┃  • Output: {retrieved_memories, feedback}                    ┃         ║
║   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛         ║
║     │                                                                         ║
║     ▼                                                                         ║
║   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓         ║
║   ┃  Node 2: reform_with_feedback_node                           ┃         ║
║   ┃  ───────────────────────────────────                         ┃         ║
║   ┃  • Module 1: Query Planning & Retrieval                      ┃         ║
║   ┃  • LLMBase.query_planner(query, feedback=...)                ┃         ║
║   ┃  • Output: {generated_queries}                               ┃         ║
║   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛         ║
║     │                                                                         ║
║     ▼                                                                         ║
║   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓         ║
║   ┃  Node 3: retrieve_node                                       ┃         ║
║   ┃  ─────────────────────                                       ┃         ║
║   ┃  • Execute hybrid retrieval (BM25 + Dense)                   ┃         ║
║   ┃  • Parallel if decomposition mode                            ┃         ║
║   ┃  • Output: {retrieved_chunks, contexts}                      ┃         ║
║   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛         ║
║     │                                                                         ║
║     ▼                                                                         ║
║   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓         ║
║   ┃  Node 4: rerank_with_feedback_node                           ┃         ║
║   ┃  ───────────────────────────────────                         ┃         ║
║   ┃  • Module 2: Reranking                                       ┃         ║
║   ┃  • LLMBase.rerank_documents(query, docs, feedback=...)       ┃         ║
║   ┃  • Output: {reranked_docs}                                   ┃         ║
║   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛         ║
║     │                                                                         ║
║     ▼                                                                         ║
║   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓         ║
║   ┃  Node 5: generate_with_feedback_node                         ┃         ║
║   ┃  ─────────────────────────────────────                       ┃         ║
║   ┃  • Module 3: Generation                                      ┃         ║
║   ┃  • LLMBase.generate_answer(query, context, feedback=...)     ┃         ║
║   ┃  • Output: {answer, rationale}                               ┃         ║
║   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛         ║
║     │                                                                         ║
║     ▼                                                                         ║
║   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓         ║
║   ┃  Node 6: evaluate_node                                       ┃         ║
║   ┃  ─────────────────────                                       ┃         ║
║   ┃  • Evaluate all 3 modules separately                         ┃         ║
║   ┃  • Retrieval: context_precision + context_recall             ┃         ║
║   ┃  • Reranking: post-rerank precision + recall                 ┃         ║
║   ┃  • Generation: faithfulness + answer_correctness             ┃         ║
║   ┃  • Output: {all scores, overall_quality}                     ┃         ║
║   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛         ║
║     │                                                                         ║
║     ▼                                                                         ║
║   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓         ║
║   ┃  Node 7: memory_update_node                                  ┃         ║
║   ┃  ────────────────────────────                                ┃         ║
║   ┃  • Create ExecutionMemory with module-specific memories      ┃         ║
║   ┃  • Update weights (TD-learning)                              ┃         ║
║   ┃  • Add to MemoryStore                                        ┃         ║
║   ┃  • Prune low-weight memories if > 100                        ┃         ║
║   ┃  • Output: {iteration++, improvement_history}                ┃         ║
║   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛         ║
║     │                                                                         ║
║     ▼                                                                         ║
║   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓         ║
║   ┃  Routing: _should_continue()                                 ┃         ║
║   ┃  ────────────────────────────                                ┃         ║
║   ┃  Stop if:                                                     ┃         ║
║   ┃    • Reached max_iterations                                  ┃         ║
║   ┃    • Quality >= threshold (converged)                        ┃         ║
║   ┃    • No improvement for 2 iterations                         ┃         ║
║   ┃  Otherwise: CONTINUE → loop back to Node 1                   ┃         ║
║   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛         ║
║     │                                                                         ║
║     ▼                                                                         ║
║   END                                                                         ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════════════╗
║                MEMORY SYSTEM (memory_system.py)                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌──────────────────────────────────────────────────────────────────┐       ║
║  │  ExecutionMemory                                                  │       ║
║  │  ────────────────                                                 │       ║
║  │  • memory_id: str                                                 │       ║
║  │  • query: str                                                     │       ║
║  │  • Module memories:                                               │       ║
║  │    - query_planning_memory: ModuleMemory                          │       ║
║  │    - reranking_memory: ModuleMemory                               │       ║
║  │    - generation_memory: ModuleMemory                              │       ║
║  │  • Scores: retrieval_f1, reranking_f1, generation_quality         │       ║
║  │  • Weight metadata: weight, access_count, success_count           │       ║
║  └──────────────────────────────────────────────────────────────────┘       ║
║                                                                               ║
║  ┌──────────────────────────────────────────────────────────────────┐       ║
║  │  MemoryStore                                                      │       ║
║  │  ───────────                                                      │       ║
║  │  • memories: List[ExecutionMemory]                                │       ║
║  │  • judge: LLMBase (frozen for consistency)                        │       ║
║  │  • _embedding_cache: Dict[str, np.ndarray]                        │       ║
║  │                                                                    │       ║
║  │  Methods:                                                         │       ║
║  │  ──────────                                                       │       ║
║  │  1. retrieve_with_reflection()                                    │       ║
║  │     ├─ Stage 1: Semantic retrieval (0.1s)                         │       ║
║  │     │   └─ Score = similarity × weight × recency                  │       ║
║  │     └─ Stage 2: Judge reflection (1s)                             │       ║
║  │         └─ LLM evaluates top 20 → top 5 + feedback                │       ║
║  │                                                                    │       ║
║  │  2. update_memory_weights()                                       │       ║
║  │     └─ TD-learning inspired weight updates                        │       ║
║  │                                                                    │       ║
║  │  3. add_memory() / _prune_memories()                              │       ║
║  │     └─ Keep top 100 by weight                                     │       ║
║  └──────────────────────────────────────────────────────────────────┘       ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════════════╗
║                          WEIGHT UPDATE SYSTEM                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  new_weight = old_weight + α × [                                             ║
║      + 0.1   if quality_improved           (Success bonus)                   ║
║      + 0.05 × access_count                 (Recurrence bonus, capped)        ║
║      + 0.2  × overall_quality              (Quality bonus)                   ║
║      × 0.99^age_hours                      (Temporal decay)                  ║
║      × 0.8  if success_rate < 0.4          (Diversity penalty)               ║
║  ]                                                                            ║
║                                                                               ║
║  Inspired by:                                                                ║
║  • TD-learning (Sutton & Barto) - Temporal difference updates                ║
║  • R2D2 (DeepMind) - Prioritized experience replay                           ║
║  • RLHF (OpenAI) - Reward modeling for preferences                           ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════════════╗
║                      MODULE-SPECIFIC OPTIMIZATION                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  Module 1: Query Planning & Retrieval                               │    ║
║  │  ───────────────────────────────────────                            │    ║
║  │  Metric: Retrieval F1                                               │    ║
║  │  Feedback Example:                                                   │    ║
║  │    "For financial queries requiring calculations, use decomposition │    ║
║  │     mode to break down into: revenue query + cost query"            │    ║
║  │                                                                       │    ║
║  │  Integration: LLMBase.query_planner(query, feedback=...)            │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  Module 2: Reranking                                                │    ║
║  │  ────────────────────                                               │    ║
║  │  Metric: Reranking F1 (post-rerank precision/recall)                │    ║
║  │  Feedback Example:                                                   │    ║
║  │    "For queries with specific terms like 'Q4' and 'margin',         │    ║
║  │     prioritize exact keyword matches over semantic similarity"      │    ║
║  │                                                                       │    ║
║  │  Integration: LLMBase.rerank_documents(query, docs, feedback=...)   │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  Module 3: Generation                                               │    ║
║  │  ──────────────────                                                 │    ║
║  │  Metric: (Faithfulness + Answer Correctness) / 2                    │    ║
║  │  Feedback Example:                                                   │    ║
║  │    "For numerical answers, be concise and show calculation.         │    ║
║  │     Always cite the specific context used."                         │    ║
║  │                                                                       │    ║
║  │  Integration: LLMBase.generate_answer(query, context, feedback=...) │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════════════╗
║                       CONVERGENCE TRAJECTORY                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  Quality                                                                      ║
║   1.0 ┤                                                    ●                  ║
║   0.9 ┤                                             ●                         ║
║   0.8 ┤                                      ●                                ║
║   0.7 ┤                               ●                                       ║
║   0.6 ┤                        ●                                              ║
║   0.5 ┤                 ●                                                     ║
║   0.4 ┤          ●                                                            ║
║   0.3 ┤   ●                                                                   ║
║   0.0 └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴                   ║
║       0    10   20   30   40   50   60   70   80   90   100                 ║
║                        Number of Queries Processed                            ║
║                                                                               ║
║  Phase 1: Cold Start (0-10 queries)                                          ║
║    • No judge (< 10 memories)                                                ║
║    • Simple semantic feedback                                                ║
║    • Slow improvement                                                        ║
║                                                                               ║
║  Phase 2: Learning (10-50 queries)                                           ║
║    • Judge kicks in                                                          ║
║    • Rapid improvement                                                       ║
║    • Building high-quality memory                                            ║
║                                                                               ║
║  Phase 3: Maturity (50+ queries)                                             ║
║    • Weighted sampling favors proven patterns                                ║
║    • Fine-tuning                                                             ║
║    • Convergence                                                             ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Key Files

```
/Users/ajeon/Work/Columbia/rag-optimization/
├── memory_system.py              ⭐ NEW: Memory management (372 lines)
├── rag_pipeline.py               ✏️ UPDATED: Added OPTIMIZE mode (+300 lines)
├── demo_optimize.py              ⭐ NEW: Complete demo
├── MEMORY_SYSTEM_DESIGN.md       ⭐ NEW: Design document
├── IMPLEMENTATION_SUMMARY.md     ⭐ NEW: This summary
└── ARCHITECTURE_DIAGRAM.md       ⭐ NEW: This file
```

## Quick Start

```bash
# Run demonstration
python demo_optimize.py

# Expected: See memory building over 3 queries, then faster convergence on 4th
```

---

**Status:** ✅ Complete implementation ready for testing!

