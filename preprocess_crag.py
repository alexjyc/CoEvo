"""
CRAG Data Preprocessing Script
Decompresses CRAG bz2 files and creates train/valid/test splits for RAG pipeline.
"""

import bz2
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import random
from collections import defaultdict
from bs4 import BeautifulSoup
import re

# Configuration
CRAG_DATA_DIR = "crag_data"
OUTPUT_DIR = "data"
TRAIN_RATIO = 0.8  # For splitting validation data into train/valid
RANDOM_SEED = 42

# Files to process
CRAG_FILES = [
    "CRAG Task 1 and 2 Dev v4.jsonl.bz2",
    # "CRAG Task 1 and 2 Dev v5.jsonl.bz2",
]


def extract_text_from_html(html_content: str, max_length: int = 2000) -> str:
    """Extract clean text from HTML content."""
    if not html_content:
        return ""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate if too long
        # if len(text) > max_length:
        #     text = text[:max_length] + "..."
        
        return text
    except Exception as e:
        return ""


def extract_context_from_search_results(search_results: List[Dict]) -> List[Dict[str, str]]:
    """
    Extract context passages from search results for RAG.
    Returns a list of context documents with metadata.
    """
    contexts = []
    
    if not search_results:
        return contexts
    
    for idx, result in enumerate(search_results):
        page_name = result.get("page_name", "")
        page_url = result.get("page_url", "")
        page_snippet = result.get("page_snippet", "")
        page_result = result.get("page_result", "")  # Full HTML
        page_last_modified = result.get("page_last_modified", "")
        
        # Extract text from HTML
        page_text = extract_text_from_html(page_result)
        
        # Create context document
        context = {
            "doc_id": idx,
            "title": page_name,
            "url": page_url,
            "snippet": page_snippet,
            "content": page_text if page_text else page_snippet,
            "last_modified": page_last_modified,
        }
        contexts.append(context)
    
    return contexts


def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single example from CRAG dataset."""
    
    # Extract search result contexts
    search_results = example.get("search_results", [])
    contexts = extract_context_from_search_results(search_results)
    
    # Build the processed example
    processed = {
        "interaction_id": example.get("interaction_id", ""),
        "query": example.get("query", ""),
        "answer": example.get("answer", ""),
        "alt_ans": example.get("alt_ans", []),
        "domain": example.get("domain", ""),
        "question_type": example.get("question_type", ""),
        "static_or_dynamic": example.get("static_or_dynamic", ""),
        "query_time": example.get("query_time", ""),
        "popularity": example.get("popularity", ""),
        "split": example.get("split", 0),
        # RAG-specific fields
        "contexts": contexts,
        "num_contexts": len(contexts),
    }
    
    return processed


def load_crag_file(filepath: str) -> List[Dict[str, Any]]:
    """Load and decompress a CRAG bz2 file."""
    examples = []
    print(f"Loading {filepath}...")
    
    with bz2.open(filepath, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if line.strip():
                try:
                    example = json.loads(line)
                    processed = process_example(example)
                    examples.append(processed)
                    
                    if (line_num + 1) % 500 == 0:
                        print(f"  Processed {line_num + 1} examples...")
                except json.JSONDecodeError as e:
                    print(f"  Warning: Failed to parse line {line_num + 1}: {e}")
                    continue
    
    print(f"  Loaded {len(examples)} examples from {filepath}")
    return examples


def split_data(
    examples: List[Dict[str, Any]], 
    train_ratio: float = 0.8,
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split data into train/valid/test sets.
    
    Uses the 'split' field from CRAG:
    - split=0: validation data -> further split into train/valid
    - split=1: test data
    """
    random.seed(seed)
    
    # Separate by original split
    validation_data = [ex for ex in examples if ex.get("split", 0) == 0]
    test_data = [ex for ex in examples if ex.get("split", 0) == 1]
    
    print(f"\nOriginal split distribution:")
    print(f"  Validation (split=0): {len(validation_data)}")
    print(f"  Test (split=1): {len(test_data)}")
    
    # Shuffle validation data
    random.shuffle(validation_data)
    
    # Split validation into train/valid
    split_idx = int(len(validation_data) * train_ratio)
    train_data = validation_data[:split_idx]
    valid_data = validation_data[split_idx:]
    
    return {
        "train": train_data,
        "valid": valid_data,
        "test": test_data,
    }


def print_statistics(splits: Dict[str, List[Dict[str, Any]]]):
    """Print statistics about the dataset splits."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    for split_name, data in splits.items():
        print(f"\n{split_name.upper()} SET: {len(data)} examples")
        
        if not data:
            continue
            
        # Domain distribution
        domains = defaultdict(int)
        question_types = defaultdict(int)
        dynamics = defaultdict(int)
        avg_contexts = []
        
        for ex in data:
            domains[ex.get("domain", "unknown")] += 1
            question_types[ex.get("question_type", "unknown")] += 1
            dynamics[ex.get("static_or_dynamic", "unknown")] += 1
            avg_contexts.append(ex.get("num_contexts", 0))
        
        print(f"  Domains: {dict(domains)}")
        print(f"  Question Types: {dict(question_types)}")
        print(f"  Static/Dynamic: {dict(dynamics)}")
        print(f"  Avg contexts per query: {sum(avg_contexts)/len(avg_contexts):.2f}")


def save_splits(splits: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """Save the data splits to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, data in splits.items():
        # Create split directory
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Save full data
        output_path = os.path.join(split_dir, f"{split_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {split_name} set ({len(data)} examples) to {output_path}")
        
        # Also save a JSONL version for streaming
        jsonl_path = os.path.join(split_dir, f"{split_name}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"Saved {split_name} JSONL to {jsonl_path}")


def create_rag_format(splits: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """
    Create RAG-specific format with query-context pairs.
    Useful for training retrievers or evaluating RAG pipelines.
    """
    rag_dir = os.path.join(output_dir, "rag_format")
    os.makedirs(rag_dir, exist_ok=True)
    
    for split_name, data in splits.items():
        rag_examples = []
        
        for ex in data:
            # Combine all context content into a single string for simple RAG
            context_texts = []
            for ctx in ex.get("contexts", []):
                ctx_text = f"[{ctx['title']}]\n{ctx['content']}"
                context_texts.append(ctx_text)
            
            rag_example = {
                "id": ex["interaction_id"],
                "question": ex["query"],
                "answer": ex["answer"],
                "alt_answers": ex.get("alt_ans", []),
                "context": "\n\n---\n\n".join(context_texts),
                "individual_contexts": ex.get("contexts", []),
                "metadata": {
                    "domain": ex.get("domain", ""),
                    "question_type": ex.get("question_type", ""),
                    "static_or_dynamic": ex.get("static_or_dynamic", ""),
                    "query_time": ex.get("query_time", ""),
                    "popularity": ex.get("popularity", ""),
                }
            }
            rag_examples.append(rag_example)
        
        # Save RAG format
        output_path = os.path.join(rag_dir, f"{split_name}_rag.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rag_examples, f, indent=2, ensure_ascii=False)
        print(f"Saved RAG format {split_name} ({len(rag_examples)} examples) to {output_path}")


def create_optimization_format(splits: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """
    Create format compatible with run_optimization.py.

    This outputs DOCUMENT-LEVEL data. Chunking and chunk-level relevance labels
    are created by DocumentPreprocessor during the optimization pipeline.

    Outputs per split:
    - documents.json: List of {doc_id, text, title, url} - full documents
    - queries.json: List of {query_id, query, ground_truth, reference_doc_ids, ...}
    - relevance_labels.json: Dict of {query_id: [relevant_doc_ids]} - document-level ground truth

    The optimization pipeline (run_optimization.py) will:
    1. Load these documents
    2. Chunk them using DocumentPreprocessor (generates chunk IDs)
    3. Create chunk-level relevance labels from document-level labels
    """
    for split_name, data in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Collect all unique documents and build queries
        documents = []
        queries = []
        relevance_labels = {}  # query_id -> [doc_ids]
        seen_doc_ids = set()

        for query_idx, ex in enumerate(data):
            interaction_id = ex.get("interaction_id", f"q_{query_idx}")

            # Process contexts as documents
            reference_doc_ids = []
            reference_evidence_texts = []

            for ctx in ex.get("contexts", []):
                # Create unique doc_id: interaction_id + context doc_id
                doc_id = f"{interaction_id}_doc_{ctx['doc_id']}"

                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    documents.append({
                        "doc_id": doc_id,
                        "text": ctx.get("content", ""),
                        "title": ctx.get("title", ""),
                        "url": ctx.get("url", ""),
                        "query_id": query_idx,  # Track which query this doc belongs to
                    })

                # All contexts for this query are considered relevant
                reference_doc_ids.append(doc_id)

                # Use snippet or first 500 chars as evidence text
                evidence = ctx.get("snippet") or ctx.get("content", "")[:500]
                if evidence:
                    reference_evidence_texts.append({
                        "doc_id": doc_id,
                        "evidence_text": evidence
                    })

            # Build query entry
            query_entry = {
                "query_id": query_idx,
                "query": ex.get("query", ""),
                "ground_truth": ex.get("answer", ""),
                "alt_answers": ex.get("alt_ans", []),
                "reference_doc_ids": reference_doc_ids,
                "reference_evidence_texts": reference_evidence_texts,
                "metadata": {
                    "interaction_id": interaction_id,
                    "domain": ex.get("domain", ""),
                    "question_type": ex.get("question_type", ""),
                    "static_or_dynamic": ex.get("static_or_dynamic", ""),
                }
            }
            queries.append(query_entry)

            # Ground truth relevance labels (doc_id level)
            relevance_labels[query_idx] = reference_doc_ids

        # Save documents.json
        docs_path = os.path.join(split_dir, "documents.json")
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(documents)} documents to {docs_path}")

        # Save queries.json
        queries_path = os.path.join(split_dir, "queries.json")
        with open(queries_path, "w", encoding="utf-8") as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(queries)} queries to {queries_path}")

        # Save relevance_labels.json (document-level ground truth)
        # Note: Chunk-level labels are created by DocumentPreprocessor
        labels_path = os.path.join(split_dir, "relevance_labels.json")
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(relevance_labels, f, indent=2, ensure_ascii=False)
        print(f"Saved document-level relevance labels for {len(relevance_labels)} queries to {labels_path}")

        # Print relevance stats
        total_relevant = sum(len(v) for v in relevance_labels.values())
        avg_relevant = total_relevant / len(relevance_labels) if relevance_labels else 0
        print(f"  Avg relevant docs per query: {avg_relevant:.1f}")


def main():
    print("="*60)
    print("CRAG Data Preprocessing for RAG Pipeline")
    print("="*60)
    
    # Collect all examples from both files
    all_examples = []
    seen_ids = set()
    
    for filename in CRAG_FILES:
        filepath = os.path.join(CRAG_DATA_DIR, filename)
        if os.path.exists(filepath):
            examples = load_crag_file(filepath)
            
            # Deduplicate based on interaction_id
            for ex in examples:
                ex_id = ex.get("interaction_id", "")
                if ex_id not in seen_ids:
                    seen_ids.add(ex_id)
                    all_examples.append(ex)
                else:
                    print(f"  Skipping duplicate: {ex_id}")
        else:
            print(f"Warning: File not found: {filepath}")
    
    print(f"\nTotal unique examples: {len(all_examples)}")
    
    if not all_examples:
        print("No examples found. Exiting.")
        return
    
    # Split the data
    splits = split_data(all_examples, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED)
    
    # Print statistics
    print_statistics(splits)
    
    # Save splits
    print("\n" + "="*60)
    print("SAVING DATA")
    print("="*60)
    save_splits(splits, OUTPUT_DIR)
    
    # Create RAG-specific format
    print("\n" + "="*60)
    print("CREATING RAG FORMAT")
    print("="*60)
    # create_rag_format(splits, OUTPUT_DIR)
    
    # Create optimization format (for run_optimization.py)
    print("\n" + "="*60)
    print("CREATING OPTIMIZATION FORMAT")
    print("="*60)
    create_optimization_format(splits, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("  - train/train.json, train.jsonl, documents.json, queries.json")
    print("  - valid/valid.json, valid.jsonl, documents.json, queries.json")
    print("  - test/test.json, test.jsonl, documents.json, queries.json")
    print("  - rag_format/[split]_rag.json")


if __name__ == "__main__":
    main()

