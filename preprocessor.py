"""
Enhanced Preprocessor Module for RAG Pipeline with Contextual Retrieval
Implements Anthropic's Contextual Retrieval approach to improve chunk retrieval accuracy
Reference: https://www.anthropic.com/engineering/contextual-retrieval
"""

import os
import numpy as np
from typing import List, Dict, Any
import faiss
from openai import OpenAI
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from rank_bm25 import BM25Okapi
import re

class DocumentPreprocessor:
    """
    Enhanced document preprocessor with Contextual Retrieval support
    Adds context to chunks before embedding to improve retrieval accuracy
    """
    
    def __init__(self,  
                 chunk_size: int = 500, 
                 chunk_overlap: int = 80,
                 contextualizer_model: str = "gpt-4o-mini",
                 embedding_model: str = "text-embedding-3-small",
                 max_workers: int = 5):
        """
        Initialize the enhanced preprocessor with contextual retrieval support
        
        Args:
            openai_api_key: OpenAI API key for embedding generation and contextualization
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            contextualizer_model: OpenAI model for generating chunk context
            embedding_model: OpenAI model for generating embeddings
            max_workers: Maximum concurrent workers for contextualization
        """
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.contextualizer_model = contextualizer_model
        self.embedding_model = embedding_model
        self.max_workers = max_workers
        self.use_contextual_retrieval = False

        # Storage
        self.chunks = []
        self.contextualized_chunks = []
        self.embeddings = []
        self.bm25_index = None
        self.faiss_index = None
        self.representative_queries = []
        
        # Contextualizer prompt based on Anthropic's research
        self.contextualizer_prompt = """
        <document>
        {document}
        </document>
        Here is the chunk we want to situate within the whole document:
        <chunk>
        {chunk}
        </chunk>
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
        """
                
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks with improved boundary detection
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to find a sentence boundary near the end
            if end < len(text):
                # Look for sentence endings within the last 100 chars
                search_start = max(end - 100, start)
                sentence_endings = ['. ', '! ', '? ', '\n\n']
                
                best_break = end
                for ending in sentence_endings:
                    pos = text.rfind(ending, search_start, end)
                    if pos != -1:
                        best_break = pos + len(ending)
                        break
                
                end = best_break
            
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start = end - self.chunk_overlap
            
        return chunks
    
    def generate_chunk_context(self, chunk: str, document: str) -> str:
        """
        Generate contextual information for a chunk using the full document
        
        Args:
            chunk: The text chunk to contextualize
            document: The full document containing the chunk
            
        Returns:
            Contextual information for the chunk
        """
        try:
            # Use prompt caching if available (reduces cost by up to 90%)
            messages = [
                {
                    "role": "user",
                    "content": self.contextualizer_prompt.format(
                        document=document,
                        chunk=chunk
                    ),
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.contextualizer_model,
                messages=messages,
                temperature=0
            )
            
            context = response.choices[0].message.content.strip()
            return context
            
        except Exception as e:
            print(f"Error generating context for chunk: {e}")
            return ""  # Return empty context on error
    
    def contextualize_chunks_parallel(self, chunks: List[str], document: str) -> List[str]:
        """
        Generate contexts for all chunks in parallel for better performance
        
        Args:
            chunks: List of text chunks
            document: Full document text
            
        Returns:
            List of contextualized chunks
        """
        print(f"Generating contexts for {len(chunks)} chunks...")
        
        # Use ThreadPoolExecutor for I/O-bound OpenAI API calls
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all context generation tasks
            future_to_chunk = {
                executor.submit(self.generate_chunk_context, chunk, document): (idx, chunk)
                for idx, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            results = [None] * len(chunks)
            
            for future in tqdm(as_completed(future_to_chunk), total=len(chunks), desc="Contextualizing chunks"):
                idx, original_chunk = future_to_chunk[future]
                try:
                    context = future.result()
                    
                    # Prepend context to chunk if context was generated
                    if context:
                        contextualized_chunk = f"{context}\n\n{original_chunk}"
                    else:
                        contextualized_chunk = original_chunk
                    
                    results[idx] = contextualized_chunk
                    
                except Exception as e:
                    print(f"Error contextualizing chunk {idx}: {e}")
                    results[idx] = original_chunk  # Fallback to original chunk
        
        return results
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI's embedding model
        Now supports batch processing for better performance
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        batch_size = 128  # OpenAI supports up to 2048 inputs per request
        
        print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}...")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.embedding_model
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size}: {e}")
                # Add zero vectors as placeholders
                embedding_dim = 1536 if "3-small" in self.embedding_model else 3072
                for _ in batch:
                    embeddings.append([0.0] * embedding_dim)
                
        return embeddings

    def tokenize_text(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        text = re.sub(r'[^\w\s]', ' ', text.lower())

        tokens = text.split()

        tokens = [token for token in tokens if len(token) >= 2]
        
        return tokens

    def create_bm25_index(self, chunks: List[str]) -> List[BM25Okapi]:
        """
        Create a BM25 index from chunks
        
        Args:
            chunks: List of chunks to index
        """

        bm25_corpus = []
        for chunk in chunks:
            tokens = self.tokenize_text(chunk)
            bm25_corpus.append(tokens)

        bm25_index = BM25Okapi(bm25_corpus)

        return bm25_index
    
    def create_faiss_index(self, embeddings: List[List[float]]) -> faiss.IndexFlatIP:
        """
        Create a FAISS index from embeddings for efficient similarity search
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            FAISS index for similarity search
        """
        embeddings_array = np.array(embeddings).astype(np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Create FAISS index (Inner Product for cosine similarity with normalized vectors)
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)
        
        return index

    def get_representative_data(
        self,
        evaluation_queries: List[Dict[str, Any]],
        hash_precision: int = 2,
        hash_dims: int = 64,
        max_per_bucket: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Use embedding hashing to pick representative queries for training.
        
        Args:
            evaluation_queries: List of evaluation queries (str or dict with 'query').
            hash_precision: Decimal precision for quantizing embeddings when hashing.
            hash_dims: Number of leading dimensions to use for the hash.
            max_per_bucket: Representatives to keep per hash bucket.
        
        Returns:
            List of representative queries with clustering metadata.
        """
        if not evaluation_queries:
            print("No evaluation queries provided for representative selection.")
            return []

        # Normalize inputs to plain text + optional metadata
        query_texts: List[str] = []
        query_meta: List[Dict[str, Any]] = []
        for item in evaluation_queries:
            query = item.get('query', '')
            ground_truth = item.get('ground_truth', '')

            query_texts.append(query)
            query_meta.append({
                "ground_truth": ground_truth,
                # "metadata": extra_meta
            })

        if not query_texts:
            print("No valid evaluation queries to process.")
            return []

        embeddings = self.generate_embeddings(query_texts)
        embeddings_array = np.array(embeddings, dtype=np.float32)

        if embeddings_array.size == 0:
            print("Failed to compute embeddings for representative selection.")
            return []

        faiss.normalize_L2(embeddings_array)

        dims_to_use = min(hash_dims, embeddings_array.shape[1])
        scaling = 10 ** hash_precision

        def hash_embedding(vec: np.ndarray) -> str:
            quantized = np.round(vec[:dims_to_use] * scaling).astype(int)
            return "|".join(map(str, quantized.tolist()))

        buckets: Dict[str, List[int]] = {}
        for idx, vec in enumerate(embeddings_array):
            bucket_key = hash_embedding(vec)
            buckets.setdefault(bucket_key, []).append(idx)

        representatives: List[Dict[str, Any]] = []
        for bucket_key, indices in buckets.items():
            ranked = sorted(indices, key=lambda i: len(query_texts[i]), reverse=True)
            for idx in ranked[:max_per_bucket]:
                rep = {
                    "query": query_texts[idx],
                    "ground_truth": query_meta[idx].get("ground_truth"),
                    "cluster_size": len(indices),
                    "hash": bucket_key
                }

                if query_meta[idx].get("metadata"):
                    rep["metadata"] = query_meta[idx]["metadata"]
                    
                representatives.append(rep)

        self.representative_queries = representatives

        return representatives
    
    def process_documents(self, documents: List[Dict[str, Any]], use_context: bool = True) -> None:
        """
        Process documents with enhanced contextual retrieval pipeline
        
        Args:
            documents: List of documents with 'text' field
        """
        print(f"Processing {len(documents)} documents with contextual retrieval...")
        self.use_contextual_retrieval = use_context

        # Extract and chunk all documents
        all_chunks = []
        all_contextualized_chunks = []
        chunk_metadata = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get('text', '')
            
            print(f"Processing document {doc_idx + 1}/{len(documents)}...")
            
            # Create chunks
            chunks = self.chunk_text(text)
            contextualized_chunks = chunks
            
            # Generate contextual information if enabled
            if use_context and chunks:
                contextualized_chunks = self.contextualize_chunks_parallel(chunks, text)
            
            # Store both original and contextualized chunks
            for chunk_idx, (chunk, contextualized_chunk) in enumerate(zip(chunks, contextualized_chunks)):
                all_chunks.append(chunk)
                all_contextualized_chunks.append(contextualized_chunk)
                
                chunk_metadata.append({
                    'doc_id': doc_idx,
                    'chunk_id': chunk_idx,
                    'original_doc': doc,
                    'original_chunk': chunk,
                    'contextualized_chunk': contextualized_chunk if use_context else None,
                    'has_context': use_context
                })
        
        # Store chunks and metadata
        self.chunks = all_chunks
        self.contextualized_chunks = all_contextualized_chunks
        self.chunk_metadata = chunk_metadata
        
        # Conditionally generate embeddings based on retrieval methods
        embedding_input = all_contextualized_chunks if use_context else all_chunks
        bm25_input = all_chunks

        print(f"Generating embeddings for {len(embedding_input)} {'contextualized ' if use_context else ''}chunks...")
        self.embeddings = self.generate_embeddings(embedding_input)
        
        print("Creating BM25 index...")
        self.bm25_index = self.create_bm25_index(bm25_input)

        print("Creating FAISS index...")
        self.faiss_index = self.create_faiss_index(self.embeddings)
    
    def save_index(self, index_path: str, metadata_path: str) -> None:
        """
        Save the FAISS index and enhanced metadata to disk
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save chunk metadata
        """
        # Save FAISS index only if it exists
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, index_path)
            print(f"✅ FAISS index saved to {index_path}")
        else:
            print("📊 No FAISS index to save (BM25-only mode)")
            
        # Always save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'contextualized_chunks': self.contextualized_chunks,
                'chunk_metadata': self.chunk_metadata,
                'embeddings': self.embeddings,
                'use_contextual_retrieval': self.use_contextual_retrieval,
                'contextualizer_model': self.contextualizer_model,
                'embedding_model': self.embedding_model
            }, f)
        
        print(f"✅ Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str, use_context: bool = True) -> None:
        """
        Load the FAISS index and enhanced metadata from disk
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
        """
        # Load metadata first to determine what indices should exist
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.contextualized_chunks = data.get('contextualized_chunks', self.chunks)
            self.chunk_metadata = data['chunk_metadata']
            self.embeddings = data.get('embeddings', [])
            self.use_contextual_retrieval = data.get('use_contextual_retrieval', use_context)
            self.contextualizer_model = data.get('contextualizer_model', 'gpt-4o-mini')
            self.embedding_model = data.get('embedding_model', 'text-embedding-3-small')

        if self.chunks:
            # bm25_source = self.contextualized_chunks if (self.use_contextual_retrieval and self.contextualized_chunks) else self.chunks
            # self.bm25_index = self.create_bm25_index(bm25_source)
            self.bm25_index = self.create_bm25_index(self.chunks)
        else:
            self.bm25_index = None

        self.faiss_index = None
        try:
            if os.path.exists(index_path):
                self.faiss_index = faiss.read_index(index_path)
        except Exception as e:
            print(f"Error loading FAISS index from {index_path}: {e}")
            self.faiss_index = None

        if self.faiss_index is None and self.embeddings:
            try:
                self.faiss_index = self.create_faiss_index(self.embeddings)
                print("⚠️ Recreated FAISS index from stored embeddings.")
            except Exception as e:
                print(f"Error recreating FAISS index from embeddings: {e}")
                self.faiss_index = None

        # Print contextual retrieval info
        if hasattr(self, 'chunk_metadata') and self.chunk_metadata:
            contextual_chunks = sum(1 for meta in self.chunk_metadata if meta.get('has_context', False))
            total_chunks = len(self.chunk_metadata)
            if self.use_contextual_retrieval:
                print(f"📊 Contextual Retrieval: {contextual_chunks}/{total_chunks} chunks have context")
