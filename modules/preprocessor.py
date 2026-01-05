"""
Enhanced Preprocessor Module for RAG Pipeline with Contextual Retrieval
Implements Anthropic's Contextual Retrieval approach to improve chunk retrieval accuracy
Reference: https://www.anthropic.com/engineering/contextual-retrieval

Optimizations:
- AsyncOpenAI client for non-blocking API calls
- Parallel batch processing for embeddings
- Optimal batch size (512) for throughput
"""

import os
# Fix OpenMP conflict on macOS - must be set before importing faiss/numpy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans
import faiss
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from rank_bm25 import BM25Okapi
import re

def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    Generate a unique chunk-level ID.

    Format: {doc_id}_chunk_{chunk_index:02d}
    Example: doc_manual_v1_chunk_05

    This provides:
    - Traceability: Can identify source document
    - Uniqueness: Combination of doc_id and chunk index
    - Industry-standard: Chunk-level granularity for relevance
    """
    return f"{doc_id}_chunk_{chunk_index:02d}"


class DocumentPreprocessor:
    """
    Enhanced document preprocessor with Contextual Retrieval support
    Adds context to chunks before embedding to improve retrieval accuracy.

    Now uses chunk-level IDs for industry-standard relevance labeling.
    """

    def __init__(self,
                 chunk_size: int = 500,
                 chunk_overlap: int = 80,
                 contextualizer_model: str = "gpt-5-nano",
                 embedding_model: str = "text-embedding-3-small",
                 max_workers: int = 5,
                 embedding_batch_size: int = 512,
                 max_concurrent_batches: int = 5):
        """
        Initialize the enhanced preprocessor with contextual retrieval support

        Args:
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            contextualizer_model: OpenAI model for generating chunk context
            embedding_model: OpenAI model for generating embeddings
            max_workers: Maximum concurrent workers for contextualization
            embedding_batch_size: Batch size for embedding generation (default 512, max 2048)
            max_concurrent_batches: Max concurrent embedding API calls (default 5)
        """
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key, timeout=60.0)
        self.async_client = AsyncOpenAI(api_key=api_key, timeout=60.0, max_retries=2)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.contextualizer_model = contextualizer_model
        self.embedding_model = embedding_model
        self.max_workers = max_workers
        self.embedding_batch_size = min(embedding_batch_size, 2048)  # OpenAI limit
        self.max_concurrent_batches = max_concurrent_batches
        self.use_contextual_retrieval = False

        # Storage
        self.chunks = []
        self.contextualized_chunks = []
        self.embeddings = []
        self.bm25_index = None
        self.faiss_index = None
        self.representative_queries = []

        # Chunk ID index for O(1) lookup
        self.chunk_id_to_index: Dict[str, int] = {}

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
            text = re.sub(r'(\d)', r' \1 ', text)
            text = re.sub(r'\s+', ' ', text).strip()
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
            if chunk:
                chunk = re.sub(r'(\d)', r' \1 ', chunk)
                chunk = re.sub(r'\s+', ' ', chunk).strip()
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
        Generate embeddings using async parallel batch processing.

        This is a sync wrapper that runs the async version.
        For direct async usage, call generate_embeddings_async().
        """
        import nest_asyncio

        try:
            # Try to apply nest_asyncio to allow nested event loops
            nest_asyncio.apply()
        except Exception:
            pass

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already in async context - use nest_asyncio
            try:
                return loop.run_until_complete(self.generate_embeddings_async(texts))
            except RuntimeError:
                # Fallback to synchronous processing
                return self._generate_embeddings_sync(texts)
        else:
            # No running loop - create new one
            return asyncio.run(self.generate_embeddings_async(texts))

    def _generate_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """Fallback synchronous embedding generation with retry logic."""
        embeddings = []
        batch_size = self.embedding_batch_size
        embedding_dim = 1536 if "3-small" in self.embedding_model else 3072

        print(f"Generating embeddings (sync fallback) for {len(texts)} texts...")

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]

            valid_texts = []
            valid_positions = []
            for pos, text in enumerate(batch):
                if text and isinstance(text, str) and text.strip():
                    valid_texts.append(text)
                    valid_positions.append(pos)

            if not valid_texts:
                embeddings.extend([[0.0] * embedding_dim for _ in batch])
                continue

            # Retry logic with exponential backoff
            max_retries = 5
            base_delay = 1.0

            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        input=valid_texts,
                        model=self.embedding_model
                    )

                    response_embeddings = [data.embedding for data in response.data]
                    pos_to_embedding = dict(zip(valid_positions, response_embeddings))

                    for pos in range(len(batch)):
                        if pos in pos_to_embedding:
                            embeddings.append(pos_to_embedding[pos])
                        else:
                            embeddings.append([0.0] * embedding_dim)
                    break  # Success, exit retry loop

                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = '429' in error_str or 'rate limit' in error_str
                    is_connection = 'connection' in error_str or 'timeout' in error_str

                    if (is_rate_limit or is_connection) and attempt < max_retries - 1:
                        import time
                        delay = base_delay * (2 ** attempt)
                        print(f"  Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {str(e)[:50]}...")
                        time.sleep(delay)
                    else:
                        print(f"Error in batch {i//batch_size}: {e}")
                        embeddings.extend([[0.0] * embedding_dim for _ in batch])
                        break

        return embeddings

    async def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using AsyncOpenAI with parallel batch processing.

        Optimizations:
        - Uses AsyncOpenAI for non-blocking API calls
        - Processes multiple batches in parallel (controlled by max_concurrent_batches)
        - Larger batch size (512 default) for better throughput

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (same length as input texts)
        """
        embedding_dim = 1536 if "3-small" in self.embedding_model else 3072
        batch_size = self.embedding_batch_size
        n_texts = len(texts)

        if n_texts == 0:
            return []

        print(f"Generating embeddings for {n_texts} texts (batch_size={batch_size}, concurrency={self.max_concurrent_batches})...")

        # Prepare all batches
        batches = []
        for i in range(0, n_texts, batch_size):
            batch = texts[i:i + batch_size]
            batches.append((i, batch))

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)

        async def process_batch(batch_idx: int, batch: List[str]) -> tuple:
            """Process a single batch with retry logic for rate limits."""
            async with semaphore:
                # Identify valid texts and their positions
                valid_texts = []
                valid_positions = []
                for pos, text in enumerate(batch):
                    if text and isinstance(text, str) and text.strip():
                        valid_texts.append(text)
                        valid_positions.append(pos)

                if not valid_texts:
                    return (batch_idx, [[0.0] * embedding_dim for _ in batch])

                # Retry logic with exponential backoff for rate limits
                max_retries = 5
                base_delay = 1.0

                for attempt in range(max_retries):
                    try:
                        response = await self.async_client.embeddings.create(
                            input=valid_texts,
                            model=self.embedding_model
                        )

                        # Build position -> embedding mapping
                        response_embeddings = [data.embedding for data in response.data]
                        pos_to_embedding = dict(zip(valid_positions, response_embeddings))

                        # Generate embeddings preserving original order
                        batch_embeddings = []
                        for pos in range(len(batch)):
                            if pos in pos_to_embedding:
                                batch_embeddings.append(pos_to_embedding[pos])
                            else:
                                batch_embeddings.append([0.0] * embedding_dim)

                        return (batch_idx, batch_embeddings)

                    except Exception as e:
                        error_str = str(e).lower()
                        is_rate_limit = '429' in error_str or 'rate limit' in error_str
                        is_connection = 'connection' in error_str or 'timeout' in error_str or 'network' in error_str

                        if (is_rate_limit or is_connection) and attempt < max_retries - 1:
                            # Exponential backoff with jitter
                            delay = base_delay * (2 ** attempt) + (0.1 * batch_idx % 1)
                            error_type = "Rate limit" if is_rate_limit else "Connection error"
                            print(f"  {error_type} in batch {batch_idx}, retry {attempt + 1}/{max_retries}...")
                            await asyncio.sleep(delay)
                        else:
                            print(f"Error in batch {batch_idx}: {e}")
                            return (batch_idx, [[0.0] * embedding_dim for _ in batch])

                return (batch_idx, [[0.0] * embedding_dim for _ in batch])

        # Create all tasks
        tasks = [process_batch(i, batch) for i, batch in batches]

        # Run all tasks with progress bar
        results = await tqdm_asyncio.gather(*tasks, desc="Embedding batches")

        # Sort results by batch index and flatten
        results.sort(key=lambda x: x[0])
        embeddings = []
        for _, batch_embeddings in results:
            embeddings.extend(batch_embeddings)

        return embeddings

    async def embed_texts_batch_async(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embed multiple texts in a single API call (for query embedding).

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embedding_dim = 1536 if "3-small" in self.embedding_model else 3072

        # Filter valid texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            return [[0.0] * embedding_dim for _ in texts]

        try:
            response = await self.async_client.embeddings.create(
                input=valid_texts,
                model=self.embedding_model
            )

            # Build index -> embedding mapping
            idx_to_embedding = dict(zip(
                valid_indices,
                [data.embedding for data in response.data]
            ))

            # Return embeddings in original order
            return [
                idx_to_embedding.get(i, [0.0] * embedding_dim)
                for i in range(len(texts))
            ]

        except Exception as e:
            print(f"Error batch embedding: {e}")
            return [[0.0] * embedding_dim for _ in texts]

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
    ) -> List[Dict[str, Any]]:
        query_texts = [q['query'] for q in evaluation_queries]
        embeddings = self.generate_embeddings(query_texts)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        # Cluster
        inertias = []
        
        for k in tqdm(range(5, min(30, len(query_texts)))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_array)
            inertias.append(kmeans.inertia_)

        deltas = np.diff(inertias)
        second_deltas = np.diff(deltas)
        if len(deltas) > 1:
            second_deltas = np.diff(deltas)
            elbow_idx = np.argmax(second_deltas) + 5 + 1
            n_representatives = elbow_idx
        else:
            n_representatives = 5

        n_clusters = min(n_representatives, len(query_texts))
        print(f"\n⏳ Clustering into {n_clusters} groups...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_array)
                
        # Select representative from each cluster
        representatives = []
        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_indices = np.where(cluster_mask)[0]
            cluster_embeddings = embeddings_array[cluster_indices]
            
            # Find closest to centroid
            centroid = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            rep_idx = cluster_indices[np.argmin(distances)]

            # sorted_order = np.argsort(distances)
            # sorted_indices = cluster_indices[sorted_order]
            # sorted_distances = distances[sorted_order]

            # print(f"CLUSTER {i+1}/{n_clusters}")

            # rep_query = query_texts[rep_idx]
            # print(f"\n⭐ REPRESENTATIVE (closest to centroid):")
            # print(f"   {rep_query}")
            # if evaluation_queries[rep_idx].get('ground_truth'):
            #     gt = evaluation_queries[rep_idx]['ground_truth']
            #     gt_preview = (gt[:80] + "...") if len(gt) > 80 else gt
            #     print(f"   GT: {gt_preview}")

            # print(f"\n📋 ALL QUERIES IN THIS CLUSTER (sorted by similarity to centroid):")
            # for rank, (idx, dist) in enumerate(zip(sorted_indices, sorted_distances), 1):
            #     query = query_texts[idx]
            #     marker = "⭐" if idx == rep_idx else "  "
                
            #     # Truncate long queries
            #     if len(query) > 75:
            #         query_display = query[:75] + "..."
            #     else:
            #         query_display = query
                
            #     print(f"   {marker} {rank:2d}. [dist: {dist:.3f}] {query_display}")
                    
            
            representatives.append({
                "query_id": evaluation_queries[rep_idx].get('query_id'),  # ✅ CRITICAL: Preserve query_id
                "query": query_texts[rep_idx],
                "ground_truth": evaluation_queries[rep_idx].get('ground_truth'),
                "reference_doc_ids": evaluation_queries[rep_idx].get('reference_doc_ids'),  # ✅ Preserve references
                "reference_evidence_texts": evaluation_queries[rep_idx].get('reference_evidence_texts'),  # ✅ Preserve evidence
                "cluster_size": len(cluster_indices),
                "cluster_id": i
            })
        
        return representatives
    
    async def process_documents(self, documents: List[Dict[str, Any]], use_context: bool = True) -> None:
        """
        Process documents with enhanced contextual retrieval pipeline.

        Now generates chunk-level IDs for industry-standard relevance labeling.
        Each chunk gets a unique ID: {doc_id}_chunk_{chunk_index:02d}

        Args:
            documents: List of documents with 'text' field
                       Can be either full documents (will be chunked) or pre-chunked
                       (if 'chunk_id' field exists, uses that directly)
        """
        print(f"Processing {len(documents)} documents with contextual retrieval...")
        self.use_contextual_retrieval = use_context

        # Extract and chunk all documents
        all_chunks = []
        all_contextualized_chunks = []
        chunk_metadata = []
        self.chunk_id_to_index = {}  # Reset index

        for doc_idx, doc in enumerate(documents):
            text = doc.get('text', None)
            doc_id = doc.get('doc_id', f"doc_{doc_idx}")

            if text is None:
                continue

            # Check if this is a pre-chunked document (from preprocess_crag.py)
            if 'chunk_id' in doc:
                # Pre-chunked: use existing chunk_id directly
                chunk_id = doc['chunk_id']
                chunk_text = text

                global_chunk_idx = len(all_chunks)
                all_chunks.append(chunk_text)
                all_contextualized_chunks.append(chunk_text)  # Will be contextualized below if needed

                # Store chunk_id -> index mapping for O(1) lookup
                self.chunk_id_to_index[chunk_id] = global_chunk_idx

                chunk_metadata.append({
                    'chunk_id': chunk_id,
                    'doc_id': doc_id,  # For backwards compatibility
                    'chunk_index': doc.get('chunk_index', 0),
                    'parent_doc_id': doc.get('parent_doc_id', doc_id),
                    'original_doc': doc,
                    'original_chunk': chunk_text,
                    'contextualized_chunk': None,
                    'has_context': False
                })
            else:
                # Full document: chunk it and generate chunk IDs
                print(f"Processing document {doc_idx + 1}/{len(documents)}...")

                chunks = self.chunk_text(text)
                contextualized_chunks = chunks

                # Generate contextual information if enabled
                if use_context and chunks:
                    contextualized_chunks = self.contextualize_chunks_parallel(chunks, text)

                # Store both original and contextualized chunks with chunk-level IDs
                for chunk_idx, (chunk, contextualized_chunk) in enumerate(zip(chunks, contextualized_chunks)):
                    # Generate unique chunk ID
                    chunk_id = generate_chunk_id(doc_id, chunk_idx)
                    global_chunk_idx = len(all_chunks)

                    all_chunks.append(chunk)
                    all_contextualized_chunks.append(contextualized_chunk)

                    # Store chunk_id -> index mapping for O(1) lookup
                    self.chunk_id_to_index[chunk_id] = global_chunk_idx

                    chunk_metadata.append({
                        'chunk_id': chunk_id,
                        'doc_id': doc_id,  # Parent doc ID for backwards compatibility
                        'chunk_index': chunk_idx,
                        'parent_doc_id': doc_id,
                        'original_doc': doc,
                        'original_chunk': chunk,
                        'contextualized_chunk': contextualized_chunk if use_context else None,
                        'has_context': use_context
                    })

        # Store chunks and metadata
        self.chunks = all_chunks
        self.contextualized_chunks = all_contextualized_chunks
        self.chunk_metadata = chunk_metadata

        print(f"Created {len(self.chunk_id_to_index)} chunk IDs for O(1) relevance lookup")

        # Conditionally generate embeddings based on retrieval methods
        embedding_input = all_contextualized_chunks if use_context else all_chunks
        bm25_input = all_chunks

        print(f"Generating embeddings for {len(embedding_input)} {'contextualized ' if use_context else ''}chunks...")
        self.embeddings = await self.generate_embeddings_async(embedding_input)

        print("Creating BM25 index...")
        self.bm25_index = self.create_bm25_index(bm25_input)

        print("Creating FAISS index...")
        self.faiss_index = self.create_faiss_index(self.embeddings)

    
    def load_relevance_labels(
        self,
        labels_path,
    ) -> Dict[int, List[int]]:
        """
        Load document-level relevance labels and convert to chunk-level indices.

        The labels file contains document-level IDs (from preprocess_crag.py).
        This method converts them to chunk indices using the chunk metadata.

        Chunk IDs are generated as: {doc_id}_chunk_{chunk_index:02d}
        All chunks from a relevant document are considered relevant.

        Args:
            labels_path: Path to relevance_labels.json (query_id -> [doc_ids])

        Returns:
            Dict mapping query_id to list of relevant chunk indices
        """
        import json

        print(f"Loading relevance labels from {labels_path}...")

        with open(labels_path, 'r') as f:
            doc_level_labels = json.load(f)

        # Convert string keys to int (JSON keys are always strings)
        doc_level_labels = {int(k): v for k, v in doc_level_labels.items()}

        # Build doc_id -> chunk_indices mapping
        doc_id_to_chunks = self._build_doc_id_index()

        # Convert doc_ids to chunk indices
        chunk_level_labels = {}
        queries_with_no_chunks = []

        for query_id, doc_ids in doc_level_labels.items():
            chunk_indices = set()
            for doc_id in doc_ids:
                # Get all chunks from this document
                chunks = doc_id_to_chunks.get(doc_id, [])
                chunk_indices.update(chunks)

            chunk_level_labels[query_id] = list(chunk_indices)

            if not chunk_indices:
                queries_with_no_chunks.append(query_id)

        # Print statistics
        total_relevant = sum(len(v) for v in chunk_level_labels.values())
        avg_relevant = total_relevant / len(chunk_level_labels) if chunk_level_labels else 0

        print(f"Converted document-level to chunk-level relevance:")
        print(f"  - {len(chunk_level_labels)} queries")
        print(f"  - {total_relevant} total relevant chunks")
        print(f"  - {avg_relevant:.1f} avg relevant chunks per query")

        if queries_with_no_chunks:
            print(f"  - {len(queries_with_no_chunks)} queries have no chunks (docs may not be indexed)")

        return chunk_level_labels

    def _build_chunk_id_index(self) -> Dict[str, int]:
        """Build chunk_id -> chunk_index mapping for O(1) lookup."""
        if self.chunk_id_to_index:
            return self.chunk_id_to_index

        chunk_id_to_index: Dict[str, int] = {}
        for chunk_idx, meta in enumerate(self.chunk_metadata):
            chunk_id = meta.get('chunk_id')
            if chunk_id:
                chunk_id_to_index[chunk_id] = chunk_idx
        self.chunk_id_to_index = chunk_id_to_index
        return chunk_id_to_index

    def _build_doc_id_index(self) -> Dict[str, List[int]]:
        """Build parent_doc_id -> chunk_indices mapping (for backwards compatibility)."""
        doc_id_to_chunks: Dict[str, List[int]] = {}
        for chunk_idx, meta in enumerate(self.chunk_metadata):
            parent_doc_id = meta.get('parent_doc_id') or meta.get('doc_id')
            if parent_doc_id:
                if parent_doc_id not in doc_id_to_chunks:
                    doc_id_to_chunks[parent_doc_id] = []
                doc_id_to_chunks[parent_doc_id].append(chunk_idx)
        return doc_id_to_chunks

    def create_relevance_labels(
        self,
        evaluation_queries: List[Dict[str, Any]],
    ) -> Dict[int, List[int]]:
        """
        Create relevance labels from evaluation queries using chunk-level IDs.

        This method supports both:
        1. New format: reference_chunk_ids (chunk-level, preferred)
        2. Legacy format: reference_doc_ids (doc-level, for backwards compatibility)

        Args:
            evaluation_queries: List of query dicts with query_id and
                               reference_chunk_ids or reference_doc_ids

        Returns:
            Dict mapping query_id to list of relevant chunk indices
        """
        print(f"Creating relevance labels for {len(evaluation_queries)} queries...")

        # Ensure chunk_id index is built
        if not self.chunk_id_to_index:
            self._build_chunk_id_index()

        relevance_labels = {}
        queries_with_no_relevant = []
        using_chunk_ids = False
        using_doc_ids = False

        for query in tqdm(evaluation_queries, desc="Creating labels"):
            query_id = query['query_id']
            relevant_chunk_indices = set()

            reference_chunk_ids = query.get('reference_chunk_ids', [])
            if reference_chunk_ids:
                using_chunk_ids = True
                for chunk_id in reference_chunk_ids:
                    if chunk_id in self.chunk_id_to_index:
                        relevant_chunk_indices.add(self.chunk_id_to_index[chunk_id])

            if not relevant_chunk_indices:
                reference_doc_ids = query.get('reference_doc_ids', [])
                if reference_doc_ids:
                    using_doc_ids = True
                    doc_id_to_chunks = self._build_doc_id_index()
                    for doc_id in reference_doc_ids:
                        # Check if doc_id is actually a chunk_id
                        if doc_id in self.chunk_id_to_index:
                            relevant_chunk_indices.add(self.chunk_id_to_index[doc_id])
                        else:
                            # Legacy: get all chunks from parent doc
                            chunk_indices = doc_id_to_chunks.get(doc_id, [])
                            relevant_chunk_indices.update(chunk_indices)

            relevance_labels[query_id] = list(relevant_chunk_indices)

            if not relevant_chunk_indices:
                queries_with_no_relevant.append(query_id)

        # Print statistics
        total_relevant = sum(len(v) for v in relevance_labels.values())
        avg_relevant = total_relevant / len(relevance_labels) if relevance_labels else 0
        queries_with_relevant = sum(1 for v in relevance_labels.values() if v)

        print(f"\nRelevance labels created:")
        format_str = 'chunk-level IDs' if using_chunk_ids else ('doc-level IDs (legacy)' if using_doc_ids else 'no IDs found')
        print(f"  - Format: {format_str}")
        print(f"  - Queries with relevant chunks: {queries_with_relevant}/{len(relevance_labels)}")
        print(f"  - Total relevant chunks: {total_relevant}")
        print(f"  - Average relevant chunks per query: {avg_relevant:.1f}")

        if queries_with_no_relevant:
            print(f"  - {len(queries_with_no_relevant)} queries have NO relevant chunks")
            if len(queries_with_no_relevant) <= 5:
                print(f"    Query IDs: {queries_with_no_relevant}")

        return relevance_labels

    
    def save_index(self, index_path: str, metadata_path: str) -> None:
        """
        Save the FAISS index and enhanced metadata to disk.

        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save chunk metadata
        """
        # Save FAISS index only if it exists
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, index_path)
            print(f"FAISS index saved to {index_path}")
        else:
            print("No FAISS index to save (BM25-only mode)")

        # Always save metadata including chunk_id_to_index for O(1) lookup
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'contextualized_chunks': self.contextualized_chunks,
                'chunk_metadata': self.chunk_metadata,
                'chunk_id_to_index': self.chunk_id_to_index,
                'embeddings': self.embeddings,
                'use_contextual_retrieval': self.use_contextual_retrieval,
                'contextualizer_model': self.contextualizer_model,
                'embedding_model': self.embedding_model
            }, f)

        print(f"Metadata saved to {metadata_path}")
        print(f"  - {len(self.chunk_id_to_index)} chunk IDs indexed")
    
    def load_index(self, index_path: str, metadata_path: str, use_context: bool = True) -> None:
        """
        Load the FAISS index and enhanced metadata from disk.

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

            # Load chunk_id_to_index if available, otherwise rebuild it
            self.chunk_id_to_index = data.get('chunk_id_to_index', {})
            if not self.chunk_id_to_index:
                print("Rebuilding chunk_id_to_index from metadata...")
                self._build_chunk_id_index()

        if self.chunks:
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
                print("Recreated FAISS index from stored embeddings.")
            except Exception as e:
                print(f"Error recreating FAISS index from embeddings: {e}")
                self.faiss_index = None

        # Print loading summary
        if hasattr(self, 'chunk_metadata') and self.chunk_metadata:
            contextual_chunks = sum(1 for meta in self.chunk_metadata if meta.get('has_context', False))
            total_chunks = len(self.chunk_metadata)
            print(f"Loaded {total_chunks} chunks with {len(self.chunk_id_to_index)} chunk IDs indexed")
            if self.use_contextual_retrieval:
                print(f"  Contextual Retrieval: {contextual_chunks}/{total_chunks} chunks have context")
