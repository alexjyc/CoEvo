"""
Preprocessor Module for RAG Pipeline
Handles document chunking, embedding generation, and FAISS index creation
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import faiss
import openai
from openai import OpenAI
from tqdm import tqdm
import pickle


class DocumentPreprocessor:
    """
    Handles document preprocessing including chunking and embedding generation
    """
    
    def __init__(self, openai_api_key: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the preprocessor with OpenAI client and chunking parameters
        
        Args:
            openai_api_key: OpenAI API key for embedding generation
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.embeddings = []
        self.faiss_index = None
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
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
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start += self.chunk_size - self.chunk_overlap
            
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI's embedding model
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in tqdm(texts, desc="Generating embeddings"):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"Error generating embedding: {e}")
                # Add a zero vector as placeholder
                embeddings.append([0.0] * 1536)  # text-embedding-3-small dimension
                
        return embeddings
    
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
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Process a list of documents: chunk them, generate embeddings, and create FAISS index
        
        Args:
            documents: List of documents with 'text' field
        """
        print("Processing documents...")
        
        # Extract and chunk all documents
        all_chunks = []
        chunk_metadata = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get('text', '')
            chunks = self.chunk_text(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'doc_id': doc_idx,
                    'chunk_id': chunk_idx,
                    'original_doc': doc
                })
        
        self.chunks = all_chunks
        self.chunk_metadata = chunk_metadata
        
        # Generate embeddings
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        self.embeddings = self.generate_embeddings(all_chunks)
        
        # Create FAISS index
        print("Creating FAISS index...")
        self.faiss_index = self.create_faiss_index(self.embeddings)
        
        print(f"Preprocessing complete. Processed {len(documents)} documents into {len(all_chunks)} chunks.")
    
    def save_index(self, index_path: str, metadata_path: str) -> None:
        """
        Save the FAISS index and metadata to disk
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save chunk metadata
        """
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, index_path)
            
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'chunk_metadata': self.chunk_metadata,
                    'embeddings': self.embeddings
                }, f)
            
            print(f"Index saved to {index_path}")
            print(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str) -> None:
        """
        Load the FAISS index and metadata from disk
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
        """
        self.faiss_index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_metadata = data['chunk_metadata']
            self.embeddings = data['embeddings']
        
        print(f"Index loaded from {index_path}")
        print(f"Metadata loaded from {metadata_path}")
