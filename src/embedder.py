"""
Embedding generation using sentence-transformers.
"""
import numpy as np
from typing import List, Union
import logging
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, USE_GPU, BATCH_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder:
    """Wrapper for embedding model."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, use_gpu: bool = USE_GPU):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.device = 'cuda' if use_gpu else 'cpu'
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"Model loaded on device: {self.device}")
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = BATCH_SIZE, use_multiprocessing: bool = False) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            use_multiprocessing: Whether to use multiprocessing for encoding
            
        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts with batch_size={batch_size}, multiprocessing={use_multiprocessing}")
        
        # Determine pool size for multiprocessing
        pool_size = None
        if use_multiprocessing:
            import multiprocessing
            pool_size = multiprocessing.cpu_count()
            logger.info(f"Using multiprocessing with {pool_size} workers")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            device=self.device if not use_multiprocessing else None,
        )
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        Same as embed_text but kept separate for clarity.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        return self.embed_text(query)
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        return self.embedding_dim
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'max_seq_length': self.model.max_seq_length
        }


def create_embedder(model_name: str = EMBEDDING_MODEL) -> Embedder:
    """
    Factory function to create an Embedder instance.
    
    Args:
        model_name: Name of the sentence-transformers model
        
    Returns:
        Initialized Embedder instance
    """
    return Embedder(model_name=model_name)
