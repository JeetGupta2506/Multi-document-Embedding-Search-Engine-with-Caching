"""
Vector search engine with FAISS and ranking explanations.
"""
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from .config import (
    FAISS_INDEX_PATH, 
    METADATA_PATH, 
    EMBEDDING_DIM, 
    DEFAULT_TOP_K,
    MAX_PREVIEW_LENGTH,
    USE_MULTIPROCESSING,
    MULTIPROCESS_BATCH_SIZE
)
from .embedder import Embedder
from .cache_manager import CacheManager
from .preprocessor import DocumentPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchEngine:
    """Vector search engine with FAISS backend."""
    
    def __init__(
        self,
        embedder: Embedder,
        cache_manager: CacheManager,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_PATH
    ):
        """
        Initialize search engine.
        
        Args:
            embedder: Embedder instance
            cache_manager: CacheManager instance
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load document metadata
        """
        self.embedder = embedder
        self.cache_manager = cache_manager
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.preprocessor = DocumentPreprocessor()
        
        # FAISS index and metadata
        self.index: Optional[faiss.Index] = None
        self.doc_ids: List[str] = []
        self.documents: Dict[str, str] = {}  # doc_id -> text
        self.metadata: Dict[str, Dict] = {}  # doc_id -> metadata
        
    def build_index(self, documents: Dict[str, str], force_rebuild: bool = False):
        """
        Build FAISS index from documents with smart caching.
        
        Args:
            documents: Dictionary mapping doc_id to document text
            force_rebuild: If True, regenerate all embeddings
        """
        logger.info(f"Building index for {len(documents)} documents")
        
        self.doc_ids = list(documents.keys())
        self.documents = documents
        embeddings_list = []
        
        cache_hits = 0
        cache_misses = 0
        
        # Separate cached and uncached documents
        docs_to_embed = []
        doc_ids_to_embed = []
        doc_hashes = {}
        
        for doc_id, text in documents.items():
            # Compute hash of cleaned text
            document_hash = self.preprocessor.compute_hash(text)
            doc_hashes[doc_id] = document_hash
            
            # Try to get cached embedding
            if not force_rebuild:
                cached_embedding = self.cache_manager.get_cached_embedding(
                    doc_id, document_hash
                )
                if cached_embedding is not None:
                    embeddings_list.append((doc_id, cached_embedding))
                    cache_hits += 1
                    continue
            
            # Mark for batch embedding
            docs_to_embed.append(text)
            doc_ids_to_embed.append(doc_id)
            cache_misses += 1
        
        # Batch embed uncached documents
        if docs_to_embed:
            logger.info(f"Batch embedding {len(docs_to_embed)} documents with multiprocessing={USE_MULTIPROCESSING}")
            new_embeddings = self.embedder.embed_batch(
                docs_to_embed,
                batch_size=MULTIPROCESS_BATCH_SIZE,
                use_multiprocessing=USE_MULTIPROCESSING
            )
            
            # Cache new embeddings
            for idx, (doc_id, embedding) in enumerate(zip(doc_ids_to_embed, new_embeddings)):
                embeddings_list.append((doc_id, embedding))
                
                self.cache_manager.save_embedding(
                    doc_id=doc_id,
                    embedding=embedding,
                    document_hash=doc_hashes[doc_id],
                    filename=doc_id,
                    document_length=len(documents[doc_id])
                )
        
        logger.info(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")
        
        # Sort embeddings by doc_id order to maintain consistency
        embeddings_dict = {doc_id: emb for doc_id, emb in embeddings_list}
        embeddings_sorted = [embeddings_dict[doc_id] for doc_id in self.doc_ids]
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings_sorted, dtype=np.float32)
        
        # Build FAISS index
        dimension = embeddings_array.shape[1]
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings_array)
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        
        # Store metadata
        self._build_metadata()
        
    def _build_metadata(self):
        """Build metadata for all documents."""
        for doc_id, text in self.documents.items():
            self.metadata[doc_id] = {
                'doc_id': doc_id,
                'length': len(text),
                'word_count': len(text.split()),
                'preview': self.preprocessor.create_preview(text, MAX_PREVIEW_LENGTH)
            }
    
    def search(
        self, 
        query: str, 
        top_k: int = DEFAULT_TOP_K,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            return_scores: Whether to include similarity scores
            
        Returns:
            List of search results with explanations
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        query_tokens = set(query.lower().split())
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # No more results
                break
            
            doc_id = self.doc_ids[idx]
            doc_text = self.documents[doc_id]
            doc_tokens = set(doc_text.split())
            
            # Generate explanation
            explanation = self._generate_explanation(
                query_tokens, doc_tokens, doc_text, float(score)
            )
            
            result = {
                'doc_id': doc_id,
                'score': float(score),
                'preview': self.metadata[doc_id]['preview'],
                'explanation': explanation
            }
            
            results.append(result)
        
        return results
    
    def _generate_explanation(
        self,
        query_tokens: set,
        doc_tokens: set,
        doc_text: str,
        similarity_score: float
    ) -> Dict[str, Any]:
        """
        Generate ranking explanation for a search result.
        
        Args:
            query_tokens: Set of query tokens
            doc_tokens: Set of document tokens
            doc_text: Full document text
            similarity_score: Cosine similarity score
            
        Returns:
            Explanation dictionary
        """
        # Find matched keywords
        matched_keywords = list(query_tokens.intersection(doc_tokens))
        matched_keywords.sort()
        
        # Calculate overlap ratio
        if len(query_tokens) > 0:
            overlap_ratio = len(matched_keywords) / len(query_tokens)
        else:
            overlap_ratio = 0.0
        
        # Length normalization (simple penalty for very short/long docs)
        doc_length = len(doc_text.split())
        if doc_length < 50:
            length_penalty = 0.8
        elif doc_length > 1000:
            length_penalty = 0.9
        else:
            length_penalty = 1.0
        
        length_normalized_score = similarity_score * length_penalty
        
        return {
            'matched_keywords': matched_keywords[:10],  # Limit to 10
            'overlap_ratio': round(overlap_ratio, 3),
            'length_normalized_score': round(length_normalized_score, 3),
            'cosine_similarity': round(similarity_score, 3),
            'match_reasoning': self._generate_reasoning(
                matched_keywords, overlap_ratio, similarity_score
            )
        }
    
    def _generate_reasoning(
        self,
        matched_keywords: List[str],
        overlap_ratio: float,
        similarity_score: float
    ) -> str:
        """
        Generate human-readable reasoning for the match.
        
        Args:
            matched_keywords: List of matched keywords
            overlap_ratio: Ratio of query terms found
            similarity_score: Cosine similarity score
            
        Returns:
            Reasoning text
        """
        if similarity_score > 0.7:
            strength = "strong"
        elif similarity_score > 0.5:
            strength = "moderate"
        else:
            strength = "weak"
        
        if len(matched_keywords) > 0:
            keywords_str = ", ".join(matched_keywords[:5])
            reason = f"This document shows {strength} semantic similarity ({similarity_score:.2f}). "
            reason += f"Matched {len(matched_keywords)} query terms: {keywords_str}. "
            reason += f"Query term overlap: {overlap_ratio:.1%}."
        else:
            reason = f"This document shows {strength} semantic similarity ({similarity_score:.2f}) "
            reason += "through contextual relevance despite no exact keyword matches."
        
        return reason
    
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        logger.info(f"FAISS index saved to {self.index_path}")
        
        # Save metadata
        metadata_to_save = {
            'doc_ids': self.doc_ids,
            'documents': self.documents,
            'metadata': self.metadata
        }
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata_to_save, f)
        
        logger.info(f"Metadata saved to {self.metadata_path}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.warning("Index files not found")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"FAISS index loaded from {self.index_path}")
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.doc_ids = saved_data['doc_ids']
            self.documents = saved_data['documents']
            self.metadata = saved_data['metadata']
            
            logger.info(f"Metadata loaded with {len(self.doc_ids)} documents")
            return True
        
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the search index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {'status': 'not_built'}
        
        return {
            'status': 'ready',
            'total_documents': len(self.doc_ids),
            'index_vectors': self.index.ntotal,
            'embedding_dimension': self.index.d,
            'cache_stats': self.cache_manager.get_cache_stats()
        }
