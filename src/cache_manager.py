"""
Smart caching layer for embeddings using SQLite.
"""
import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

from .config import CACHE_DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """Manages embedding cache with SQLite backend."""
    
    def __init__(self, db_path: Path = CACHE_DB_PATH):
        """
        Initialize cache manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create embeddings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    doc_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    hash TEXT NOT NULL,
                    filename TEXT,
                    document_length INTEGER,
                    embedding_dim INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Create index on hash for faster lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_hash 
                ON embeddings(hash)
            ''')
            
            conn.commit()
            logger.info(f"Cache database initialized at {self.db_path}")
    
    def get_cached_embedding(self, doc_id: str, current_hash: str) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding if hash matches.
        
        Args:
            doc_id: Document identifier
            current_hash: Current document hash for validation
            
        Returns:
            Cached embedding array or None if cache miss/invalid
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT embedding, hash 
                FROM embeddings 
                WHERE doc_id = ?
            ''', (doc_id,))
            
            result = cursor.fetchone()
            
            if result is None:
                logger.debug(f"Cache miss: {doc_id} (not found)")
                return None
            
            cached_embedding_blob, cached_hash = result
            
            # Validate hash
            if cached_hash != current_hash:
                logger.info(f"Cache invalidated: {doc_id} (hash mismatch)")
                return None
            
            # Deserialize embedding
            embedding = np.frombuffer(cached_embedding_blob, dtype=np.float32)
            logger.debug(f"Cache hit: {doc_id}")
            return embedding
    
    def save_embedding(
        self,
        doc_id: str,
        embedding: np.ndarray,
        document_hash: str,
        filename: str,
        document_length: int
    ):
        """
        Save or update embedding in cache.
        
        Args:
            doc_id: Document identifier
            embedding: Embedding vector
            document_hash: SHA-256 hash of document
            filename: Original filename
            document_length: Length of document text
        """
        # Convert embedding to bytes
        embedding_blob = embedding.astype(np.float32).tobytes()
        embedding_dim = len(embedding)
        timestamp = datetime.utcnow().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if entry exists
            cursor.execute('SELECT doc_id FROM embeddings WHERE doc_id = ?', (doc_id,))
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update existing entry
                cursor.execute('''
                    UPDATE embeddings 
                    SET embedding = ?, hash = ?, filename = ?, 
                        document_length = ?, embedding_dim = ?, updated_at = ?
                    WHERE doc_id = ?
                ''', (embedding_blob, document_hash, filename, 
                      document_length, embedding_dim, timestamp, doc_id))
                logger.info(f"Updated cache: {doc_id}")
            else:
                # Insert new entry
                cursor.execute('''
                    INSERT INTO embeddings 
                    (doc_id, embedding, hash, filename, document_length, 
                     embedding_dim, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (doc_id, embedding_blob, document_hash, filename,
                      document_length, embedding_dim, timestamp, timestamp))
                logger.info(f"Cached new embedding: {doc_id}")
            
            conn.commit()
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Retrieve all cached embeddings.
        
        Returns:
            Dictionary mapping doc_id to embedding array
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT doc_id, embedding FROM embeddings')
            results = cursor.fetchall()
            
            embeddings = {}
            for doc_id, embedding_blob in results:
                embeddings[doc_id] = np.frombuffer(embedding_blob, dtype=np.float32)
            
            logger.info(f"Retrieved {len(embeddings)} cached embeddings")
            return embeddings
    
    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """
        Retrieve metadata for all cached documents.
        
        Returns:
            List of metadata dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT doc_id, hash, filename, document_length, 
                       embedding_dim, created_at, updated_at
                FROM embeddings
            ''')
            
            results = cursor.fetchall()
            
            metadata_list = []
            for row in results:
                metadata_list.append({
                    'doc_id': row[0],
                    'hash': row[1],
                    'filename': row[2],
                    'length': row[3],
                    'embedding_dim': row[4],
                    'created_at': row[5],
                    'updated_at': row[6]
                })
            
            return metadata_list
    
    def delete_embedding(self, doc_id: str):
        """
        Delete a cached embedding.
        
        Args:
            doc_id: Document identifier
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM embeddings WHERE doc_id = ?', (doc_id,))
            conn.commit()
            logger.info(f"Deleted cache entry: {doc_id}")
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM embeddings')
            conn.commit()
            logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count total entries
            cursor.execute('SELECT COUNT(*) FROM embeddings')
            total_count = cursor.fetchone()[0]
            
            # Get database size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            # Get last update time
            cursor.execute('SELECT MAX(updated_at) FROM embeddings')
            last_update = cursor.fetchone()[0]
            
            return {
                'total_documents': total_count,
                'database_size_bytes': db_size,
                'database_size_mb': round(db_size / (1024 * 1024), 2),
                'last_update': last_update
            }
