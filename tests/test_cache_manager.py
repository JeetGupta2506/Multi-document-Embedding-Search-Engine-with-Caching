"""
Unit tests for cache manager.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.cache_manager import CacheManager


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def cache_manager(temp_cache_dir):
    """Create cache manager with temporary database."""
    db_path = temp_cache_dir / "test_cache.db"
    return CacheManager(db_path)


def test_cache_initialization(cache_manager):
    """Test cache database is initialized."""
    assert cache_manager.db_path.exists()


def test_save_and_retrieve_embedding(cache_manager):
    """Test saving and retrieving embeddings."""
    doc_id = "doc_001"
    embedding = np.random.rand(384).astype(np.float32)
    doc_hash = "test_hash_123"
    
    # Save embedding
    cache_manager.save_embedding(
        doc_id=doc_id,
        embedding=embedding,
        document_hash=doc_hash,
        filename="test.txt",
        document_length=100
    )
    
    # Retrieve embedding
    cached = cache_manager.get_cached_embedding(doc_id, doc_hash)
    
    assert cached is not None
    np.testing.assert_array_almost_equal(embedding, cached)


def test_cache_miss_nonexistent(cache_manager):
    """Test cache miss for non-existent document."""
    cached = cache_manager.get_cached_embedding("nonexistent", "hash")
    assert cached is None


def test_cache_invalidation(cache_manager):
    """Test cache is invalidated when hash changes."""
    doc_id = "doc_001"
    embedding = np.random.rand(384).astype(np.float32)
    old_hash = "old_hash"
    new_hash = "new_hash"
    
    # Save with old hash
    cache_manager.save_embedding(
        doc_id=doc_id,
        embedding=embedding,
        document_hash=old_hash,
        filename="test.txt",
        document_length=100
    )
    
    # Try to retrieve with new hash (should fail)
    cached = cache_manager.get_cached_embedding(doc_id, new_hash)
    assert cached is None


def test_update_existing_embedding(cache_manager):
    """Test updating an existing cache entry."""
    doc_id = "doc_001"
    embedding1 = np.random.rand(384).astype(np.float32)
    embedding2 = np.random.rand(384).astype(np.float32)
    hash1 = "hash1"
    hash2 = "hash2"
    
    # Save first version
    cache_manager.save_embedding(
        doc_id=doc_id,
        embedding=embedding1,
        document_hash=hash1,
        filename="test.txt",
        document_length=100
    )
    
    # Update with new version
    cache_manager.save_embedding(
        doc_id=doc_id,
        embedding=embedding2,
        document_hash=hash2,
        filename="test.txt",
        document_length=150
    )
    
    # Retrieve with new hash
    cached = cache_manager.get_cached_embedding(doc_id, hash2)
    np.testing.assert_array_almost_equal(embedding2, cached)


def test_get_all_embeddings(cache_manager):
    """Test retrieving all embeddings."""
    # Save multiple embeddings
    for i in range(5):
        embedding = np.random.rand(384).astype(np.float32)
        cache_manager.save_embedding(
            doc_id=f"doc_{i:03d}",
            embedding=embedding,
            document_hash=f"hash_{i}",
            filename=f"doc_{i}.txt",
            document_length=100 + i
        )
    
    all_embeddings = cache_manager.get_all_embeddings()
    assert len(all_embeddings) == 5
    assert all(isinstance(v, np.ndarray) for v in all_embeddings.values())


def test_get_cache_stats(cache_manager):
    """Test cache statistics."""
    # Save some embeddings
    for i in range(3):
        embedding = np.random.rand(384).astype(np.float32)
        cache_manager.save_embedding(
            doc_id=f"doc_{i:03d}",
            embedding=embedding,
            document_hash=f"hash_{i}",
            filename=f"doc_{i}.txt",
            document_length=100
        )
    
    stats = cache_manager.get_cache_stats()
    assert stats['total_documents'] == 3
    assert stats['database_size_bytes'] > 0
    assert 'last_update' in stats


def test_clear_cache(cache_manager):
    """Test clearing cache."""
    # Save some embeddings
    embedding = np.random.rand(384).astype(np.float32)
    cache_manager.save_embedding(
        doc_id="doc_001",
        embedding=embedding,
        document_hash="hash",
        filename="test.txt",
        document_length=100
    )
    
    # Clear cache
    cache_manager.clear_cache()
    
    # Verify cache is empty
    stats = cache_manager.get_cache_stats()
    assert stats['total_documents'] == 0
