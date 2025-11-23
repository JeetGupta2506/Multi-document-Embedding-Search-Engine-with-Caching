"""
Configuration settings for the embedding search engine.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "data" / "cache"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Cache configuration
CACHE_DB_PATH = CACHE_DIR / "embeddings_cache.db"
FAISS_INDEX_PATH = CACHE_DIR / "faiss_index.bin"
METADATA_PATH = CACHE_DIR / "metadata.pkl"

# Search configuration
DEFAULT_TOP_K = 5
MAX_TOP_K = 50
SIMILARITY_THRESHOLD = 0.0  # Minimum similarity score to return

# Text preprocessing
MAX_PREVIEW_LENGTH = 200  # Characters to show in preview

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Multi-Document Embedding Search Engine"
API_VERSION = "1.0.0"

# Performance
BATCH_SIZE = 32  # For batch embedding generation
USE_GPU = False  # Set to True if CUDA is available
USE_MULTIPROCESSING = True  # Use multiprocessing for batch embeddings
NUM_WORKERS = None  # Number of workers (None = auto-detect CPU count)
MULTIPROCESS_BATCH_SIZE = 64  # Batch size for multiprocessing
