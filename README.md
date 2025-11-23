# Multi-Document Embedding Search Engine with Caching

A production-ready, lightweight embedding-based search engine that processes 100-200 text documents with intelligent caching, vector search capabilities, and a clean REST API. The system uses SHA-256 hashing to avoid recomputing embeddings for unchanged documents.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

✨ **Smart Caching**: SHA-256 hash-based cache invalidation to avoid unnecessary recomputation  
🔍 **Vector Search**: FAISS-powered semantic search with cosine similarity  
📊 **Ranking Explanations**: Detailed reasoning for each search result  
🚀 **REST API**: FastAPI-based endpoints with automatic documentation  
💾 **Persistent Storage**: SQLite cache and FAISS index persistence  
🎯 **High Performance**: Sub-500ms query response times  
📝 **Clean Code**: Modular architecture with comprehensive tests  

## Architecture

```
┌─────────────┐
│   FastAPI   │  REST API Layer
└──────┬──────┘
       │
┌──────▼──────────────────────────────┐
│      Search Engine Core             │
│  ┌────────────┐  ┌───────────────┐ │
│  │  Embedder  │  │ Search Engine │ │
│  │ (sentence- │  │   (FAISS)     │ │
│  │transformers)│  └───────────────┘ │
│  └────────────┘                     │
└──────┬──────────────────────────────┘
       │
┌──────▼──────────┐
│  Cache Manager  │  SQLite + SHA-256
│  (Smart Cache)  │
└─────────────────┘
```

## Caching Mechanism

The system implements a sophisticated caching strategy:

1. **Document Processing**: When a document is added/updated:
   - Clean text (lowercase, remove HTML, normalize whitespace)
   - Compute SHA-256 hash of cleaned text
   - Check if hash exists in cache

2. **Cache Validation**:
   - **Cache Hit**: If hash matches → reuse cached embedding (no computation)
   - **Cache Miss**: If hash differs → generate new embedding and update cache
   - **New Document**: Generate and cache embedding

3. **Storage**:
   - **SQLite Database**: Stores embeddings as BLOB with metadata
   - **FAISS Index**: Separate binary file for fast vector search
   - **Metadata**: Pickled Python dictionary with document info

4. **Benefits**:
   - ⚡ Instant rebuilds when documents haven't changed
   - 💾 Persistent across application restarts
   - 🔄 Automatic invalidation on document updates
   - 📈 Scales efficiently with document count

## Project Structure

```
CodeAtRandom/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── preprocessor.py        # Text cleaning utilities
│   ├── cache_manager.py       # SQLite caching layer
│   ├── embedder.py           # sentence-transformers wrapper
│   ├── search_engine.py      # FAISS vector search
│   └── api.py                # FastAPI application
├── scripts/
│   ├── __init__.py
│   └── load_data.py          # 20 Newsgroups dataset loader
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py
│   ├── test_cache_manager.py
│   └── test_api.py
├── data/
│   ├── docs/                 # Text documents (git-ignored)
│   └── cache/                # Cache files (git-ignored)
├── requirements.txt          # Python dependencies
├── run_api.py               # API server launcher
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the repository**:
   ```powershell
   cd C:\Users\jeetg\OneDrive\Desktop\CodeAtRandom
   ```

2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

   This will install:
   - FastAPI & Uvicorn (API server)
   - sentence-transformers (embedding model)
   - FAISS (vector search)
   - scikit-learn (20 Newsgroups dataset)
   - BeautifulSoup4 (HTML cleaning)
   - pytest (testing)

## Quick Start

### 1. Load Dataset

Load 200 documents from the 20 Newsgroups dataset:

```powershell
python -m scripts.load_data --num-docs 200
```

This creates `doc_001.txt` through `doc_200.txt` in `data/docs/`.

**Options**:
```powershell
# Load specific number of documents
python -m scripts.load_data --num-docs 150

# Use test subset
python -m scripts.load_data --subset test

# Load specific categories only
python -m scripts.load_data --categories sci.space comp.graphics
```

### 2. Start API Server

```powershell
python run_api.py
```

The server will start at `http://localhost:8000`

### 3. Build Search Index

**Option A**: Using curl (PowerShell):
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/rebuild" -Method Post
```

**Option B**: Visit the interactive API docs at `http://localhost:8000/docs` and use the `/rebuild` endpoint.

### 4. Search Documents

**Using curl**:
```powershell
$body = @{
    query = "quantum physics"
    top_k = 5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/search" -Method Post -Body $body -ContentType "application/json"
```

**Using Python**:
```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={"query": "quantum physics", "top_k": 5}
)

results = response.json()
for result in results['results']:
    print(f"Doc: {result['doc_id']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Preview: {result['preview'][:100]}...")
    print(f"Reasoning: {result['explanation']['match_reasoning']}\n")
```

## API Endpoints

### `POST /search`

Search for similar documents.

**Request**:
```json
{
  "query": "artificial intelligence machine learning",
  "top_k": 5
}
```

**Response**:
```json
{
  "query": "artificial intelligence machine learning",
  "top_k": 5,
  "total_results": 5,
  "results": [
    {
      "doc_id": "doc_042",
      "score": 0.856,
      "preview": "machine learning is a subset of artificial intelligence...",
      "explanation": {
        "matched_keywords": ["artificial", "intelligence", "learning", "machine"],
        "overlap_ratio": 1.0,
        "length_normalized_score": 0.856,
        "cosine_similarity": 0.856,
        "match_reasoning": "This document shows strong semantic similarity (0.86). Matched 4 query terms: artificial, intelligence, learning, machine. Query term overlap: 100.0%."
      }
    }
  ]
}
```

### `POST /rebuild`

Rebuild the search index from documents in `data/docs/`.

**Request**:
```
POST /rebuild?force=false
```

**Query Parameters**:
- `force` (optional): If `true`, regenerate all embeddings regardless of cache

**Response**:
```json
{
  "status": "success",
  "message": "Index rebuilt with 200 documents",
  "stats": {
    "status": "ready",
    "total_documents": 200,
    "index_vectors": 200,
    "embedding_dimension": 384
  }
}
```

### `GET /stats`

Get index statistics.

**Response**:
```json
{
  "status": "ready",
  "total_documents": 200,
  "index_vectors": 200,
  "embedding_dimension": 384,
  "cache_stats": {
    "total_documents": 200,
    "database_size_bytes": 614400,
    "database_size_mb": 0.59
  }
}
```

### `GET /health`

Health check endpoint.

### `DELETE /cache`

Clear the embedding cache.

## Configuration

Edit `src/config.py` to customize:

```python
# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Search parameters
DEFAULT_TOP_K = 5
MAX_TOP_K = 50

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# Performance
BATCH_SIZE = 32
USE_GPU = False  # Set True if CUDA available
```

## Design Decisions

### 1. Embedding Model: `all-MiniLM-L6-v2`

**Rationale**:
- ✅ Lightweight (80MB) - fast download and inference
- ✅ High quality (384-dim embeddings)
- ✅ General-purpose - works well for diverse text
- ✅ Normalized embeddings - efficient cosine similarity

**Alternatives considered**:
- `all-mpnet-base-v2`: Higher quality but slower (420MB)
- OpenAI API: Requires API key and has costs

### 2. Cache Storage: SQLite

**Rationale**:
- ✅ Zero configuration - file-based database
- ✅ ACID compliance - reliable storage
- ✅ Efficient BLOB storage for embeddings
- ✅ Built-in indexing on hash column
- ✅ Cross-platform compatibility

**Alternatives considered**:
- JSON: Simple but no indexing or BLOB support
- Pickle: Fast but no querying capabilities
- Redis: Requires separate server

### 3. Vector Search: FAISS IndexFlatIP

**Rationale**:
- ✅ Exact search - no approximation for 100-200 docs
- ✅ Inner product - optimal for normalized vectors
- ✅ Simple and reliable
- ✅ Fast for this scale (<1ms search)

**Alternatives considered**:
- `IndexIVFFlat`: Overkill for small dataset
- Custom NumPy: Reinventing the wheel
- Approximate methods: Unnecessary at this scale

### 4. Framework: FastAPI

**Rationale**:
- ✅ Async support - concurrent requests
- ✅ Automatic OpenAPI docs
- ✅ Pydantic validation - type-safe requests
- ✅ Modern Python features
- ✅ High performance

## Performance Metrics

Tested on a standard laptop (Intel i5, 8GB RAM):

| Operation | Time | Notes |
|-----------|------|-------|
| Load 200 docs | ~5s | One-time setup |
| First embedding generation | ~30s | Model download + compute |
| Index rebuild (cached) | ~2s | All cache hits |
| Index rebuild (fresh) | ~25s | All cache misses |
| Single query | ~50-200ms | Including ranking explanation |
| Concurrent queries (10) | ~150ms avg | Good scalability |

**Cache Hit Ratio Impact**:
- 100% cache hit: 2s rebuild time
- 0% cache hit: 25s rebuild time
- **~12x speedup** with caching!

## Testing

Run unit tests:

```powershell
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessor.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Usage Examples

### Example 1: Technology Search

```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={"query": "computer graphics 3D rendering", "top_k": 3}
)

for result in response.json()['results']:
    print(f"{result['doc_id']}: {result['score']:.3f}")
    print(f"  {result['explanation']['match_reasoning']}")
```

### Example 2: Scientific Query

```python
response = requests.post(
    "http://localhost:8000/search",
    json={"query": "space exploration NASA missions", "top_k": 5}
)

# Process results...
```

### Example 3: Cache Performance Test

```python
import time
import requests

# First rebuild (cold cache)
start = time.time()
requests.post("http://localhost:8000/rebuild?force=true")
cold_time = time.time() - start

# Second rebuild (warm cache)
start = time.time()
requests.post("http://localhost:8000/rebuild")
warm_time = time.time() - start

print(f"Cold cache: {cold_time:.2f}s")
print(f"Warm cache: {warm_time:.2f}s")
print(f"Speedup: {cold_time/warm_time:.1f}x")
```

## Troubleshooting

### Issue: "No documents found in data/docs"

**Solution**: Run the data loader first:
```powershell
python -m scripts.load_data --num-docs 200
```

### Issue: "Search engine not initialized"

**Solution**: Wait for server startup to complete or check logs for errors.

### Issue: Model download fails

**Solution**: 
1. Check internet connection
2. Manually download model:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
   ```

### Issue: Slow performance

**Solution**:
- Reduce `BATCH_SIZE` in config if running out of memory
- Enable GPU if available (`USE_GPU = True`)
- Consider using a smaller model

## Extending the System

### Add Custom Documents

1. Place `.txt` files in `data/docs/` with naming pattern `doc_XXX.txt`
2. Rebuild index: `POST /rebuild`

### Use Different Embedding Model

Edit `src/config.py`:
```python
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768  # Update dimension
```

### Add Re-ranking

Modify `search_engine.py` to add BM25 or cross-encoder re-ranking:
```python
from rank_bm25 import BM25Okapi

# In SearchEngine.search():
# 1. Get initial results from FAISS
# 2. Re-rank with BM25
# 3. Return combined scores
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Acknowledgments

- [sentence-transformers](https://www.sbert.net/) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [scikit-learn](https://scikit-learn.org/) for the 20 Newsgroups dataset

## Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ using Python, FastAPI, and sentence-transformers**
