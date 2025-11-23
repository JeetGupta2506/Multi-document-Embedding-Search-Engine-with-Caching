# 🚀 Getting Started - 5 Minute Quick Guide

Welcome! This guide will get you up and running in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- ~500 MB free disk space (for model and dependencies)

## Option 1: Automated Setup (Recommended)

```powershell
python quickstart.py
```

This script will:
1. Check your Python version
2. Install dependencies
3. Load sample data
4. Offer to start the server

Just follow the prompts!

## Option 2: Manual Setup (3 Steps)

### Step 1: Install Dependencies (2 minutes)

```powershell
pip install -r requirements.txt
```

This installs:
- FastAPI (API framework)
- sentence-transformers (embeddings)
- FAISS (vector search)
- scikit-learn (dataset)
- Other utilities

### Step 2: Load Sample Data (1 minute)

```powershell
python -m scripts.load_data --num-docs 200
```

This downloads the 20 Newsgroups dataset and saves 200 text documents to `data/docs/`.

### Step 3: Start the Server (10 seconds)

```powershell
python run_api.py
```

Server starts at: `http://localhost:8000`

## First Actions

### 1. View API Documentation

Open in browser: `http://localhost:8000/docs`

You'll see interactive API documentation with all endpoints.

### 2. Build the Search Index

**Option A - Using PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/rebuild" -Method Post
```

**Option B - Using API Docs:**
1. Go to `http://localhost:8000/docs`
2. Find `POST /rebuild`
3. Click "Try it out" → "Execute"

Wait ~30 seconds for first build (downloads model + generates embeddings).

### 3. Try Your First Search

**Option A - Using PowerShell:**
```powershell
$body = @{
    query = "space exploration"
    top_k = 5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/search" -Method Post -Body $body -ContentType "application/json"
```

**Option B - Using API Docs:**
1. Find `POST /search`
2. Click "Try it out"
3. Enter query: `"space exploration"`
4. Click "Execute"

**Option C - Using Streamlit UI (Best Experience):**

In a new terminal:
```powershell
streamlit run streamlit_app.py
```

Then open the URL shown (usually `http://localhost:8501`).

## What You Get

### Search Results Include:

```json
{
  "doc_id": "doc_042",
  "score": 0.856,
  "preview": "NASA's space exploration missions...",
  "explanation": {
    "matched_keywords": ["space", "exploration", "nasa"],
    "overlap_ratio": 0.667,
    "cosine_similarity": 0.856,
    "match_reasoning": "This document shows strong semantic similarity..."
  }
}
```

## Next Steps

### Try Different Queries

```
"quantum physics"
"computer graphics 3D rendering"
"artificial intelligence machine learning"
"medical diagnosis treatment"
"climate change environment"
```

### Explore the Cache

Run rebuild again - it will be 12x faster!

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/rebuild" -Method Post
```

Notice the speed difference? That's the smart caching in action!

### Run the Interactive Demo

```powershell
python demo.py
```

This provides:
- Interactive search
- Cache performance testing
- Sample queries
- Statistics

### Check Statistics

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/stats" -Method Get
```

See how many documents are indexed and cache size.

## Troubleshooting

### "ModuleNotFoundError"

```powershell
pip install -r requirements.txt
```

### "No documents found"

```powershell
python -m scripts.load_data --num-docs 200
```

### "Port 8000 already in use"

Edit `src/config.py`:
```python
API_PORT = 8001  # Change to available port
```

### Model download fails

Check internet connection. Model is ~80MB and auto-downloads on first use.

### Slow performance

First run is slower (model download + embedding generation). Subsequent runs use cache and are much faster.

## File Locations

- **Documents**: `data/docs/*.txt`
- **Cache**: `data/cache/embeddings_cache.db`
- **FAISS Index**: `data/cache/faiss_index.bin`
- **Logs**: Console output (can be configured)

## Common Tasks

### Add Your Own Documents

1. Place `.txt` files in `data/docs/` with names like `doc_201.txt`, `doc_202.txt`, etc.
2. Rebuild: `POST /rebuild`
3. Search!

### Clear Cache (Force Rebuild)

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/cache" -Method Delete
```

Or rebuild with force flag:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/rebuild?force=true" -Method Post
```

### Stop the Server

Press `Ctrl+C` in the terminal where the server is running.

### Run Tests

```powershell
pytest tests/ -v
```

## Architecture Overview

```
Your Query → API → Search Engine → FAISS Index → Results
                        ↓
                   Cache Manager (SQLite)
                        ↓
                   Smart Caching (SHA-256)
```

When you search:
1. Query is embedded using sentence-transformers
2. FAISS finds similar document embeddings
3. Results are ranked and explained
4. You get top-k matches with reasoning

When you rebuild:
1. Documents are loaded and cleaned
2. For each document:
   - Check cache using SHA-256 hash
   - If cached → reuse embedding (fast!)
   - If not → generate embedding (slower)
3. Build FAISS index
4. Save for next time

## Learn More

- **Full Documentation**: `README.md`
- **Setup Guide**: `SETUP.md`
- **Usage Examples**: `EXAMPLES.md`
- **Project Summary**: `PROJECT_SUMMARY.md`
- **Checklist**: `CHECKLIST.md`

## Support

If you encounter issues:
1. Check `SETUP.md` troubleshooting section
2. Review logs in the terminal
3. Verify all files exist in correct locations
4. Ensure Python 3.8+ is installed

## Quick Reference Commands

```powershell
# Setup
pip install -r requirements.txt
python -m scripts.load_data --num-docs 200

# Run
python run_api.py                  # Start API
streamlit run streamlit_app.py    # Start UI
python demo.py                     # Run demo

# Test
pytest tests/ -v                   # Run tests

# API Operations (in another terminal)
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
Invoke-RestMethod -Uri "http://localhost:8000/rebuild" -Method Post
Invoke-RestMethod -Uri "http://localhost:8000/stats" -Method Get
```

---

**Ready to search! 🔍 Enjoy your new semantic search engine!**

For the best experience, start with the Streamlit UI:
```powershell
streamlit run streamlit_app.py
```
