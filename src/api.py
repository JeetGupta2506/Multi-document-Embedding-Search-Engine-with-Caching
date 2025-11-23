"""
FastAPI application for the embedding search engine.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from .config import (
    API_TITLE,
    API_VERSION,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    DATA_DIR
)
from .embedder import Embedder, create_embedder
from .cache_manager import CacheManager
from .search_engine import SearchEngine
from .preprocessor import DocumentPreprocessor, process_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
embedder: Optional[Embedder] = None
cache_manager: Optional[CacheManager] = None
search_engine: Optional[SearchEngine] = None

# FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Production-ready embedding-based search engine with intelligent caching"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class SearchRequest(BaseModel):
    """Search request schema."""
    query: str = Field(..., description="Search query text", min_length=1)
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        description=f"Number of results to return (max {MAX_TOP_K})",
        ge=1,
        le=MAX_TOP_K
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty after stripping."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class ExplanationModel(BaseModel):
    """Explanation schema."""
    matched_keywords: List[str]
    overlap_ratio: float
    length_normalized_score: float
    cosine_similarity: float
    match_reasoning: str


class SearchResult(BaseModel):
    """Single search result schema."""
    doc_id: str
    score: float
    preview: str
    explanation: ExplanationModel


class SearchResponse(BaseModel):
    """Search response schema."""
    query: str
    top_k: int
    results: List[SearchResult]
    total_results: int


class IndexStats(BaseModel):
    """Index statistics schema."""
    status: str
    total_documents: Optional[int] = None
    index_vectors: Optional[int] = None
    embedding_dimension: Optional[int] = None
    cache_stats: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
    index_ready: bool


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize search engine on startup."""
    global embedder, cache_manager, search_engine
    
    logger.info("Initializing search engine...")
    
    try:
        # Initialize components
        embedder = create_embedder()
        cache_manager = CacheManager()
        search_engine = SearchEngine(embedder, cache_manager)
        
        # Try to load existing index
        if search_engine.load_index():
            logger.info("Loaded existing search index")
        else:
            logger.info("No existing index found. Will build on first search or use /rebuild endpoint.")
        
        logger.info("Search engine initialized successfully")
    
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down search engine...")
    
    if search_engine and search_engine.index is not None:
        search_engine.save_index()
        logger.info("Index saved")


# API endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    index_ready = search_engine is not None and search_engine.index is not None
    
    return {
        "status": "healthy",
        "message": f"{API_TITLE} v{API_VERSION}",
        "index_ready": index_ready
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    index_ready = search_engine.index is not None
    
    return {
        "status": "healthy",
        "message": "Search engine is running",
        "index_ready": index_ready
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for similar documents.
    
    Args:
        request: Search request with query and top_k
        
    Returns:
        Search results with explanations
    """
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    if search_engine.index is None:
        raise HTTPException(
            status_code=400,
            detail="Search index not built. Please use /rebuild endpoint first."
        )
    
    try:
        # Perform search
        results = search_engine.search(
            query=request.query,
            top_k=request.top_k
        )
        
        return {
            "query": request.query,
            "top_k": request.top_k,
            "results": results,
            "total_results": len(results)
        }
    
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/rebuild")
async def rebuild_index(force: bool = Query(False, description="Force rebuild all embeddings")):
    """
    Rebuild the search index from documents in the data directory.
    
    Args:
        force: If True, regenerate all embeddings regardless of cache
        
    Returns:
        Status message with statistics
    """
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    try:
        # Load documents from data directory
        documents = load_documents_from_directory(DATA_DIR)
        
        if not documents:
            raise HTTPException(
                status_code=400,
                detail=f"No documents found in {DATA_DIR}"
            )
        
        logger.info(f"Rebuilding index with {len(documents)} documents")
        
        # Build index
        search_engine.build_index(documents, force_rebuild=force)
        
        # Save index
        search_engine.save_index()
        
        stats = search_engine.get_index_stats()
        
        return {
            "status": "success",
            "message": f"Index rebuilt with {len(documents)} documents",
            "stats": stats
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {str(e)}")


@app.get("/stats", response_model=IndexStats)
async def get_stats():
    """
    Get index statistics.
    
    Returns:
        Current index statistics
    """
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    stats = search_engine.get_index_stats()
    return stats


@app.delete("/cache")
async def clear_cache():
    """
    Clear the embedding cache.
    
    Returns:
        Status message
    """
    if cache_manager is None:
        raise HTTPException(status_code=503, detail="Cache manager not initialized")
    
    try:
        cache_manager.clear_cache()
        return {
            "status": "success",
            "message": "Cache cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


# Helper functions
def load_documents_from_directory(directory: Path) -> Dict[str, str]:
    """
    Load all text documents from a directory (including subdirectories).
    
    Args:
        directory: Path to documents directory
        
    Returns:
        Dictionary mapping doc_id to cleaned text
    """
    documents = {}
    preprocessor = DocumentPreprocessor()
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return documents
    
    # Load all .txt files recursively (including subdirectories)
    txt_files = sorted(directory.rglob("*.txt"))
    
    logger.info(f"Found {len(txt_files)} text files")
    
    for filepath in txt_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
            
            # Clean text
            cleaned_text = preprocessor.clean_text(raw_text)
            
            # Use category/filename as doc_id (e.g., "business/001" or "tech/042")
            relative_path = filepath.relative_to(directory)
            doc_id = str(relative_path.with_suffix('')).replace('\\', '/')
            documents[doc_id] = cleaned_text
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
    
    return documents


# For running with uvicorn
if __name__ == "__main__":
    import uvicorn
    from .config import API_HOST, API_PORT
    
    uvicorn.run(
        "src.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
