"""
Streamlit UI for the embedding search engine.
"""
import streamlit as st
import requests
from typing import List, Dict, Any
import time

# Configuration
# For local development, use localhost
# For deployment, update to your deployed API URL
API_URL = st.sidebar.text_input(
    "API URL", 
    value="http://localhost:8000",
    help="Enter your FastAPI backend URL"
)

st.set_page_config(
    page_title="Document Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-box {
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .score-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
    }
    .doc-id {
        color: #666;
        font-family: monospace;
    }
    .explanation {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .keyword-tag {
        background-color: #ffd700;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def search_documents(query: str, top_k: int) -> Dict[str, Any]:
    """Search for documents."""
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={"query": query, "top_k": top_k},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def get_stats() -> Dict[str, Any]:
    """Get index statistics."""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}


def rebuild_index(force: bool = False) -> Dict[str, Any]:
    """Rebuild the search index."""
    try:
        response = requests.post(
            f"{API_URL}/rebuild",
            params={"force": force},
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def display_result(result: Dict[str, Any], rank: int):
    """Display a single search result."""
    with st.container():
        st.markdown(f"""
        <div class="result-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3>#{rank} - <span class="doc-id">{result['doc_id']}</span></h3>
                <span class="score-badge">Score: {result['score']:.3f}</span>
            </div>
            <p style="margin-top: 1rem; line-height: 1.6;">{result['preview']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Explanation section (collapsible)
        with st.expander("📊 See Explanation"):
            exp = result['explanation']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cosine Similarity", f"{exp['cosine_similarity']:.3f}")
            with col2:
                st.metric("Overlap Ratio", f"{exp['overlap_ratio']:.1%}")
            with col3:
                st.metric("Normalized Score", f"{exp['length_normalized_score']:.3f}")
            
            # Matched keywords
            if exp['matched_keywords']:
                st.markdown("**Matched Keywords:**")
                keywords_html = " ".join([
                    f'<span class="keyword-tag">{kw}</span>' 
                    for kw in exp['matched_keywords']
                ])
                st.markdown(keywords_html, unsafe_allow_html=True)
            
            # Reasoning
            st.markdown("**Match Reasoning:**")
            st.info(exp['match_reasoning'])


def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Document Search Engine</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ Cannot connect to API. Please ensure the server is running at " + API_URL)
        st.info("Start the server with: `python run_api.py`")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Search parameters
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
        
        st.divider()
        
        # Index management
        st.header("🔧 Index Management")
        
        # Get stats
        stats = get_stats()
        if stats.get('status') == 'ready':
            st.success("✅ Index is ready")
            st.metric("Total Documents", stats.get('total_documents', 'N/A'))
            if 'cache_stats' in stats:
                cache_stats = stats['cache_stats']
                st.metric("Cache Size", f"{cache_stats.get('database_size_mb', 0):.2f} MB")
        else:
            st.warning("⚠️ Index not built")
        
        # Rebuild button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Rebuild", use_container_width=True):
                with st.spinner("Rebuilding index..."):
                    result = rebuild_index(force=False)
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        st.success("Index rebuilt!")
                        st.rerun()
        
        with col2:
            if st.button("⚡ Force", use_container_width=True):
                with st.spinner("Force rebuilding..."):
                    result = rebuild_index(force=True)
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        st.success("Index rebuilt!")
                        st.rerun()
        
        st.divider()
        
        # About
        st.header("ℹ️ About")
        st.markdown("""
        This search engine uses:
        - **Embeddings**: sentence-transformers
        - **Vector Search**: FAISS
        - **Caching**: SQLite with SHA-256
        - **API**: FastAPI
        """)
    
    # Main search interface
    st.markdown("### 🔎 Search Documents")
    
    # Search box
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., quantum physics, space exploration, computer graphics...",
        key="search_query"
    )
    
    # Example queries
    st.markdown("**Try these examples:**")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("🌌 Space exploration"):
            st.session_state.search_query = "space exploration NASA missions"
            st.rerun()
    
    with example_col2:
        if st.button("💻 Computer graphics"):
            st.session_state.search_query = "computer graphics 3D rendering"
            st.rerun()
    
    with example_col3:
        if st.button("🧬 Biology genetics"):
            st.session_state.search_query = "biology genetics DNA research"
            st.rerun()
    
    # Search button
    if st.button("🔍 Search", type="primary", use_container_width=True) or query:
        if query.strip():
            with st.spinner("Searching..."):
                start_time = time.time()
                results = search_documents(query, top_k)
                search_time = time.time() - start_time
                
                if 'error' in results:
                    st.error(results['error'])
                elif 'results' in results:
                    # Display results
                    st.success(f"Found {results['total_results']} results in {search_time:.3f}s")
                    
                    st.divider()
                    
                    if results['results']:
                        for idx, result in enumerate(results['results'], 1):
                            display_result(result, idx)
                    else:
                        st.info("No results found. Try a different query.")
        else:
            st.warning("Please enter a search query.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        Built with ❤️ using Python, FastAPI, and sentence-transformers
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
