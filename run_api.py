"""
Simple script to run the API server.
"""
import uvicorn
from src.config import API_HOST, API_PORT

if __name__ == "__main__":
    print(f"Starting search engine API on {API_HOST}:{API_PORT}")
    print(f"API docs available at http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(
        "src.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
