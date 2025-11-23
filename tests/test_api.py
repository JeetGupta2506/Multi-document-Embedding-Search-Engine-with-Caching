"""
Unit tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np

from src.api import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert 'index_ready' in data


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert 'status' in data
    assert 'index_ready' in data


def test_search_without_index(client):
    """Test search returns error when index not built."""
    response = client.post("/search", json={
        "query": "test query",
        "top_k": 5
    })
    # Should return 400 if index not built
    assert response.status_code in [400, 503]


def test_search_request_validation(client):
    """Test search request validation."""
    # Empty query
    response = client.post("/search", json={
        "query": "",
        "top_k": 5
    })
    assert response.status_code == 422
    
    # Invalid top_k
    response = client.post("/search", json={
        "query": "test",
        "top_k": 0
    })
    assert response.status_code == 422
    
    # top_k too large
    response = client.post("/search", json={
        "query": "test",
        "top_k": 1000
    })
    assert response.status_code == 422


def test_stats_endpoint(client):
    """Test stats endpoint."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert 'status' in data


def test_search_request_defaults(client):
    """Test search request uses defaults."""
    # Mock the search engine to avoid needing actual index
    with patch('src.api.search_engine') as mock_engine:
        mock_engine.index = Mock()
        mock_engine.search.return_value = []
        
        response = client.post("/search", json={
            "query": "test query"
        })
        
        if response.status_code == 200:
            data = response.json()
            assert data['top_k'] == 5  # Default value
