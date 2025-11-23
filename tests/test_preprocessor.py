"""
Unit tests for preprocessor module.
"""
import pytest
from src.preprocessor import DocumentPreprocessor, process_document


def test_clean_text_lowercase():
    """Test text is converted to lowercase."""
    preprocessor = DocumentPreprocessor()
    text = "Hello WORLD"
    cleaned = preprocessor.clean_text(text)
    assert cleaned == "hello world"


def test_clean_text_whitespace():
    """Test extra whitespace is removed."""
    preprocessor = DocumentPreprocessor()
    text = "Hello    world\n\ntest\t\ttabs"
    cleaned = preprocessor.clean_text(text)
    assert cleaned == "hello world test tabs"


def test_clean_text_html():
    """Test HTML tags are stripped."""
    preprocessor = DocumentPreprocessor()
    text = "<p>Hello <b>world</b></p>"
    cleaned = preprocessor.clean_text(text)
    assert cleaned == "hello world"


def test_compute_hash_consistency():
    """Test hash is consistent for same text."""
    preprocessor = DocumentPreprocessor()
    text = "Test document"
    hash1 = preprocessor.compute_hash(text)
    hash2 = preprocessor.compute_hash(text)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex length


def test_compute_hash_different():
    """Test different text produces different hashes."""
    preprocessor = DocumentPreprocessor()
    hash1 = preprocessor.compute_hash("Text 1")
    hash2 = preprocessor.compute_hash("Text 2")
    assert hash1 != hash2


def test_extract_metadata():
    """Test metadata extraction."""
    preprocessor = DocumentPreprocessor()
    text = "This is a test document."
    metadata = preprocessor.extract_metadata(text, "test.txt", "doc_001")
    
    assert metadata['doc_id'] == "doc_001"
    assert metadata['filename'] == "test.txt"
    assert metadata['length'] == len(text)
    assert metadata['word_count'] == 5
    assert 'hash' in metadata


def test_create_preview_short():
    """Test preview for short text."""
    preprocessor = DocumentPreprocessor()
    text = "Short text"
    preview = preprocessor.create_preview(text, max_length=200)
    assert preview == "Short text"
    assert "..." not in preview


def test_create_preview_long():
    """Test preview for long text."""
    preprocessor = DocumentPreprocessor()
    text = "word " * 100  # Long text
    preview = preprocessor.create_preview(text, max_length=50)
    assert len(preview) <= 54  # 50 + "..."
    assert preview.endswith("...")


def test_process_document():
    """Test complete document processing."""
    text = "<p>Hello WORLD</p>   Test   "
    result = process_document(text, "test.txt", "doc_001")
    
    assert 'text' in result
    assert 'metadata' in result
    assert result['text'] == "hello world test"
    assert result['metadata']['doc_id'] == "doc_001"
