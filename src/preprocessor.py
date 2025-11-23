"""
Text preprocessing utilities for document cleaning.
"""
import re
import hashlib
from typing import Dict, Any
from bs4 import BeautifulSoup


class DocumentPreprocessor:
    """Handles text cleaning and document metadata extraction."""
    
    def __init__(self):
        self.html_parser = "lxml"
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by:
        - Removing HTML tags
        - Converting to lowercase
        - Removing extra whitespace
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text string
        """
        # Remove HTML tags
        text = self._strip_html(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = self._normalize_whitespace(text)
        
        return text.strip()
    
    def _strip_html(self, text: str) -> str:
        """Remove HTML tags using BeautifulSoup."""
        try:
            soup = BeautifulSoup(text, self.html_parser)
            return soup.get_text()
        except Exception:
            # Fallback to simple regex if parsing fails
            return re.sub(r'<[^>]+>', '', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Replace multiple whitespace characters with a single space."""
        # Replace tabs, newlines, etc. with space
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def compute_hash(self, text: str) -> str:
        """
        Compute SHA-256 hash of text for cache validation.
        
        Args:
            text: Text to hash
            
        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def extract_metadata(self, text: str, filename: str, doc_id: str) -> Dict[str, Any]:
        """
        Extract document metadata.
        
        Args:
            text: Cleaned document text
            filename: Original filename
            doc_id: Document identifier
            
        Returns:
            Dictionary containing metadata
        """
        return {
            'doc_id': doc_id,
            'filename': filename,
            'length': len(text),
            'word_count': len(text.split()),
            'hash': self.compute_hash(text)
        }
    
    def create_preview(self, text: str, max_length: int = 200) -> str:
        """
        Create a preview snippet of the document.
        
        Args:
            text: Document text
            max_length: Maximum preview length in characters
            
        Returns:
            Preview string with ellipsis if truncated
        """
        if len(text) <= max_length:
            return text
        
        # Try to cut at last complete word
        preview = text[:max_length]
        last_space = preview.rfind(' ')
        
        if last_space > max_length * 0.8:  # If space is not too far back
            preview = preview[:last_space]
        
        return preview + "..."


def process_document(text: str, filename: str, doc_id: str) -> Dict[str, Any]:
    """
    Complete document processing pipeline.
    
    Args:
        text: Raw document text
        filename: Original filename
        doc_id: Document identifier
        
    Returns:
        Dictionary with cleaned text and metadata
    """
    preprocessor = DocumentPreprocessor()
    
    cleaned_text = preprocessor.clean_text(text)
    metadata = preprocessor.extract_metadata(cleaned_text, filename, doc_id)
    
    return {
        'text': cleaned_text,
        'metadata': metadata
    }
