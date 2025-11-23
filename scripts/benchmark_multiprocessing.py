"""
Benchmark script to compare single vs multiprocessing embedding performance.
"""
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import Embedder
from src.config import EMBEDDING_MODEL


def generate_sample_texts(n: int) -> list:
    """Generate sample texts for benchmarking."""
    texts = [
        f"This is a sample document number {i} about various topics including "
        f"technology, science, business, and politics. It contains information "
        f"that could be relevant for semantic search and information retrieval."
        for i in range(n)
    ]
    return texts


def benchmark_embedding(embedder: Embedder, texts: list, use_multiprocessing: bool, batch_size: int):
    """Benchmark embedding performance."""
    start_time = time.time()
    
    embeddings = embedder.embed_batch(
        texts,
        batch_size=batch_size,
        use_multiprocessing=use_multiprocessing
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    return elapsed, embeddings.shape


def main():
    print("=" * 60)
    print("Multiprocessing Embedding Benchmark")
    print("=" * 60)
    
    # Initialize embedder
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedder = Embedder()
    print(f"Model loaded on device: {embedder.device}")
    print(f"Embedding dimension: {embedder.embedding_dim}")
    
    # Test different document counts
    document_counts = [50, 100, 200, 500]
    
    for n_docs in document_counts:
        print(f"\n{'=' * 60}")
        print(f"Testing with {n_docs} documents")
        print(f"{'=' * 60}")
        
        texts = generate_sample_texts(n_docs)
        
        # Test without multiprocessing
        print(f"\n[1/2] Without multiprocessing (batch_size=32)...")
        time_single, shape_single = benchmark_embedding(
            embedder, texts, use_multiprocessing=False, batch_size=32
        )
        print(f"  ✓ Time: {time_single:.2f}s")
        print(f"  ✓ Shape: {shape_single}")
        print(f"  ✓ Speed: {n_docs/time_single:.1f} docs/sec")
        
        # Test with multiprocessing
        print(f"\n[2/2] With multiprocessing (batch_size=64)...")
        time_multi, shape_multi = benchmark_embedding(
            embedder, texts, use_multiprocessing=True, batch_size=64
        )
        print(f"  ✓ Time: {time_multi:.2f}s")
        print(f"  ✓ Shape: {shape_multi}")
        print(f"  ✓ Speed: {n_docs/time_multi:.1f} docs/sec")
        
        # Calculate speedup
        speedup = time_single / time_multi
        print(f"\n  🚀 Speedup: {speedup:.2f}x faster")
        print(f"  💾 Time saved: {time_single - time_multi:.2f}s ({(1 - time_multi/time_single)*100:.1f}% faster)")
    
    print(f"\n{'=' * 60}")
    print("Benchmark complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
