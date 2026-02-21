
import sys
import os
import time
import numpy as np
from unittest.mock import MagicMock
import unittest

# Mock google.genai and google.genai.types before importing the module
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["google.genai.types"] = MagicMock()

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# src path is ../src relative to this file
src_path = os.path.join(os.path.dirname(current_dir), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Try to import MatryoshkaIndexer
try:
    from antigravity.k3_mrl_indexer import MatryoshkaIndexer
except ImportError:
    # If package import fails, try direct import from file location
    sys.path.insert(0, os.path.join(src_path, "antigravity"))
    import k3_mrl_indexer
    MatryoshkaIndexer = k3_mrl_indexer.MatryoshkaIndexer

class TestIndexerPerf(unittest.TestCase):
    def setUp(self):
        # Mock API key
        self.indexer = MatryoshkaIndexer(api_key="fake", target_dir=".", index_file="bench_index.pkl")

        # Populate index with dummy data
        count = 5000
        dim = 768
        print(f"\nPopulating index with {count} vectors of dimension {dim}...")

        # Create random vectors
        vectors = np.random.rand(count, dim).astype(np.float32)

        # Populate internal index directly
        self.indexer.index = {}
        for i in range(count):
            self.indexer.index[f"file_{i}"] = {
                "vector": vectors[i],
                "hash": "hash",
                "snippet": "snippet"
            }

        # Mock client.models.embed_content for search
        mock_resp = MagicMock()
        mock_resp.embeddings = [MagicMock(values=np.random.rand(dim).tolist())]
        self.indexer.client.models.embed_content.return_value = mock_resp

    def test_search_performance(self):
        iterations = 20
        print(f"Running search {iterations} times...")

        start_time = time.time()
        for i in range(iterations):
            self.indexer.search("query", top_k=5)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / iterations
        print(f"Total time: {total_time:.4f}s")
        print(f"Average time per search: {avg_time:.4f}s")

        # Simple assertion to ensure it runs
        self.assertLess(avg_time, 1.0, "Search took too long!")

if __name__ == "__main__":
    unittest.main()
