import sys
import unittest
from unittest.mock import MagicMock
import time
import numpy as np
import os
import shutil
from pathlib import Path

# Setup mocks
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["google.genai.types"] = MagicMock()

# Setup path
sys.path.insert(0, os.path.abspath("antigravity-logicware/src"))

try:
    from antigravity.k3_mrl_indexer import MatryoshkaIndexer
except ImportError:
    # Fallback if running from root
    sys.path.insert(0, os.path.abspath("src"))
    from antigravity.k3_mrl_indexer import MatryoshkaIndexer

class TestMatryoshkaIndexerPerformance(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_indexer_perf_tmp")
        self.test_dir.mkdir(exist_ok=True)
        # Mock API key to avoid error
        self.indexer = MatryoshkaIndexer("fake_key", str(self.test_dir), "test_index.pkl")

        # Populate with dummy data
        self.num_vectors = 5000
        self.dim = 768
        self.vectors = np.random.rand(self.num_vectors, self.dim).astype(np.float32)
        # Normalize
        self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)

        for i in range(self.num_vectors):
            self.indexer.index[f"file_{i}.txt"] = {
                "vector": self.vectors[i],
                "hash": "hash",
                "snippet": f"Snippet for file {i}"
            }

        # Mock embedding
        self.mock_resp = MagicMock()
        self.mock_resp.embeddings = [MagicMock()]
        self.mock_resp.embeddings[0].values = np.random.rand(self.dim).tolist()
        self.indexer.client.models.embed_content.return_value = self.mock_resp

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        if os.path.exists("test_index.pkl"):
            os.remove("test_index.pkl")
        if os.path.exists("mrl_index.pkl"): # Default name
             if os.path.isfile("mrl_index.pkl"):
                os.remove("mrl_index.pkl")

    def test_search_caching(self):
        print("\n--- Benchmarking Search ---")
        # First search (should populate cache)
        start_time = time.perf_counter()
        self.indexer.search("test query 1")
        first_duration = (time.perf_counter() - start_time) * 1000
        print(f"First search time: {first_duration:.2f} ms")

        # Second search (should use cache)
        start_time = time.perf_counter()
        self.indexer.search("test query 2")
        second_duration = (time.perf_counter() - start_time) * 1000
        print(f"Second search time: {second_duration:.2f} ms")

        # Verify cache attributes exist (TDD: this will fail before implementation)
        self.assertTrue(hasattr(self.indexer, "_matrix_cache"), "Indexer should have _matrix_cache attribute")
        self.assertTrue(hasattr(self.indexer, "_paths_cache"), "Indexer should have _paths_cache attribute")

        # Verify cache population
        if self.indexer._matrix_cache is not None:
            print("Cache is populated.")
            self.assertEqual(self.indexer._matrix_cache.shape[0], self.num_vectors)

            # Verify performance improvement
            # We expect significant speedup, but let's be lenient in assertion
            # e.g. second should be at least 20% faster or just very fast
            # np.stack takes ~5-10ms. Cache lookup is ~0ms.
            # Embedding takes time but it is mocked with constant time return.
            # So the difference is mainly np.stack.

            diff = first_duration - second_duration
            print(f"Improvement: {diff:.2f} ms")

            # Assert that second is faster (unless first was paradoxically fast)
            # This might be flaky on extremely fast machines or small N, but with 5000 it should be stable.
            self.assertLess(second_duration, first_duration)

if __name__ == "__main__":
    unittest.main()
