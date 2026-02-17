import unittest
import sys
import os
import numpy as np
from unittest.mock import MagicMock

# Mock google.genai
mock_genai = MagicMock()
mock_types = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = mock_genai
sys.modules["google.genai.types"] = mock_types

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Handle import
try:
    from antigravity.k3_mrl_indexer import MatryoshkaIndexer
except ImportError:
    # Try importing directly if package structure is tricky
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/antigravity")))
    import k3_mrl_indexer
    MatryoshkaIndexer = k3_mrl_indexer.MatryoshkaIndexer

class TestIndexerCaching(unittest.TestCase):
    def setUp(self):
        self.indexer = MatryoshkaIndexer("fake_key", ".", "test_index.pkl")
        self.indexer.index = {} # Start empty

        # Mock embed response
        self.mock_resp = MagicMock()
        self.mock_resp.embeddings = [MagicMock()]
        self.mock_resp.embeddings[0].values = np.zeros(768).tolist()
        self.indexer.client.models.embed_content.return_value = self.mock_resp

    def test_cache_initialization(self):
        """Test that cache is None on init."""
        self.assertIsNone(self.indexer._matrix_cache)
        self.assertIsNone(self.indexer._paths_cache)

    def test_cache_population(self):
        """Test that cache is populated after access."""
        # Add some data
        self.indexer.index["file1"] = {"vector": np.zeros(768), "hash": "h1", "snippet": "s1"}

        # Access cache via helper
        matrix, paths = self.indexer._get_matrix_and_paths()

        self.assertIsNotNone(self.indexer._matrix_cache)
        self.assertIsNotNone(self.indexer._paths_cache)
        self.assertEqual(len(matrix), 1)
        self.assertEqual(len(paths), 1)

        # Verify it's the same object (cached)
        matrix2, paths2 = self.indexer._get_matrix_and_paths()
        self.assertIs(matrix, matrix2)
        self.assertIs(paths, paths2)

    def test_cache_invalidation(self):
        """Test that cache is invalidated when modified (simulated)."""
        # Populate cache
        self.indexer.index["file1"] = {"vector": np.zeros(768), "hash": "h1", "snippet": "s1"}
        self.indexer._get_matrix_and_paths()
        self.assertIsNotNone(self.indexer._matrix_cache)

        # Simulate invalidation (as done in run_indexing or load_index)
        self.indexer._matrix_cache = None
        self.indexer._paths_cache = None

        self.assertIsNone(self.indexer._matrix_cache)

        # Re-populate
        self.indexer._get_matrix_and_paths()
        self.assertIsNotNone(self.indexer._matrix_cache)

    def test_search_uses_cache(self):
        """Test that search triggers cache population."""
        self.indexer.index["file1"] = {"vector": np.zeros(768), "hash": "h1", "snippet": "s1"}

        self.assertIsNone(self.indexer._matrix_cache)
        self.indexer.search("query")
        self.assertIsNotNone(self.indexer._matrix_cache)

if __name__ == "__main__":
    unittest.main()
