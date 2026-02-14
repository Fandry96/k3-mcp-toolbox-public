import sys
import unittest
import types
import numpy as np
import os
import pickle

# Mock google.genai and google.genai.types BEFORE importing MatryoshkaIndexer
mock_google = types.ModuleType("google")
mock_genai = types.ModuleType("genai")
mock_genai_types = types.ModuleType("types")

class MockClient:
    def __init__(self, api_key=None):
        pass
    @property
    def models(self):
        return MockModels()

class MockModels:
    def embed_content(self, model=None, contents=None, config=None):
        return MockResponse()

class MockResponse:
    def __init__(self):
        self.embeddings = [MockEmbedding()]

class MockEmbedding:
    def __init__(self):
        self.values = np.random.rand(768).tolist()

class MockEmbedContentConfig:
    def __init__(self, output_dimensionality=None):
        pass

mock_genai.Client = MockClient
mock_genai.types = mock_genai_types
mock_genai_types.EmbedContentConfig = MockEmbedContentConfig

mock_google.genai = mock_genai
sys.modules["google"] = mock_google
sys.modules["google.genai"] = mock_genai
sys.modules["google.genai.types"] = mock_genai_types

# Set up path to import MatryoshkaIndexer
# Adjusting path to point to src correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from antigravity.k3_mrl_indexer import MatryoshkaIndexer

class TestMatryoshkaIndexerCaching(unittest.TestCase):
    def setUp(self):
        self.index_file = "test_index_caching.pkl"
        if os.path.exists(self.index_file):
            os.remove(self.index_file)

        # Create a dummy index file
        with open(self.index_file, "wb") as f:
            pickle.dump({}, f)

        self.indexer = MatryoshkaIndexer(api_key="dummy", target_dir=".", index_file=self.index_file)

        # Populate with some dummy data
        self.indexer.index = {
            "file1": {"vector": np.random.rand(768).astype(np.float32), "hash": "h1", "snippet": "s1"},
            "file2": {"vector": np.random.rand(768).astype(np.float32), "hash": "h2", "snippet": "s2"}
        }

    def tearDown(self):
        if os.path.exists(self.index_file):
            try:
                os.remove(self.index_file)
            except:
                pass
        if os.path.exists(self.index_file + ".tmp"):
            try:
                os.remove(self.index_file + ".tmp")
            except:
                pass

    def test_caching_mechanism(self):
        # 1. Verify attributes exist (this checks if we updated __init__)
        self.assertTrue(hasattr(self.indexer, "_matrix_cache"), "Indexer should have _matrix_cache attribute")
        self.assertTrue(hasattr(self.indexer, "_paths_cache"), "Indexer should have _paths_cache attribute")

        # Initially None
        self.assertIsNone(self.indexer._matrix_cache)
        self.assertIsNone(self.indexer._paths_cache)

        # 2. Run search, which should populate cache
        self.indexer.search("query", top_k=1)

        self.assertIsNotNone(self.indexer._matrix_cache)
        self.assertIsNotNone(self.indexer._paths_cache)
        self.assertEqual(len(self.indexer._paths_cache), 2)
        self.assertEqual(self.indexer._matrix_cache.shape[0], 2)

        # 3. Store reference to current cache objects
        matrix_ref = self.indexer._matrix_cache
        paths_ref = self.indexer._paths_cache

        # 4. Run search again, should reuse same objects
        self.indexer.search("query2", top_k=1)
        self.assertIs(self.indexer._matrix_cache, matrix_ref, "Should reuse matrix cache")
        self.assertIs(self.indexer._paths_cache, paths_ref, "Should reuse paths cache")

        # 5. Invalidate cache manually and check rebuild
        self.indexer._matrix_cache = None
        self.indexer._paths_cache = None

        self.indexer.search("query3", top_k=1)
        self.assertIsNot(self.indexer._matrix_cache, matrix_ref, "Should rebuild matrix cache")
        self.assertIsNot(self.indexer._paths_cache, paths_ref, "Should rebuild paths cache")

if __name__ == "__main__":
    unittest.main()
