"""
Unit tests for SentenceTransformer Embedding Engine
"""

import unittest
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.embedding_engine import get_embedding_engine
from config.config import EMBEDDING_DIM
from data.dataset_loader import get_dataset_loader


class TestEmbeddingEngine(unittest.TestCase):

    def setUp(self):
        self.engine = get_embedding_engine()
        self.loader = get_dataset_loader()
        self.df = self.loader.load()

    def test_query_embedding_shape_and_norm(self):
        query = "laptop for python machine learning"
        vec = self.engine.encode_query(query)
        self.assertEqual(vec.shape, (EMBEDDING_DIM,))
        # Check L2 normalization (norm should be ~1.0)
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_product_embeddings_cache(self):
        embs = self.engine.get_or_create_product_embeddings(self.df)
        self.assertEqual(embs.shape[0], len(self.df))
        self.assertEqual(embs.shape[1], EMBEDDING_DIM)

    def test_cosine_similarity_bounds(self):
        q_vec = self.engine.encode_query("gaming laptop")
        embs = self.engine.get_or_create_product_embeddings(self.df)
        sims = self.engine.compute_similarity(q_vec, embs)
        self.assertTrue((sims >= 0.0).all() and (sims <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
