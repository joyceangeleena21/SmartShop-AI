"""
Unit tests for Product Dataset Loader & Feature Enrichment
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset_loader import get_dataset_loader


class TestProductDataset(unittest.TestCase):

    def setUp(self):
        self.loader = get_dataset_loader()
        self.df = self.loader.load()

    def test_catalog_not_empty(self):
        self.assertGreater(len(self.df), 20, "Catalog should have at least 20 items.")

    def test_required_columns_exist(self):
        required = ["product_id", "title", "price", "rating", "semantic_representation", "category"]
        for col in required:
            self.assertIn(col, self.df.columns)

    def test_prices_and_ratings_valid(self):
        self.assertTrue((self.df["price"] > 0).all(), "All product prices must be positive.")
        self.assertTrue((self.df["rating"] >= 1.0).all() and (self.df["rating"] <= 5.0).all(), "Ratings must be between 1.0 and 5.0.")

    def test_semantic_representation_is_rich(self):
        sample = self.df.iloc[0]["semantic_representation"]
        self.assertIsInstance(sample, str)
        self.assertGreater(len(sample), 80, "Semantic representation must be comprehensive.")

    def test_filtering(self):
        laptops = self.loader.filter_catalog(category="Laptop")
        self.assertTrue((laptops["category"] == "Laptop").all())


if __name__ == "__main__":
    unittest.main()
