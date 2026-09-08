"""
Unit tests for Hybrid Recommendation & Multi-Factor Ranker
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommendation.ranker import get_recommendation_engine


class TestHybridRanker(unittest.TestCase):

    def setUp(self):
        self.engine = get_recommendation_engine()

    def test_ranking_descending_order(self):
        query = "laptop under 70000 for coding"
        results, intent = self.engine.recommend(query, top_k=5)
        self.assertGreaterEqual(len(results), 1)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].composite_score, results[i + 1].composite_score)

    def test_budget_penalty_suppresses_expensive_items(self):
        # When user asks for under 40000, 1.5 Lakh laptops should not be top-1
        query = "budget laptop under 40000"
        results, intent = self.engine.recommend(query, top_k=3)
        top_prod = results[0].product
        self.assertLessEqual(top_prod["price"], 45000, "Top item should respect budget constraint.")

    def test_category_restriction(self):
        query = "wireless noise cancelling headphones"
        results, intent = self.engine.recommend(query, top_k=3, strict_category=True)
        for r in results:
            self.assertEqual(r.product["category"], "Headphones")


if __name__ == "__main__":
    unittest.main()
