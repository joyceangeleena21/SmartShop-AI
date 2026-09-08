"""
Unit tests for Ranking Metrics & Evaluator
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator import RecommendationEvaluator


class TestEvaluator(unittest.TestCase):

    def test_precision_at_k(self):
        rec_ids = ["P1", "P2", "P3", "P4", "P5"]
        ground_truth = {"P1": 3, "P2": 2, "P3": 0, "P4": 1, "P5": 0}
        # P1 (3) and P2 (2) meet threshold >= 2 -> 2/5 = 0.40
        p = RecommendationEvaluator.precision_at_k(rec_ids, ground_truth, k=5, threshold=2)
        self.assertEqual(p, 0.40)

    def test_mrr(self):
        rec_ids = ["P1", "P2", "P3"]
        ground_truth = {"P1": 0, "P2": 3, "P3": 2}
        # First relevant is at rank 2 -> MRR = 0.5
        mrr = RecommendationEvaluator.reciprocal_rank(rec_ids, ground_truth, threshold=2)
        self.assertEqual(mrr, 0.5)

    def test_ndcg_at_k(self):
        # Perfect ordering
        rec_ids = ["P1", "P2"]
        ground_truth = {"P1": 3, "P2": 2}
        ndcg = RecommendationEvaluator.ndcg_at_k(rec_ids, ground_truth, k=2)
        self.assertAlmostEqual(ndcg, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
