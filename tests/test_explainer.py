"""
Unit tests for Explainable AI (XAI) Explainer
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommendation.ranker import get_recommendation_engine
from recommendation.explainer import get_recommendation_explainer


class TestExplainer(unittest.TestCase):

    def setUp(self):
        self.engine = get_recommendation_engine()
        self.explainer = get_recommendation_explainer()

    def test_explanation_structure(self):
        query = "I need a laptop under 70000 for Python, machine learning and college work."
        recs, intent = self.engine.recommend(query, top_k=1)
        self.assertTrue(len(recs) > 0)

        dossier = self.explainer.explain(recs[0], intent)
        self.assertIn("headline", dossier)
        self.assertIn("narrative", dossier)
        self.assertIn("score_breakdown", dossier)
        self.assertIn("highlights", dossier)
        self.assertIn("trade_offs", dossier)
        self.assertGreater(len(dossier["narrative"]), 20)
    def test_integrated_gpu_no_dedicated_warning(self):
        # LAP_008 has Intel Iris Xe (Integrated)
        from data.dataset_loader import get_dataset_loader
        loader = get_dataset_loader()
        p = loader.get_product_by_id("LAP_008")
        self.assertIsNotNone(p)

        from recommendation.ranker import RecommendationResult
        from models.intent_extractor import UserIntent

        res = RecommendationResult(
            product=p,
            composite_score=0.75,
            semantic_score=0.60,
            budget_score=0.95,
            rating_score=0.80,
            feature_score=0.70,
            matched_specs=[],
            matched_use_cases=["Python", "College Work"],
            budget_status="₹15,010 Under Budget"
        )
        intent = UserIntent(raw_query="laptop for python and college", category="Laptop", budget_max=70000)
        dossier = self.explainer.explain(res, intent)

        # Check: Must NOT describe GPU as dedicated
        for h in dossier["highlights"]:
            self.assertNotIn("Dedicated", h)
        # Check: Trade-offs must NOT mention dedicated GPU battery warning
        for t in dossier["trade_offs"]:
            self.assertNotIn("Dedicated GPU", t)

    def test_dedicated_gpu_accuracy(self):
        # LAP_001 has NVIDIA RTX 3050 (Dedicated)
        from data.dataset_loader import get_dataset_loader
        loader = get_dataset_loader()
        p = loader.get_product_by_id("LAP_001")
        self.assertIsNotNone(p)

        from recommendation.ranker import RecommendationResult
        from models.intent_extractor import UserIntent

        res = RecommendationResult(
            product=p,
            composite_score=0.85,
            semantic_score=0.70,
            budget_score=0.98,
            rating_score=0.85,
            feature_score=0.90,
            matched_specs=["GPU: NVIDIA GeForce RTX 3050"],
            matched_use_cases=["Machine Learning", "Python"],
            budget_status="₹3,010 Under Budget"
        )
        intent = UserIntent(raw_query="laptop for machine learning", category="Laptop", budget_max=70000, use_cases=["machine_learning"])
        dossier = self.explainer.explain(res, intent)

        # Check: MUST accurately identify dedicated GPU
        has_gpu_highlight = any("Dedicated" in h or "RTX" in h for h in dossier["highlights"])
        self.assertTrue(has_gpu_highlight, "Dedicated GPU should have a dedicated hardware highlight.")


if __name__ == "__main__":
    unittest.main()
