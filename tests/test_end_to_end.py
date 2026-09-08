"""
End-to-End Integration Test for SmartShop AI
Verifies that the canonical query:
"I need a laptop under 70000 for Python, machine learning and college work."
executes through intent extraction, vector embedding, hybrid ranking,
and returns budget-compliant, ML-ready laptops with complete explainability dossiers.
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommendation.ranker import get_recommendation_engine
from recommendation.explainer import get_recommendation_explainer


class TestEndToEndPipeline(unittest.TestCase):

    def setUp(self):
        self.engine = get_recommendation_engine()
        self.explainer = get_recommendation_explainer()

    def test_canonical_student_query(self):
        query = "I need a laptop under 70000 for Python, machine learning and college work."
        recs, intent = self.engine.recommend(query, top_k=5)

        # 1. Intent verification
        self.assertEqual(intent.category, "Laptop")
        self.assertEqual(intent.budget_max, 70000)
        self.assertIn("machine_learning", intent.use_cases)
        self.assertIn("python", intent.use_cases)

        # 2. Recommendations returned
        self.assertEqual(len(recs), 5)

        # 3. Verify top recommendation is budget compliant
        top_prod = recs[0].product
        self.assertLessEqual(top_prod["price"], 70000, "Top recommendation must be under ₹70,000.")
        self.assertEqual(top_prod["category"], "Laptop")

        # 4. Verify technical capabilities (16GB RAM for ML)
        self.assertIn("16GB", top_prod["ram"], "Should recommend 16GB RAM for machine learning workloads.")

        # 5. Explainability verification
        dossier = self.explainer.explain(recs[0], intent)
        self.assertTrue(len(dossier["narrative"]) > 50)
        self.assertIn("score_breakdown", dossier)
        self.assertGreaterEqual(dossier["score_breakdown"]["composite_pct"], 60)


if __name__ == "__main__":
    unittest.main()
