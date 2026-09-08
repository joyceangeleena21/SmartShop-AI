"""
Unit tests for Intent Extraction Pipeline
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.intent_extractor import get_intent_extractor


class TestIntentExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = get_intent_extractor()

    def test_primary_student_query(self):
        query = "I need a laptop under 70000 for Python, machine learning and college work."
        intent = self.extractor.extract_intent(query)
        self.assertEqual(intent.category, "Laptop")
        self.assertEqual(intent.budget_max, 70000)
        self.assertIn("machine_learning", intent.use_cases)
        self.assertIn("python", intent.use_cases)
        self.assertIn("college_work", intent.use_cases)

    def test_k_multiplier_budget(self):
        query = "Gaming laptop under 75k with RTX 3050 and 16GB RAM"
        intent = self.extractor.extract_intent(query)
        self.assertEqual(intent.budget_max, 75000)
        self.assertEqual(intent.specs.get("ram"), "16GB")
        self.assertEqual(intent.specs.get("gpu"), "RTX3050")

    def test_lakh_multiplier_budget(self):
        query = "MacBook for iOS development under 1.2 lakh"
        intent = self.extractor.extract_intent(query)
        self.assertEqual(intent.budget_max, 120000)
        self.assertEqual(intent.category, "Laptop")
        self.assertIn("Apple", intent.brands)

    def test_budget_range(self):
        query = "Laptops between 50000 and 70000 for college"
        intent = self.extractor.extract_intent(query)
        self.assertEqual(intent.budget_min, 50000)
        self.assertEqual(intent.budget_max, 70000)


if __name__ == "__main__":
    unittest.main()
