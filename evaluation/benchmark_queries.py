"""
SmartShop AI - Benchmark Queries & Ground Truth Evaluation Dataset
Contains 10 canonical multi-intent test scenarios with graded relevance judgements
for offline evaluation of Precision@K, Recall@K, MRR, and NDCG@K.
"""

from typing import Any, Dict, List

BENCHMARK_QUERIES: List[Dict[str, Any]] = [
    {
        "query_id": "Q01",
        "query": "I need a laptop under 70000 for Python, machine learning and college work.",
        "category": "Laptop",
        "budget_max": 70000,
        "description": "Student seeking ML/coding laptop under 70k with GPU/16GB RAM",
        # Graded relevance: 3 = Perfect, 2 = Good, 1 = Acceptable, 0 = Irrelevant
        "ground_truth_relevance": {
            "LAP_001": 3,  # Acer Nitro V 15 (RTX 3050, 16GB, 66.9k)
            "LAP_002": 3,  # Lenovo LOQ 15 (RTX 3050, 16GB, 68.9k)
            "LAP_003": 3,  # HP Victus 15 (RTX 3050, 16GB, 62.9k)
            "LAP_007": 3,  # Acer Aspire 7 (RTX 2050, 16GB, 50.9k)
            "LAP_018": 3,  # MSI Thin 15 (RTX 3050, 16GB, 58.9k)
            "LAP_006": 2,  # ASUS Vivobook 16X (RTX 2050, 16GB, 53.9k)
            "LAP_008": 1,  # Lenovo IdeaPad Slim 3 (Iris Xe, 16GB, 54.9k)
            "LAP_019": 1,  # Dell Inspiron 14 (Iris Xe, 16GB, 64.9k)
            "LAP_004": 0,  # ASUS TUF A15 (Over budget: 79.9k)
            "LAP_016": 0,  # Lenovo Legion (Over budget: 1.54L)
        }
    },
    {
        "query_id": "Q02",
        "query": "Gaming laptop with RTX 3050 and 16GB RAM under 75k",
        "category": "Laptop",
        "budget_max": 75000,
        "description": "Gamer looking for RTX 3050 + 16GB RAM within 75k budget",
        "ground_truth_relevance": {
            "LAP_002": 3,  # Lenovo LOQ 15 (95W RTX 3050, 68.9k)
            "LAP_001": 3,  # Acer Nitro V (RTX 3050, 66.9k)
            "LAP_003": 3,  # HP Victus 15 (RTX 3050, 62.9k)
            "LAP_005": 3,  # Dell G15 (RTX 3050, 71.9k)
            "LAP_018": 3,  # MSI Thin 15 (RTX 3050, 58.9k)
            "LAP_007": 2,  # Acer Aspire 7 (RTX 2050, 50.9k)
            "LAP_004": 0,  # ASUS TUF (RTX 4050, 79.9k - over budget)
            "LAP_008": 0,  # IdeaPad (No GPU)
        }
    },
    {
        "query_id": "Q03",
        "query": "Lightweight ultrabook under 60000 for long battery life and coding",
        "category": "Laptop",
        "budget_max": 60000,
        "description": "Programmer needing portable ultrabook with great battery under 60k",
        "ground_truth_relevance": {
            "LAP_008": 3,  # Lenovo IdeaPad Slim 3 (1.62kg, 8hr battery, 54.9k)
            "LAP_009": 3,  # HP 15s (1.69kg, 7hr battery, 51.9k)
            "LAP_006": 2,  # ASUS Vivobook 16X (1.8kg, 53.9k)
            "LAP_019": 1,  # Dell Inspiron 14 (1.54kg, but 64.9k - slight over budget)
            "LAP_010": 0,  # Acer Swift Go (74.9k - over budget)
            "LAP_002": 0,  # Lenovo LOQ (Heavy gaming 2.4kg)
        }
    },
    {
        "query_id": "Q04",
        "query": "Budget student laptop under 40000 for online classes and MS Office",
        "category": "Laptop",
        "budget_max": 40000,
        "description": "Student looking for affordable everyday laptop under 40k",
        "ground_truth_relevance": {
            "LAP_011": 3,  # Lenovo IdeaPad 1 (Ryzen 3, 32.9k)
            "LAP_012": 3,  # ASUS Vivobook Go 15 (i3, 35.9k)
            "LAP_013": 3,  # HP 255 G9 (Ryzen 5, 38.9k)
            "LAP_007": 0,  # Over budget (50.9k)
            "LAP_001": 0,  # Over budget (66.9k)
        }
    },
    {
        "query_id": "Q05",
        "query": "MacBook for iOS development and Swift under 1.3 lakh",
        "category": "Laptop",
        "budget_max": 130000,
        "description": "iOS app developer seeking Apple MacBook within 1.3L",
        "ground_truth_relevance": {
            "LAP_014": 3,  # MacBook Air M2 (89.9k)
            "LAP_015": 3,  # MacBook Air M3 16GB (1.24L)
            "LAP_017": 0,  # ASUS Zephyrus (Windows)
            "LAP_016": 0,  # Lenovo Legion (Windows)
        }
    },
    {
        "query_id": "Q06",
        "query": "Best smartphone under 25000 with 120Hz AMOLED display and good battery",
        "category": "Smartphone",
        "budget_max": 25000,
        "description": "Phone buyer wanting 120Hz AMOLED & strong battery under 25k",
        "ground_truth_relevance": {
            "PHN_002": 3,  # Nothing Phone 2a (120Hz AMOLED, 5000mAh, 23.9k)
            "PHN_007": 3,  # OnePlus Nord CE 4 (120Hz AMOLED, 5500mAh, 24.9k)
            "PHN_005": 1,  # Redmi Note 13 Pro+ (31.9k - over budget)
            "PHN_001": 0,  # OnePlus 12R (45.9k - over budget)
        }
    },
    {
        "query_id": "Q07",
        "query": "Noise cancelling wireless headphones under 10000 for study and focus",
        "category": "Headphones",
        "budget_max": 10000,
        "description": "Student seeking ANC headphones under 10k for library focus",
        "ground_truth_relevance": {
            "AUD_002": 3,  # Sony WH-CH720N (ANC, 35hr, 8.9k)
            "AUD_005": 3,  # OnePlus Buds Pro 2 (48dB ANC, 8.9k)
            "AUD_007": 3,  # Nothing Ear (a) (45dB ANC, 7.9k)
            "AUD_006": 1,  # boAt Rockerz 550 (1.7k, passive noise only)
            "AUD_001": 0,  # Sony WH-1000XM5 (28.9k - over budget)
            "AUD_003": 0,  # Bose QC 45 (22.9k - over budget)
        }
    },
    {
        "query_id": "Q08",
        "query": "Top tier premium noise cancelling headphones with best sound quality",
        "category": "Headphones",
        "budget_max": None,
        "description": "Audiophile seeking absolute best ANC headphones regardless of budget",
        "ground_truth_relevance": {
            "AUD_001": 3,  # Sony WH-1000XM5 (28.9k)
            "AUD_003": 3,  # Bose QuietComfort 45 (22.9k)
            "AUD_004": 3,  # Apple AirPods Pro 2 (20.9k)
            "AUD_002": 1,  # Sony WH-CH720N (Budget ANC)
            "AUD_006": 0,  # boAt Rockerz 550 (Budget)
        }
    }
]
