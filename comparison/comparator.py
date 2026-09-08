"""
SmartShop AI - Product Comparison Matrix & Trade-off Analyzer
Enables multi-attribute side-by-side comparison between 2 to 4 products,
computes attribute differentials, highlights category winners,
and formats data for interactive Plotly radar visualizations.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CURRENCY_SYMBOL


class ProductComparator:
    """Provides side-by-side technical comparison, winner tagging, and radar metrics."""

    COMPARISON_ATTRIBUTES = [
        ("Price", "price_formatted"),
        ("Original Price", "original_price"),
        ("Discount", "discount_pct"),
        ("Rating", "rating"),
        ("Reviews", "review_count"),
        ("Brand", "brand"),
        ("Category", "category"),
        ("Processor", "processor"),
        ("Graphics / GPU", "gpu"),
        ("RAM", "ram"),
        ("Storage", "storage"),
        ("Display", "display"),
        ("Battery Life", "battery_life"),
        ("Weight", "weight"),
        ("Operating System", "os"),
        ("Intended Use Cases", "use_cases"),
    ]

    def __init__(self):
        pass

    def build_comparison_table(self, products: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Builds a transposed comparison DataFrame:
        Rows = Attributes, Columns = Product Titles (abbreviated).
        """
        if not products:
            return pd.DataFrame()

        rows = []
        for label, key in self.COMPARISON_ATTRIBUTES:
            row_dict = {"Attribute": label}
            for idx, p in enumerate(products, 1):
                col_name = f"{p.get('brand', '')} {p.get('title', f'Product {idx}')[:28]}..."
                val = p.get(key, "N/A")
                if key == "original_price" and isinstance(val, (int, float)):
                    row_dict[col_name] = f"{CURRENCY_SYMBOL}{val:,}"
                elif key == "discount_pct" and isinstance(val, (int, float)):
                    row_dict[col_name] = f"{val:.1f}% OFF"
                elif key == "rating" and isinstance(val, (int, float)):
                    row_dict[col_name] = f"{val}★"
                elif key == "review_count" and isinstance(val, (int, float)):
                    row_dict[col_name] = f"{val:,} reviews"
                elif key == "use_cases" and isinstance(val, str):
                    row_dict[col_name] = val.replace("_", " ").title()
                else:
                    row_dict[col_name] = str(val)
            rows.append(row_dict)

        df_comp = pd.DataFrame(rows)
        return df_comp

    def find_category_winners(self, products: List[Dict[str, Any]]) -> Dict[str, str]:
        """Identifies which product wins on specific dimensions."""
        if not products:
            return {}

        winners = {}

        # 1. Best Value / Lowest Price
        lowest_price_prod = min(products, key=lambda x: x.get("price", float("inf")))
        winners["Best Value (Lowest Price)"] = f"{lowest_price_prod.get('title')[:35]}... ({lowest_price_prod.get('price_formatted')})"

        # 2. Highest Customer Rating
        highest_rating_prod = max(products, key=lambda x: (x.get("rating", 0.0), x.get("review_count", 0)))
        winners["Highest Customer Rating"] = f"{highest_rating_prod.get('title')[:35]}... ({highest_rating_prod.get('rating')}★)"

        # 3. Maximum Discount
        highest_discount_prod = max(products, key=lambda x: x.get("discount_pct", 0.0))
        winners["Biggest Discount"] = f"{highest_discount_prod.get('title')[:35]}... ({highest_discount_prod.get('discount_pct')}%)"

        # 4. Most Portable (Weight)
        weights = []
        for p in products:
            wt_str = str(p.get("weight", "")).lower()
            try:
                if "kg" in wt_str:
                    val = float(wt_str.replace("kg", "").strip())
                elif "g" in wt_str:
                    val = float(wt_str.replace("g", "").strip()) / 1000.0
                else:
                    val = 99.0
                weights.append((val, p))
            except Exception:
                weights.append((99.0, p))

        lightest = min(weights, key=lambda x: x[0])[1]
        winners["Most Portable / Lightweight"] = f"{lightest.get('title')[:35]}... ({lightest.get('weight')})"

        return winners

    def compute_radar_metrics(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Computes 0-100 normalized scores for Plotly Radar Charts across 5 dimensions:
        - Performance
        - Value for Money
        - Portability
        - User Rating
        - Display Quality
        """
        radar_data = []

        for p in products:
            # Dimension 1: Performance (Processor + GPU + RAM)
            perf_score = 60
            gpu_str = str(p.get("gpu", "")).upper()
            ram_str = str(p.get("ram", "")).upper()
            proc_str = str(p.get("processor", "")).upper()

            if "RTX 4070" in gpu_str or "RTX 4060" in gpu_str:
                perf_score += 35
            elif "RTX 4050" in gpu_str or "RTX 3050" in gpu_str or "M3" in proc_str:
                perf_score += 25
            elif "RTX 2050" in gpu_str or "M2" in proc_str:
                perf_score += 15

            if "32GB" in ram_str:
                perf_score += 10
            elif "16GB" in ram_str:
                perf_score += 5

            perf_score = min(100, perf_score)

            # Dimension 2: Value for Money (Inversely proportional to price, boosted by discount)
            price = p.get("price", 50000)
            discount = p.get("discount_pct", 0)
            value_score = int(np.clip(100 - (price / 2000) + (discount * 0.5), 30, 98))

            # Dimension 3: Portability
            wt_str = str(p.get("weight", "")).lower()
            try:
                if "kg" in wt_str:
                    wt_kg = float(wt_str.replace("kg", "").strip())
                elif "g" in wt_str:
                    wt_kg = float(wt_str.replace("g", "").strip()) / 1000.0
                else:
                    wt_kg = 2.0
            except Exception:
                wt_kg = 2.0
            portability_score = int(np.clip(100 - (wt_kg * 25), 25, 95))

            # Dimension 4: User Satisfaction
            rating = float(p.get("rating", 4.0))
            user_sat_score = int((rating / 5.0) * 100)

            # Dimension 5: Display Quality
            disp_str = str(p.get("display", "")).upper()
            disp_score = 60
            if "OLED" in disp_str or "2.8K" in disp_str or "3K" in disp_str:
                disp_score += 35
            elif "144HZ" in disp_str or "120HZ" in disp_str:
                disp_score += 20
            elif "FHD" in disp_str:
                disp_score += 10
            disp_score = min(100, disp_score)

            radar_data.append({
                "product_name": f"{p.get('brand', '')} {p.get('title', '')[:20]}...",
                "scores": [perf_score, value_score, portability_score, user_sat_score, disp_score],
                "metrics": ["Performance", "Value for Money", "Portability", "User Rating", "Display Quality"]
            })

        return radar_data


# Global Singleton
_comparator_instance: Optional[ProductComparator] = None


def get_product_comparator() -> ProductComparator:
    """Returns singleton instance of ProductComparator."""
    global _comparator_instance
    if _comparator_instance is None:
        _comparator_instance = ProductComparator()
    return _comparator_instance


if __name__ == "__main__":
    from data.dataset_loader import get_dataset_loader

    loader = get_dataset_loader()
    df = loader.load()
    sample_prods = [df.iloc[0].to_dict(), df.iloc[1].to_dict()]

    comparator = get_product_comparator()
    comp_df = comparator.build_comparison_table(sample_prods)
    winners = comparator.find_category_winners(sample_prods)
    radar = comparator.compute_radar_metrics(sample_prods)

    print("=" * 70)
    print("Testing ProductComparator Matrix:")
    print("=" * 70)
    print(comp_df.head(8))
    print("\nCategory Winners:")
    for k, v in winners.items():
        print(f"  * {k}: {v}")
    print("\nRadar Data Points:", len(radar))
