"""
SmartShop AI - Product Dataset Loader & Feature Enrichment Pipeline
Loads raw product catalog data, validates schema, cleans text and numeric attributes,
and constructs dense textual representations for semantic embedding generation.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import PRODUCTS_CSV_PATH, CURRENCY_SYMBOL
from data.dataset_generator import generate_and_save_dataset


class ProductDatasetLoader:
    """Robust loader and feature enricher for the e-commerce product catalog."""

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = Path(csv_path) if csv_path else PRODUCTS_CSV_PATH
        self._df: Optional[pd.DataFrame] = None

    def load(self, force_reload: bool = False) -> pd.DataFrame:
        """Loads and pre-processes product catalog into a clean Pandas DataFrame."""
        if self._df is not None and not force_reload:
            return self._df

        if not self.csv_path.exists():
            print(f"Catalog file not found at {self.csv_path}. Generating starter catalog...")
            generate_and_save_dataset()

        df = pd.read_csv(self.csv_path)
        df = self._clean_and_enrich(df)
        self._df = df
        return self._df

    def _clean_and_enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans columns, normalizes text, and constructs the rich semantic text representation."""
        # Clean text columns
        text_cols = [
            "title", "brand", "category", "subcategory", "ram", "storage",
            "processor", "gpu", "display", "battery_life", "os", "weight",
            "use_cases", "features", "description"
        ]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()

        # Clean numeric columns
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0).astype(int)
        df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce").fillna(df["price"]).astype(int)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(4.0).astype(float)
        df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(50).astype(int)

        # Calculate discount percentage
        df["discount_pct"] = (
            ((df["original_price"] - df["price"]) / df["original_price"].clip(lower=1)) * 100
        ).round(1).clip(lower=0)

        # Formatted price string (e.g. ₹66,990)
        df["price_formatted"] = df["price"].apply(lambda p: f"{CURRENCY_SYMBOL}{p:,}")

        # Construct Dense Semantic Text Representation (T_i)
        # Structured to prioritize key retrieval tokens (category, specs, use cases, capabilities)
        def create_semantic_text(row: pd.Series) -> str:
            parts = [
                f"Product: {row['title']}.",
                f"Brand: {row['brand']}. Category: {row['category']} - {row['subcategory']}.",
                f"Key Specifications: Processor: {row['processor']}; Graphics: {row['gpu']}; RAM: {row['ram']}; Storage: {row['storage']}; Display: {row['display']}; Battery: {row['battery_life']}; OS: {row['os']}.",
                f"Intended Use Cases and Personas: {row['use_cases'].replace('_', ' ')}.",
                f"Key Highlights and Features: {row['features']}.",
                f"Detailed Overview: {row['description']}"
            ]
            return " ".join([p for p in parts if p.strip()])

        df["semantic_representation"] = df.apply(create_semantic_text, axis=1)

        return df

    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """Retrieves a single product dictionary by product_id."""
        df = self.load()
        match = df[df["product_id"] == product_id]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def filter_catalog(
        self,
        category: Optional[str] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        brand: Optional[str] = None,
        min_rating: Optional[float] = None
    ) -> pd.DataFrame:
        """Filters the catalog based on explicit structured criteria."""
        df = self.load().copy()
        if category and category.lower() != "all":
            df = df[df["category"].str.lower() == category.lower()]
        if min_price is not None:
            df = df[df["price"] >= min_price]
        if max_price is not None:
            df = df[df["price"] <= max_price]
        if brand and brand.lower() != "all":
            df = df[df["brand"].str.lower() == brand.lower()]
        if min_rating is not None:
            df = df[df["rating"] >= min_rating]
        return df

    def get_all_categories(self) -> List[str]:
        """Returns sorted list of distinct product categories."""
        df = self.load()
        return sorted(df["category"].dropna().unique().tolist())

    def get_all_brands(self) -> List[str]:
        """Returns sorted list of distinct brands."""
        df = self.load()
        return sorted(df["brand"].dropna().unique().tolist())

    def get_catalog_stats(self) -> Dict:
        """Computes high-level catalog statistics for dashboard reporting."""
        df = self.load()
        return {
            "total_products": len(df),
            "categories_count": df["category"].nunique(),
            "brands_count": df["brand"].nunique(),
            "price_min": int(df["price"].min()),
            "price_max": int(df["price"].max()),
            "price_avg": int(df["price"].mean()),
            "avg_rating": float(round(df["rating"].mean(), 2)),
        }


# Global Singleton for caching across Streamlit sessions
_loader_instance: Optional[ProductDatasetLoader] = None


def get_dataset_loader() -> ProductDatasetLoader:
    """Returns singleton instance of ProductDatasetLoader."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ProductDatasetLoader()
    return _loader_instance


if __name__ == "__main__":
    loader = get_dataset_loader()
    df = loader.load()
    print("Catalog loaded successfully!")
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nSample Semantic Representation for first item:")
    print(df.iloc[0]["semantic_representation"])
