"""
SmartShop AI - Multi-Factor Hybrid Recommendation & Ranking Engine
Combines Dense Semantic Relevance, Budget Compatibility, Rating Credibility,
and Technical Feature Alignment into a mathematically principled ranking score.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    DEFAULT_WEIGHTS,
    BUDGET_OVER_PENALTY_RATE,
    BUDGET_UNDER_DISCOUNT_RATE,
    DEFAULT_TOP_K,
)
from models.embedding_engine import EmbeddingEngine, get_embedding_engine
from models.intent_extractor import IntentExtractor, UserIntent, get_intent_extractor
from data.dataset_loader import ProductDatasetLoader, get_dataset_loader


class RecommendationResult:
    """Encapsulates a single recommended product with comprehensive scoring diagnostics."""

    def __init__(
        self,
        product: Dict[str, Any],
        composite_score: float,
        semantic_score: float,
        budget_score: float,
        rating_score: float,
        feature_score: float,
        matched_specs: List[str],
        matched_use_cases: List[str],
        budget_status: str,
    ):
        self.product = product
        self.composite_score = round(composite_score, 4)
        self.semantic_score = round(semantic_score, 4)
        self.budget_score = round(budget_score, 4)
        self.rating_score = round(rating_score, 4)
        self.feature_score = round(feature_score, 4)
        self.matched_specs = matched_specs
        self.matched_use_cases = matched_use_cases
        self.budget_status = budget_status

    def to_dict(self) -> Dict[str, Any]:
        """Returns flat dictionary with product attributes and score diagnostics."""
        item = dict(self.product)
        item.update({
            "composite_score": self.composite_score,
            "semantic_score": self.semantic_score,
            "budget_score": self.budget_score,
            "rating_score": self.rating_score,
            "feature_score": self.feature_score,
            "matched_specs": self.matched_specs,
            "matched_use_cases": self.matched_use_cases,
            "budget_status": self.budget_status,
        })
        return item


class HybridRecommendationEngine:
    """Research-grade multi-factor recommendation engine."""

    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        intent_extractor: Optional[IntentExtractor] = None,
        dataset_loader: Optional[ProductDatasetLoader] = None,
    ):
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.intent_extractor = intent_extractor or get_intent_extractor()
        self.dataset_loader = dataset_loader or get_dataset_loader()

    def recommend(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        weights: Optional[Dict[str, float]] = None,
        strict_budget: bool = False,
        strict_category: bool = True,
        custom_intent: Optional[UserIntent] = None,
    ) -> Tuple[List[RecommendationResult], UserIntent]:
        """
        Executes the end-to-end hybrid recommendation pipeline:
        1. Query intent extraction
        2. Candidate pre-filtering (category, hard budget if active)
        3. Dense semantic embedding & cosine similarity
        4. Multi-factor scoring (Semantic + Budget + Rating + Feature)
        5. Composite ranking and Top-N selection
        """
        # Step 1: Intent Extraction
        intent = custom_intent if custom_intent else self.intent_extractor.extract_intent(query)

        # Step 2: Normalize Weights
        w = self._normalize_weights(weights or DEFAULT_WEIGHTS)

        # Step 3: Load Catalog & Filter Candidates
        catalog_df = self.dataset_loader.load()
        candidates_df = catalog_df.copy()

        # Category Filtering: If user explicitly mentioned a category, focus on it
        if strict_category and intent.category:
            cat_candidates = candidates_df[
                candidates_df["category"].str.lower() == intent.category.lower()
            ]
            if not cat_candidates.empty:
                candidates_df = cat_candidates

        # Strict Budget Pre-filtering (if enabled by user toggle)
        if strict_budget and intent.budget_max:
            # Allow a tiny 3% tolerance for close deals
            budget_candidates = candidates_df[
                candidates_df["price"] <= int(intent.budget_max * 1.03)
            ]
            if not budget_candidates.empty:
                candidates_df = budget_candidates

        if candidates_df.empty:
            return [], intent

        # Step 4: Semantic Similarity
        # Encode query
        query_vector = self.embedding_engine.encode_query(query)
        # Fetch precomputed catalog vectors
        all_embeddings = self.embedding_engine.get_or_create_product_embeddings(catalog_df)
        # Slice embeddings corresponding to candidate indices
        candidate_indices = candidates_df.index.to_numpy()
        candidate_embeddings = all_embeddings[candidate_indices]
        semantic_sims = self.embedding_engine.compute_similarity(query_vector, candidate_embeddings)

        # Step 5: Multi-factor Component Scoring
        results: List[RecommendationResult] = []

        for idx, (original_idx, row) in enumerate(candidates_df.iterrows()):
            sem_score = float(semantic_sims[idx])

            # Component 1: Budget Compatibility Score
            bud_score, bud_status = self._compute_budget_score(
                price=int(row["price"]),
                budget_max=intent.budget_max,
                budget_min=intent.budget_min,
            )

            # Component 2: Rating Credibility Score (Bayesian dampened)
            rat_score = self._compute_rating_score(
                rating=float(row["rating"]),
                reviews=int(row["review_count"]),
            )

            # Component 3: Feature & Persona Alignment Score
            feat_score, matched_specs, matched_ucs = self._compute_feature_score(
                row=row,
                intent=intent,
            )

            # Composite Weighted Sum:
            # S_total = w_sem * S_sem + w_bud * S_bud + w_rat * S_rat + w_feat * S_feat
            composite = (
                w["semantic"] * sem_score
                + w["budget"] * bud_score
                + w["rating"] * rat_score
                + w["feature"] * feat_score
            )

            rec_item = RecommendationResult(
                product=row.to_dict(),
                composite_score=composite,
                semantic_score=sem_score,
                budget_score=bud_score,
                rating_score=rat_score,
                feature_score=feat_score,
                matched_specs=matched_specs,
                matched_use_cases=matched_ucs,
                budget_status=bud_status,
            )
            results.append(rec_item)

        # Step 6: Rank by Composite Score descending
        results.sort(key=lambda x: x.composite_score, reverse=True)

        return results[:top_k], intent

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Ensures weights are positive and sum exactly to 1.0."""
        keys = ["semantic", "budget", "rating", "feature"]
        w_dict = {k: max(0.0, float(weights.get(k, DEFAULT_WEIGHTS[k]))) for k in keys}
        total = sum(w_dict.values())
        if total <= 0.0:
            return DEFAULT_WEIGHTS.copy()
        return {k: round(v / total, 4) for k, v in w_dict.items()}

    def _compute_budget_score(
        self,
        price: int,
        budget_max: Optional[int],
        budget_min: Optional[int]
    ) -> Tuple[float, str]:
        """
        Mathematical budget compatibility function:
        - Within budget: High score with gentle discount for drastically cheaper items
        - Over budget: Smooth exponential penalty
        """
        if budget_max is None and budget_min is None:
            return 1.0, "Budget Open"

        if budget_max is not None:
            if price <= budget_max:
                # E.g. ₹68k / ₹70k -> delta=2k -> ratio=0.028 -> score = 1 - 0.20*(0.028) = 0.994
                under_ratio = (budget_max - price) / budget_max
                score = 1.0 - (BUDGET_UNDER_DISCOUNT_RATE * under_ratio)
                status = f"₹{budget_max - price:,} Under Budget" if budget_max > price else "Exact Budget Fit"
                return float(np.clip(score, 0.4, 1.0)), status
            else:
                # E.g. ₹75k / ₹70k -> over=5k -> ratio=0.071 -> exp(-3.5 * 0.071) = 0.78
                over_ratio = (price - budget_max) / budget_max
                penalty = math.exp(-BUDGET_OVER_PENALTY_RATE * over_ratio)
                status = f"₹{price - budget_max:,} Above Budget"
                return float(np.clip(penalty, 0.01, 0.95)), status

        if budget_min is not None and price < budget_min:
            under_ratio = (budget_min - price) / budget_min
            return float(np.clip(math.exp(-2.0 * under_ratio), 0.1, 1.0)), "Below Target Range"

        return 1.0, "Budget Compatible"

    def _compute_rating_score(self, rating: float, reviews: int) -> float:
        """
        Computes Bayesian-dampened credibility score:
        Combines star rating (1.0 to 5.0) normalized to [0, 1]
        with logarithmic confidence factor based on review volume.
        """
        # Baseline normalized rating: 1.0 -> 0.0, 5.0 -> 1.0
        normalized_rating = (rating - 1.0) / 4.0

        # Confidence weight: reaches 1.0 at 500+ reviews
        # reviews factor in [0.70, 1.0] so items with low reviews aren't killed but validated items get bonus
        confidence = 0.70 + 0.30 * min(1.0, math.log10(1 + reviews) / math.log10(500))

        return float(np.clip(normalized_rating * confidence, 0.0, 1.0))

    def _compute_feature_score(
        self,
        row: pd.Series,
        intent: UserIntent
    ) -> Tuple[float, List[str], List[str]]:
        """
        Computes spec and persona alignment score.
        Detects exact matches for RAM, GPU, Brand, and Use-case tags.
        """
        matched_specs = []
        matched_use_cases = []
        score_components = []

        # 1. Spec Checks
        if intent.specs:
            total_specs = len(intent.specs)
            spec_hits = 0

            # RAM match
            if "ram" in intent.specs:
                target_ram = intent.specs["ram"].upper()
                prod_ram = str(row.get("ram", "")).upper()
                if target_ram in prod_ram or ("16GB" in target_ram and "32GB" in prod_ram):
                    spec_hits += 1
                    matched_specs.append(f"RAM: {row.get('ram')}")

            # GPU match
            if "gpu" in intent.specs:
                target_gpu = intent.specs["gpu"].upper()
                prod_gpu = str(row.get("gpu", "")).upper()
                if target_gpu == "DEDICATED GPU" and ("RTX" in prod_gpu or "GEFORCE" in prod_gpu or "RADEON" in prod_gpu):
                    spec_hits += 1
                    matched_specs.append(f"GPU: {row.get('gpu')}")
                elif target_gpu in prod_gpu.replace(" ", ""):
                    spec_hits += 1
                    matched_specs.append(f"GPU: {row.get('gpu')}")

            # Storage match
            if "storage" in intent.specs:
                target_storage = intent.specs["storage"].upper()
                prod_storage = str(row.get("storage", "")).upper()
                if target_storage in prod_storage:
                    spec_hits += 1
                    matched_specs.append(f"Storage: {row.get('storage')}")

            # Display match
            if "display" in intent.specs:
                prod_display = str(row.get("display", "")).upper()
                if "OLED" in intent.specs["display"].upper() and "OLED" in prod_display:
                    spec_hits += 1
                    matched_specs.append("OLED Display")
                elif "120HZ" in prod_display or "144HZ" in prod_display:
                    spec_hits += 1
                    matched_specs.append("High Refresh Display (120Hz+)")

            # OS match
            if "os" in intent.specs:
                target_os = intent.specs["os"].upper()
                prod_os = str(row.get("os", "")).upper()
                if target_os in prod_os:
                    spec_hits += 1
                    matched_specs.append(f"OS: {row.get('os')}")

            score_components.append(spec_hits / max(1, total_specs))

        # 2. Brand Match
        if intent.brands:
            prod_brand = str(row.get("brand", "")).lower()
            if any(b.lower() == prod_brand for b in intent.brands):
                score_components.append(1.0)
                matched_specs.append(f"Brand: {row.get('brand')}")
            else:
                score_components.append(0.3)

        # 3. Use Case Tag Overlap
        if intent.use_cases:
            prod_use_cases = [
                u.strip().lower() for u in str(row.get("use_cases", "")).split(",")
            ]
            overlap = set(intent.use_cases).intersection(set(prod_use_cases))
            for m in overlap:
                matched_use_cases.append(m.replace("_", " ").title())

            jaccard = len(overlap) / len(set(intent.use_cases).union(set(prod_use_cases)))
            # Boost recall for non-empty overlap
            uc_score = 0.5 + 0.5 * (len(overlap) / len(intent.use_cases)) if overlap else 0.2
            score_components.append(uc_score)

        # Aggregate Feature Score
        if not score_components:
            return 0.75, matched_specs, matched_use_cases  # Neutral prior

        final_feature_score = float(np.mean(score_components))
        return float(np.clip(final_feature_score, 0.1, 1.0)), matched_specs, matched_use_cases


# Global Singleton
_ranker_instance: Optional[HybridRecommendationEngine] = None


def get_recommendation_engine() -> HybridRecommendationEngine:
    """Returns singleton instance of HybridRecommendationEngine."""
    global _ranker_instance
    if _ranker_instance is None:
        _ranker_instance = HybridRecommendationEngine()
    return _ranker_instance


if __name__ == "__main__":
    engine = get_recommendation_engine()
    test_query = "I need a laptop under 70000 for Python, machine learning and college work."
    print("=" * 80)
    print(f"Executing Hybrid Recommendation for: '{test_query}'")
    print("=" * 80)

    recommendations, extracted_intent = engine.recommend(test_query, top_k=5)
    print(f"Extracted Intent: {extracted_intent.summary()}")
    print("-" * 80)

    for i, r in enumerate(recommendations, 1):
        p = r.product
        print(f"\n#{i}. {p['title'][:65]}...")
        print(f"    Price: {p['price_formatted']} ({r.budget_status}) | Rating: {p['rating']}* ({p['review_count']} reviews)")
        print(f"    Scores -> Composite: {r.composite_score:.3f} | Semantic: {r.semantic_score:.3f} | Budget: {r.budget_score:.3f} | Rating: {r.rating_score:.3f} | Feature: {r.feature_score:.3f}")
        print(f"    Matched Specs: {r.matched_specs}")
        print(f"    Matched Use Cases: {r.matched_use_cases}")
