"""
SmartShop AI - Quantitative Recommendation Evaluation & Ablation Suite
Implements IR and Recommender Systems ranking metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Budget Compliance Rate (%)
Also conducts an Ablation Study comparing Lexical Keyword Matching vs Pure Semantic Search vs Hybrid Ranking.
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

from evaluation.benchmark_queries import BENCHMARK_QUERIES
from recommendation.ranker import HybridRecommendationEngine, get_recommendation_engine
from data.dataset_loader import get_dataset_loader


class RecommendationEvaluator:
    """Evaluation harness for calculating IR metrics and running ablation studies."""

    def __init__(self, engine: Optional[HybridRecommendationEngine] = None):
        self.engine = engine or get_recommendation_engine()
        self.catalog_df = get_dataset_loader().load()

    # =========================================================================
    # CORE METRIC CALCULATIONS
    # =========================================================================

    @staticmethod
    def precision_at_k(
        recommended_ids: List[str],
        ground_truth: Dict[str, int],
        k: int = 5,
        threshold: int = 2
    ) -> float:
        """Precision@K: Fraction of top-K recommendations that are relevant (rel >= threshold)."""
        if not recommended_ids or k <= 0:
            return 0.0
        top_k = recommended_ids[:k]
        relevant_hits = sum(1 for pid in top_k if ground_truth.get(pid, 0) >= threshold)
        return relevant_hits / k

    @staticmethod
    def recall_at_k(
        recommended_ids: List[str],
        ground_truth: Dict[str, int],
        k: int = 5,
        threshold: int = 2
    ) -> float:
        """Recall@K: Fraction of all relevant ground-truth products retrieved in top-K."""
        total_relevant = sum(1 for rel in ground_truth.values() if rel >= threshold)
        if total_relevant == 0:
            return 1.0
        top_k = recommended_ids[:k]
        relevant_hits = sum(1 for pid in top_k if ground_truth.get(pid, 0) >= threshold)
        return relevant_hits / total_relevant

    @staticmethod
    def reciprocal_rank(
        recommended_ids: List[str],
        ground_truth: Dict[str, int],
        threshold: int = 2
    ) -> float:
        """Reciprocal Rank (1/rank of the first relevant recommendation)."""
        for rank, pid in enumerate(recommended_ids, 1):
            if ground_truth.get(pid, 0) >= threshold:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(
        recommended_ids: List[str],
        ground_truth: Dict[str, int],
        k: int = 5
    ) -> float:
        """
        Normalized Discounted Cumulative Gain (NDCG@K) with graded relevance.
        DCG@K = sum((2^rel - 1) / log2(rank + 1))
        NDCG@K = DCG@K / IDCG@K
        """
        if not recommended_ids or k <= 0:
            return 0.0

        top_k = recommended_ids[:k]

        # Compute DCG@K
        dcg = 0.0
        for rank, pid in enumerate(top_k, 1):
            rel = ground_truth.get(pid, 0)
            dcg += (math.pow(2, rel) - 1.0) / math.log2(rank + 1)

        # Compute Ideal DCG@K (IDCG)
        sorted_ideal_rels = sorted(ground_truth.values(), reverse=True)[:k]
        idcg = 0.0
        for rank, rel in enumerate(sorted_ideal_rels, 1):
            idcg += (math.pow(2, rel) - 1.0) / math.log2(rank + 1)

        if idcg <= 0.0:
            return 1.0 if dcg == 0.0 else 0.0

        return dcg / idcg

    @staticmethod
    def budget_compliance_rate(
        recommended_items: List[Dict[str, Any]],
        budget_max: Optional[int],
        tolerance_pct: float = 0.03
    ) -> float:
        """Computes fraction of recommendations strictly respecting the user's budget."""
        if budget_max is None or not recommended_items:
            return 1.0

        max_allowed = budget_max * (1.0 + tolerance_pct)
        compliant = sum(1 for item in recommended_items if item.get("price", 0) <= max_allowed)
        return compliant / len(recommended_items)

    # =========================================================================
    # ABLATION STUDY & MODEL COMPARISON
    # =========================================================================

    def run_ablation_study(self, k: int = 5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compares 3 distinct recommendation paradigms across the benchmark suite:
        1. Pure Lexical Keyword Match (TF-IDF / Jaccard token baseline)
        2. Pure Dense Semantic Search (all-MiniLM-L6-v2 without budget/rating filters)
        3. SmartShop AI Multi-Factor Hybrid Ranking (Proposed architecture)
        """
        systems = [
            ("Lexical Keyword Search", {"method": "lexical"}),
            ("Pure Semantic Retrieval", {"method": "semantic_only"}),
            ("SmartShop AI (Hybrid)", {"method": "hybrid"}),
        ]

        summary_records = []
        detailed_query_results = []

        for system_name, config in systems:
            p_at_k_scores = []
            r_at_k_scores = []
            mrr_scores = []
            ndcg_scores = []
            budget_comp_scores = []

            for bq in BENCHMARK_QUERIES:
                q_text = bq["query"]
                gt = bq["ground_truth_relevance"]
                budget_max = bq["budget_max"]

                if config["method"] == "lexical":
                    # Simple lexical token match baseline
                    recs = self._run_lexical_baseline(q_text, top_k=k)
                elif config["method"] == "semantic_only":
                    # Weights: 100% semantic, 0% budget, 0% rating, 0% feature
                    recs_res, _ = self.engine.recommend(
                        q_text,
                        top_k=k,
                        weights={"semantic": 1.0, "budget": 0.0, "rating": 0.0, "feature": 0.0},
                        strict_budget=False,
                        strict_category=False,
                    )
                    recs = [r.product for r in recs_res]
                else:  # hybrid
                    recs_res, _ = self.engine.recommend(
                        q_text,
                        top_k=k,
                        weights=None,  # default balanced weights
                        strict_budget=False,
                        strict_category=True,
                    )
                    recs = [r.product for r in recs_res]

                rec_ids = [r["product_id"] for r in recs]

                p = self.precision_at_k(rec_ids, gt, k=k)
                r = self.recall_at_k(rec_ids, gt, k=k)
                mrr = self.reciprocal_rank(rec_ids, gt)
                ndcg = self.ndcg_at_k(rec_ids, gt, k=k)
                bcomp = self.budget_compliance_rate(recs, budget_max)

                p_at_k_scores.append(p)
                r_at_k_scores.append(r)
                mrr_scores.append(mrr)
                ndcg_scores.append(ndcg)
                budget_comp_scores.append(bcomp)

                detailed_query_results.append({
                    "system": system_name,
                    "query_id": bq["query_id"],
                    "query": q_text,
                    "precision_at_k": round(p, 3),
                    "recall_at_k": round(r, 3),
                    "mrr": round(mrr, 3),
                    "ndcg_at_k": round(ndcg, 3),
                    "budget_compliance": round(bcomp, 3),
                })

            summary_records.append({
                "System Architecture": system_name,
                f"Precision@{k}": round(float(np.mean(p_at_k_scores)), 3),
                f"Recall@{k}": round(float(np.mean(r_at_k_scores)), 3),
                "MRR": round(float(np.mean(mrr_scores)), 3),
                f"NDCG@{k}": round(float(np.mean(ndcg_scores)), 3),
                "Budget Compliance (%)": f"{round(float(np.mean(budget_comp_scores)) * 100, 1)}%",
            })

        summary_df = pd.DataFrame(summary_records)
        return summary_df, {"detailed": detailed_query_results}

    def _run_lexical_baseline(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Computes basic keyword overlap (Jaccard similarity on tokens)."""
        q_tokens = set(query.lower().split())
        scored_items = []

        for _, row in self.catalog_df.iterrows():
            text_tokens = set(str(row["semantic_representation"]).lower().split())
            intersection = q_tokens.intersection(text_tokens)
            union = q_tokens.union(text_tokens)
            jaccard = len(intersection) / max(1, len(union))
            scored_items.append((jaccard, row.to_dict()))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_items[:top_k]]


# Global Singleton
_evaluator_instance: Optional[RecommendationEvaluator] = None


def get_recommendation_evaluator() -> RecommendationEvaluator:
    """Returns singleton instance of RecommendationEvaluator."""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = RecommendationEvaluator()
    return _evaluator_instance


if __name__ == "__main__":
    evaluator = get_recommendation_evaluator()
    print("=" * 80)
    print("Executing Recommendation Evaluation & Ablation Study on Benchmark Suite:")
    print("=" * 80)
    summary_table, details = evaluator.run_ablation_study(k=5)
    print("\nSummary Ablation Table (Placement Presentation Ready):")
    print(summary_table.to_string(index=False))
