"""
SmartShop AI - Latency & Performance Profiler
Profiles execution latency for each phase of the recommendation pipeline:
- Intent Extraction Latency
- Query Embedding Inference Latency
- Dense Similarity Dot-Product Latency
- Multi-factor Scoring & Ranking Latency
- Total Pipeline Latency
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommendation.ranker import HybridRecommendationEngine, get_recommendation_engine
from models.intent_extractor import IntentExtractor, get_intent_extractor
from models.embedding_engine import EmbeddingEngine, get_embedding_engine
from data.dataset_loader import get_dataset_loader


class LatencyProfiler:
    """Profiles real-time latency across recommendation sub-components."""

    def __init__(self):
        self.engine = get_recommendation_engine()
        self.extractor = get_intent_extractor()
        self.embedding_engine = get_embedding_engine()
        self.catalog_df = get_dataset_loader().load()

    def profile_query(self, query: str, runs: int = 5) -> Dict[str, Any]:
        """Runs multiple iterations of a query and records millisecond latency per stage."""
        intent_times = []
        embed_times = []
        sim_times = []
        rank_times = []
        total_times = []

        all_embeddings = self.embedding_engine.get_or_create_product_embeddings(self.catalog_df)

        for _ in range(runs):
            t0 = time.perf_counter()

            # 1. Intent Extraction
            t_int_0 = time.perf_counter()
            intent = self.extractor.extract_intent(query)
            t_int_1 = time.perf_counter()
            intent_times.append((t_int_1 - t_int_0) * 1000)

            # 2. Query Embedding
            t_emb_0 = time.perf_counter()
            q_vec = self.embedding_engine.encode_query(query)
            t_emb_1 = time.perf_counter()
            embed_times.append((t_emb_1 - t_emb_0) * 1000)

            # 3. Vector Similarity
            t_sim_0 = time.perf_counter()
            sims = self.embedding_engine.compute_similarity(q_vec, all_embeddings)
            t_sim_1 = time.perf_counter()
            sim_times.append((t_sim_1 - t_sim_0) * 1000)

            # 4. Full Pipeline Ranking
            t_rank_0 = time.perf_counter()
            recs, _ = self.engine.recommend(query, top_k=5)
            t_rank_1 = time.perf_counter()
            rank_times.append((t_rank_1 - t_rank_0) * 1000)

            t1 = time.perf_counter()
            total_times.append((t1 - t0) * 1000)

        return {
            "query": query,
            "runs": runs,
            "intent_extraction_ms": round(float(np.mean(intent_times)), 2),
            "embedding_inference_ms": round(float(np.mean(embed_times)), 2),
            "vector_similarity_ms": round(float(np.mean(sim_times)), 3),
            "end_to_end_ranking_ms": round(float(np.mean(rank_times)), 2),
            "total_latency_ms": round(float(np.mean(total_times)), 2),
        }

    def get_summary_dataframe(self, sample_queries: Optional[List[str]] = None) -> pd.DataFrame:
        """Profiles a representative sample of queries and returns a summary DataFrame."""
        queries = sample_queries or [
            "I need a laptop under 70000 for Python, machine learning and college work.",
            "Best smartphone under 25000 with 120Hz AMOLED display and good camera",
            "Sony or Bose headphones under 25k with noise cancellation for study",
        ]

        records = []
        for q in queries:
            stats = self.profile_query(q, runs=3)
            records.append({
                "Sample Query": q[:40] + "...",
                "Intent Extraction (ms)": stats["intent_extraction_ms"],
                "Model Embedding (ms)": stats["embedding_inference_ms"],
                "Vector Dot Product (ms)": stats["vector_similarity_ms"],
                "Total Recommendation (ms)": stats["end_to_end_ranking_ms"],
            })

        return pd.DataFrame(records)


# Global Singleton
_profiler_instance: Optional[LatencyProfiler] = None


def get_latency_profiler() -> LatencyProfiler:
    """Returns singleton instance of LatencyProfiler."""
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = LatencyProfiler()
    return _profiler_instance


if __name__ == "__main__":
    profiler = get_latency_profiler()
    print("=" * 70)
    print("Benchmarking Pipeline Latency (Placement Defense Profile):")
    print("=" * 70)
    df_latency = profiler.get_summary_dataframe()
    print(df_latency.to_string(index=False))
