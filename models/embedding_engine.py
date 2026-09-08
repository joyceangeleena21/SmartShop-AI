"""
SmartShop AI - Semantic Embedding Engine
Generates, caches, and compares dense vector embeddings using Sentence Transformers.
Uses pretrained all-MiniLM-L6-v2 (384-dimensional dense vectors) with local disk caching
for sub-millisecond retrieval on subsequent runs.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from config.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    EMBEDDINGS_CACHE_PATH,
    BATCH_SIZE,
)
from data.dataset_loader import ProductDatasetLoader, get_dataset_loader


class EmbeddingEngine:
    """Manages model loading, dense vector encoding, embedding disk caching, and similarity computation."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        cache_path: Path = EMBEDDINGS_CACHE_PATH,
    ):
        self.model_name = model_name
        self.cache_path = Path(cache_path)
        self._model: Optional[SentenceTransformer] = None
        self._product_embeddings: Optional[np.ndarray] = None
        self._cached_product_ids: List[str] = []

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-loads the SentenceTransformer model on demand."""
        if self._model is None:
            print(f"Loading SentenceTransformer model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """Encodes a single text or list of texts into normalized embedding vectors."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes and normalizes a single user query into a 1D vector of size (384,)."""
        clean_query = query.strip()
        if not clean_query:
            # Fallback zero vector
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        emb = self.encode_text(clean_query, normalize=True, show_progress=False)
        return emb[0].astype(np.float32)

    def get_or_create_product_embeddings(
        self,
        df: pd.DataFrame,
        force_recompute: bool = False
    ) -> np.ndarray:
        """Loads cached embeddings if available and valid; otherwise computes and caches them."""
        current_ids = df["product_id"].tolist()

        # Check if cache is valid (exists, not forced, and dimensions match catalog rows)
        if not force_recompute and self.cache_path.exists():
            try:
                cached_embs = np.load(self.cache_path)
                if cached_embs.shape[0] == len(df) and cached_embs.shape[1] == EMBEDDING_DIM:
                    self._product_embeddings = cached_embs
                    self._cached_product_ids = current_ids
                    return self._product_embeddings
            except Exception as e:
                print(f"Warning: Failed to load cached embeddings ({e}). Recomputing...")

        # Compute embeddings
        print(f"Computing semantic embeddings for {len(df)} products...")
        text_corpus = df["semantic_representation"].tolist()
        embeddings = self.encode_text(text_corpus, normalize=True, show_progress=True)
        embeddings = embeddings.astype(np.float32)

        # Save to disk cache
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_path, embeddings)
        print(f"Saved product embeddings cache to {self.cache_path} (Shape: {embeddings.shape})")

        self._product_embeddings = embeddings
        self._cached_product_ids = current_ids
        return self._product_embeddings

    def compute_similarity(
        self,
        query_vector: np.ndarray,
        product_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Computes cosine similarity between a normalized query vector (384,)
        and a matrix of normalized product vectors (N, 384).
        Since vectors are L2-normalized: cos_sim = query · product.
        Clamped to [0.0, 1.0] range for intuitive percentage conversion.
        """
        if query_vector.ndim == 1:
            # Dot product (N, 384) x (384,) -> (N,)
            sims = np.dot(product_vectors, query_vector)
        else:
            sims = np.dot(product_vectors, query_vector.T).flatten()

        # Clamp from [-1, 1] to [0, 1]
        sims = np.clip(sims, 0.0, 1.0)
        return sims.astype(np.float32)

    def semantic_search(
        self,
        query: str,
        df: pd.DataFrame,
        top_k: int = 5
    ) -> pd.DataFrame:
        """Pure semantic search baseline returning top-K products sorted solely by cosine similarity."""
        prod_embeddings = self.get_or_create_product_embeddings(df)
        q_vec = self.encode_query(query)
        sim_scores = self.compute_similarity(q_vec, prod_embeddings)

        result_df = df.copy()
        result_df["semantic_score"] = sim_scores
        result_df = result_df.sort_values(by="semantic_score", ascending=False).head(top_k)
        return result_df


# Global Singleton
_engine_instance: Optional[EmbeddingEngine] = None


def get_embedding_engine() -> EmbeddingEngine:
    """Returns singleton instance of EmbeddingEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = EmbeddingEngine()
    return _engine_instance


if __name__ == "__main__":
    loader = get_dataset_loader()
    df_catalog = loader.load()

    engine = get_embedding_engine()
    print("Testing EmbeddingEngine on sample query...")
    sample_query = "I need a laptop under 70000 for Python, machine learning and college work."
    results = engine.semantic_search(sample_query, df_catalog, top_k=3)
    print("\nTop 3 Semantic Matches:")
    for _, row in results.iterrows():
        print(f"[{row['product_id']}] {row['title'][:70]}... | Price: {row['price_formatted']} | Sim: {row['semantic_score']:.4f}")
