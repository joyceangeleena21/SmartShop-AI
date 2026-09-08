import sys
from pathlib import Path

# Fix Windows console cp1252 encoding for unicode symbols (e.g. ₹)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
EVALUATION_DIR = BASE_DIR / "evaluation"

PRODUCTS_CSV_PATH = DATA_DIR / "products.csv"
EMBEDDINGS_CACHE_PATH = DATA_DIR / "product_embeddings.npy"
DATASET_METADATA_PATH = DATA_DIR / "metadata.json"

# Semantic Model Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 32

# Default Ranking Weights (Sum = 1.0)
DEFAULT_WEIGHTS = {
    "semantic": 0.45,
    "budget": 0.25,
    "rating": 0.15,
    "feature": 0.15,
}

# Budget Scoring Parameters
BUDGET_OVER_PENALTY_RATE = 3.5    # Steepness of exponential penalty for exceeding budget
BUDGET_UNDER_DISCOUNT_RATE = 0.20  # Mild preference for utilizing budget effectively

# Default Recommendation Settings
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

# Product Categories
VALID_CATEGORIES = [
    "Laptop",
    "Smartphone",
    "Headphones",
    "Tablet",
    "Smartwatch",
]

# Supported Currencies & Formatting
CURRENCY_SYMBOL = "₹"
CURRENCY_CODE = "INR"
