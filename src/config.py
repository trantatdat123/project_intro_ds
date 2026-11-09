"""Configuration settings for JobLens Vietnam project."""
from pathlib import Path
import os

# Prevent TensorFlow import conflicts with transformers/sentence-transformers
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Directory Structure
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"

CLASSIFICATION_OUTPUTS = OUTPUTS_DIR / "classification"
REGRESSION_OUTPUTS = OUTPUTS_DIR / "regression"
CLUSTERING_OUTPUTS = OUTPUTS_DIR / "clustering"

# Model Hyperparameters & Global Constants
RANDOM_STATE = 42
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SBERT_BATCH_SIZE = 64
SBERT_EMBEDDING_DIM = 384

# Currency conversion rate (reference date: early 2026, 1 USD ~ 25,000 VND)
USD_TO_VND_RATE = 25000

# Minimum and maximum plausible monthly salaries in VND
MIN_PLAUSIBLE_MONTHLY_SALARY = 3_000_000      # 3 million VND
MAX_PLAUSIBLE_MONTHLY_SALARY = 150_000_000    # 150 million VND

# Ensure required directories exist
for p in [PROCESSED_DATA_DIR, EMBEDDINGS_DIR, REPORTS_DIR / "figures", REPORTS_DIR / "tables",
          CLASSIFICATION_OUTPUTS / "metrics", CLASSIFICATION_OUTPUTS / "plots", CLASSIFICATION_OUTPUTS / "models",
          REGRESSION_OUTPUTS / "metrics", REGRESSION_OUTPUTS / "plots", REGRESSION_OUTPUTS / "models",
          CLUSTERING_OUTPUTS / "metrics", CLUSTERING_OUTPUTS / "plots", CLUSTERING_OUTPUTS / "models"]:
    p.mkdir(parents=True, exist_ok=True)
