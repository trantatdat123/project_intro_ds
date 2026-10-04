"""JobLens Vietnam: TinixAI preprocessing, features and models."""

import os
from pathlib import Path

# Numba phải có cache ghi được trước khi SHAP/UMAP import nó.
if not os.environ.get("NUMBA_CACHE_DIR"):
    _cache = Path(__file__).resolve().parents[1] / "artifacts/numba_cache"
    _cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(_cache)
