"""Sentence-BERT embedding encoder with caching and alignment tracking."""
import os
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.config import (
    SBERT_MODEL_NAME,
    SBERT_BATCH_SIZE,
    SBERT_EMBEDDING_DIM,
    EMBEDDINGS_DIR
)

def get_device() -> str:
    """Return 'cuda' if GPU is available else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_sbert_model(model_name: str = SBERT_MODEL_NAME) -> SentenceTransformer:
    """Load SentenceTransformer model on appropriate device."""
    device = get_device()
    print(f"[SBERT] Loading model '{model_name}' on device '{device}'...")
    model = SentenceTransformer(model_name, device=device)
    return model

def encode_texts(
    texts: List[str],
    unique_ids: List[str],
    cache_file: Path,
    batch_size: int = SBERT_BATCH_SIZE,
    force_recompute: bool = False
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode list of texts into normalized 384-dimensional SBERT embeddings.
    Caches to disk as .npy file and avoids recomputing if already present.
    """
    if cache_file.exists() and not force_recompute:
        print(f"[SBERT] Loading cached embeddings from {cache_file}...")
        embeddings = np.load(cache_file)
        if len(embeddings) == len(unique_ids):
            return embeddings, unique_ids
        else:
            print(f"[SBERT] Cache size mismatch ({len(embeddings)} vs {len(unique_ids)}). Recomputing...")

    model = load_sbert_model()
    # Replace empty or whitespace-only texts with a neutral placeholder
    safe_texts = [t if (t and len(t.strip()) > 0) else "Thông tin tuyển dụng" for t in texts]
    
    print(f"[SBERT] Encoding {len(safe_texts)} texts in batches of {batch_size}...")
    embeddings = model.encode(
        safe_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)
    
    # Save embeddings
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, embeddings)
    print(f"[SBERT] Saved embeddings with shape {embeddings.shape} to {cache_file}")
    
    return embeddings, unique_ids

def save_embedding_metadata(
    classification_ids: List[str],
    clustering_ids: List[str],
    cls_shape: Tuple[int, int],
    cluster_shape: Tuple[int, int]
) -> None:
    """Save metadata JSON tracking alignment and parameters."""
    metadata = {
        "model_name": SBERT_MODEL_NAME,
        "embedding_dim": SBERT_EMBEDDING_DIM,
        "normalized": True,
        "classification_regression_records": len(classification_ids),
        "classification_embedding_shape": list(cls_shape),
        "clustering_records": len(clustering_ids),
        "clustering_embedding_shape": list(cluster_shape),
        "sample_cls_ids": classification_ids[:5],
        "sample_clustering_ids": clustering_ids[:5]
    }
    
    metadata_path = EMBEDDINGS_DIR / "embedding_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[SBERT] Embedding metadata saved to {metadata_path}")

