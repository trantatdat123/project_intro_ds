"""Text processing for SBERT embeddings."""
import pandas as pd
from typing import Tuple, List

def prepare_classification_regression_text(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Prepare text for classification and regression tasks.
    Input must be job_description only (to avoid target leakage from job_title).
    Returns (texts, unique_job_ids).
    """
    texts = df['job_description'].fillna('').astype(str).tolist()
    ids = df['unique_job_id'].astype(str).tolist()
    return texts, ids

def prepare_clustering_text(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Prepare composite text for unsupervised clustering:
    job_title + job_description + requirements
    Returns (composite_texts, unique_job_ids).
    """
    titles = df['job_title'].fillna('').astype(str)
    descs = df['job_description'].fillna('').astype(str)
    reqs = df['requirements'].fillna('').astype(str)
    
    composite = (titles + ". " + descs + " Yêu cầu: " + reqs).str.strip().tolist()
    ids = df['unique_job_id'].astype(str).tolist()
    return composite, ids

