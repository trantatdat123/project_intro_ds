"""Data loading and integration module for JobLens Vietnam."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

COMMON_COLUMNS = [
    'unique_job_id',
    'source_id',
    'source',
    'job_title',
    'company_name',
    'salary',
    'location',
    'job_type',
    'job_industry',
    'experience_level',
    'education_level',
    'job_position',
    'job_description',
    'benefits',
    'requirements',
    'year'
]

def inspect_dataset(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Inspect dataset basic statistics."""
    info = {
        'name': name,
        'rows': len(df),
        'cols': df.shape[1],
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': int(df.duplicated().sum()),
        'unique_titles': df['job_title'].nunique() if 'job_title' in df.columns else 0,
        'year_distribution': df['year'].value_counts(dropna=False).to_dict() if 'year' in df.columns else {}
    }
    return info

def load_raw_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both raw datasets without modifying original files."""
    tinixai_path = RAW_DATA_DIR / 'tinixai_it_ai_data_jobs.csv'
    topcv_path = RAW_DATA_DIR / 'topcv_jobs.csv'
    
    if not tinixai_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {tinixai_path}")
    if not topcv_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {topcv_path}")
        
    df_tinixai = pd.read_csv(tinixai_path)
    df_topcv = pd.read_csv(topcv_path)
    return df_tinixai, df_topcv

def standardize_and_merge(df_tinixai: pd.DataFrame, df_topcv: pd.DataFrame) -> pd.DataFrame:
    """Standardize schemas, create unique IDs, preserve source IDs, and merge."""
    # Process TinixAI
    t_df = df_tinixai.copy()
    t_df['source'] = 'tinixai'
    t_df['source_id'] = t_df['id'].astype(str)
    t_df['unique_job_id'] = 'tinixai_' + t_df['source_id']
    
    # Process TopCV
    c_df = df_topcv.copy()
    c_df['source'] = 'topcv'
    c_df['source_id'] = c_df['id'].astype(str)
    c_df['unique_job_id'] = 'topcv_' + c_df['source_id']
    
    # Align to common columns
    for col in COMMON_COLUMNS:
        if col not in t_df.columns:
            t_df[col] = np.nan
        if col not in c_df.columns:
            c_df[col] = np.nan
            
    t_aligned = t_df[COMMON_COLUMNS].copy()
    c_aligned = c_df[COMMON_COLUMNS].copy()
    
    merged = pd.concat([t_aligned, c_aligned], ignore_index=True)
    return merged

def deduplicate_merged(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Check and remove duplicates within and across sources, preserving audit records."""
    # Deduplication key based on normalized title, company, and prefix of description
    df_clean = df.copy()
    norm_title = df_clean['job_title'].fillna('').astype(str).str.lower().str.strip()
    norm_comp = df_clean['company_name'].fillna('').astype(str).str.lower().str.strip()
    norm_desc = df_clean['job_description'].fillna('').astype(str).str.lower().str.slice(0, 100)
    
    dup_mask = (norm_title != '') & (norm_comp != '') & (norm_desc != '') & \
               df_clean.duplicated(subset=['job_title', 'company_name', 'location'], keep='first')
               
    removed_records = df_clean[dup_mask].copy()
    deduped = df_clean[~dup_mask].reset_index(drop=True)
    return deduped, removed_records

def integrate_and_save() -> pd.DataFrame:
    """Execute full integration step and save merged_jobs.csv."""
    df_tinixai, df_topcv = load_raw_datasets()
    merged = standardize_and_merge(df_tinixai, df_topcv)
    deduped, removed = deduplicate_merged(merged)
    
    output_path = PROCESSED_DATA_DIR / 'merged_jobs.csv'
    deduped.to_csv(output_path, index=False, encoding='utf-8')
    print(f"[Loader] Merged {len(merged)} records down to {len(deduped)} (removed {len(removed)} duplicates). Saved to {output_path}")
    return deduped

