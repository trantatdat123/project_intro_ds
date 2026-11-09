"""Data loading and integration module for JobLens Vietnam."""
import re
import hashlib
from typing import Tuple, Dict, Any, List
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.data.cleaner import clean_text, normalize_location

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

def _normalize_company_for_dedup(name: Any) -> str:
    """Normalize company name by stripping common legal and industry form prefixes/suffixes."""
    if not name or pd.isna(name):
        return ""
    s = clean_text(name).lower()
    s = re.sub(r'\b(công ty cổ phần|công ty tnhh|công ty|cty cp|cty tnhh|cp|tnhh|jsc|ltd|corp|corporation|group|công nghệ|technology|phần mềm|software)\b', ' ', s)
    s = re.sub(r'[^a-z0-9\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def _normalize_title_for_dedup(title: Any) -> str:
    """Normalize job title by stripping bracketed location tags."""
    if not title or pd.isna(title):
        return ""
    s = clean_text(title).lower()
    s = re.sub(r'\[.*?\]|\(.*?\)', ' ', s)
    s = re.sub(r'[^a-z0-9\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def deduplicate_merged(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Check and remove duplicates within and across sources using normalized composite keys.
    Saves deduplication audit report to reports/tables/deduplication_audit.csv.
    Returns (deduped_df, removed_records_df, audit_summary_df).
    """
    df_clean = df.copy()
    
    # 1. Normalized composite keys
    norm_title = df_clean['job_title'].apply(_normalize_title_for_dedup)
    norm_comp = df_clean['company_name'].apply(_normalize_company_for_dedup)
    norm_desc = df_clean['job_description'].apply(clean_text).str.lower().str.strip()
    norm_desc_prefix = norm_desc.str.slice(0, 150)
    
    # Composite key: normalized title + normalized company + normalized description prefix
    composite_key = norm_title + "|||" + norm_comp + "|||" + norm_desc_prefix
    
    # A record is considered duplicate if the composite key is already seen (keeping first)
    # Filter empty keys
    is_valid_key = (norm_title != '') & (norm_desc_prefix.str.len() > 20)
    
    # Deduplication mask
    dup_mask = is_valid_key & composite_key.duplicated(keep='first')
    
    removed_records = df_clean[dup_mask].copy()
    deduped = df_clean[~dup_mask].reset_index(drop=True)
    
    # Statistics breakdown
    raw_tinixai = int((df_clean['source'] == 'tinixai').sum())
    raw_topcv = int((df_clean['source'] == 'topcv').sum())
    
    # Within TinixAI duplicates
    t_mask = df_clean['source'] == 'tinixai'
    tinixai_dups = int((is_valid_key & t_mask & composite_key[t_mask].duplicated(keep='first')).sum())
    
    # Within TopCV duplicates
    c_mask = df_clean['source'] == 'topcv'
    topcv_dups = int((is_valid_key & c_mask & composite_key[c_mask].duplicated(keep='first')).sum())
    
    # Cross-source duplicates: records in TopCV whose key was already present in TinixAI
    tinixai_keys = set(composite_key[t_mask & is_valid_key])
    cross_dups = int((is_valid_key & c_mask & composite_key.isin(tinixai_keys)).sum())
    
    total_removed = len(removed_records)
    remaining_total = len(deduped)
    remaining_tinixai = int((deduped['source'] == 'tinixai').sum())
    remaining_topcv = int((deduped['source'] == 'topcv').sum())
    
    audit_rows = [
        {"Metric": "Raw Records - TinixAI", "Count": raw_tinixai, "Percentage": round(raw_tinixai / len(df_clean) * 100, 2)},
        {"Metric": "Raw Records - TopCV", "Count": raw_topcv, "Percentage": round(raw_topcv / len(df_clean) * 100, 2)},
        {"Metric": "Raw Records - Total", "Count": len(df_clean), "Percentage": 100.0},
        {"Metric": "Duplicates Removed - Within TinixAI", "Count": tinixai_dups, "Percentage": round(tinixai_dups / len(df_clean) * 100, 2)},
        {"Metric": "Duplicates Removed - Within TopCV", "Count": topcv_dups, "Percentage": round(topcv_dups / len(df_clean) * 100, 2)},
        {"Metric": "Duplicates Removed - Cross-Source (TopCV matching TinixAI)", "Count": cross_dups, "Percentage": round(cross_dups / len(df_clean) * 100, 2)},
        {"Metric": "Total Duplicate Records Removed", "Count": total_removed, "Percentage": round(total_removed / len(df_clean) * 100, 2)},
        {"Metric": "Remaining Unique Records - TinixAI", "Count": remaining_tinixai, "Percentage": round(remaining_tinixai / remaining_total * 100, 2)},
        {"Metric": "Remaining Unique Records - TopCV", "Count": remaining_topcv, "Percentage": round(remaining_topcv / remaining_total * 100, 2)},
        {"Metric": "Remaining Unique Records - Total", "Count": remaining_total, "Percentage": round(remaining_total / len(df_clean) * 100, 2)}
    ]
    audit_df = pd.DataFrame(audit_rows)
    audit_path = REPORTS_DIR / "tables" / "deduplication_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(audit_path, index=False, encoding='utf-8')
    print(f"[Deduplication] Removed {total_removed} duplicate postings ({tinixai_dups} in TinixAI, {topcv_dups} in TopCV, {cross_dups} cross-source). Audit saved to {audit_path}")
    
    return deduped, removed_records, audit_df

def integrate_and_save() -> pd.DataFrame:
    """Execute full integration step and save merged_jobs.csv."""
    df_tinixai, df_topcv = load_raw_datasets()
    merged = standardize_and_merge(df_tinixai, df_topcv)
    deduped, removed, _ = deduplicate_merged(merged)
    out_path = PROCESSED_DATA_DIR / "merged_jobs.csv"
    deduped.to_csv(out_path, index=False, encoding='utf-8')
    return deduped
