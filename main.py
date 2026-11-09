"""Main execution pipeline for JobLens Vietnam Data Science Project."""
import sys
import os
import json
import numpy as np
import pandas as pd

import src.config as config
from src.data.loader import load_raw_datasets, standardize_and_merge, deduplicate_merged
from src.data.cleaner import clean_dataframe, generate_location_audit
from src.data.salary_parser import process_salaries
from src.data.label_generator import generate_labels
from src.features.text_processor import prepare_classification_regression_text, prepare_clustering_text
from src.features.sbert_encoder import encode_texts, save_embedding_metadata
from src.models.classification import run_classification_experiments
from src.models.regression import run_regression_experiments
from src.models.clustering import run_clustering_experiments

def main():
    print("=" * 75)
    print(" JobLens Vietnam: Data Science Pipeline Execution ")
    print(" Comprehensive Bug-Fixed & Improved Academic Experimental Suite ")
    print("=" * 75)
    
    # -------------------------------------------------------------
    # STEP 1: LOAD & INTEGRATE DATASETS
    # -------------------------------------------------------------
    print("\n>>> STEP 1: Loading & Integrating Datasets...")
    df_tinixai, df_topcv = load_raw_datasets()
    print(f"Loaded raw datasets: TinixAI ({df_tinixai.shape[0]} rows), TopCV ({df_topcv.shape[0]} rows).")
    
    merged_raw = standardize_and_merge(df_tinixai, df_topcv)
    print(f"Merged raw schema: {merged_raw.shape[0]} total rows.")
    
    # Deduplicate with normalized composite keys and cross-source matching
    print("[Deduplication] Removing duplicate postings across and within sources...")
    merged_deduped, removed_dups, dedup_audit = deduplicate_merged(merged_raw)
    
    merged_path = config.PROCESSED_DATA_DIR / "merged_jobs.csv"
    merged_deduped.to_csv(merged_path, index=False, encoding='utf-8')
    print(f"Saved merged dataset: {len(merged_deduped)} records (removed {len(removed_dups)} duplicate postings).")
    
    # -------------------------------------------------------------
    # STEP 2: CLEANING & NORMALIZATION
    # -------------------------------------------------------------
    print("\n>>> STEP 2: Text Cleaning & Feature Normalization...")
    
    # Generate Location Fix Audit before and after cleaning
    print("[LocationAudit] Auditing location normalization fixes...")
    loc_audit_df = generate_location_audit(merged_deduped)
    
    cleaned_df = clean_dataframe(merged_deduped)
    
    # Salary Parsing with independent bounds and explicit status categories
    print("[SalaryParser] Parsing salary strings into VND/month with independent bounds...")
    parsed_salary_df, salary_cat_report, salary_audit = process_salaries(cleaned_df)
    
    # Rule-Based Job Role Label Generation (Weak Supervision)
    print("[LabelGenerator] Generating rule-based job role labels (weak supervision)...")
    labeled_df, label_dist, label_mapping = generate_labels(parsed_salary_df)
    
    cleaned_path = config.PROCESSED_DATA_DIR / "cleaned_jobs.csv"
    labeled_df.to_csv(cleaned_path, index=False, encoding='utf-8')
    print(f"Cleaned master dataset saved to {cleaned_path}")
    
    # -------------------------------------------------------------
    # STEP 3: PREPARE TASK-SPECIFIC DATASETS
    # -------------------------------------------------------------
    print("\n>>> STEP 3: Preparing Sub-datasets for ML Tasks...")
    
    # 3.1 Classification Dataset: valid assigned roles + non-empty descriptions
    cls_mask = (labeled_df['job_role_label'] != 'Other / Unassigned') & \
               (labeled_df['job_description'].str.strip() != '')
    df_classification = labeled_df[cls_mask].copy().reset_index(drop=True)
    cls_path = config.PROCESSED_DATA_DIR / "classification_data.csv"
    df_classification.to_csv(cls_path, index=False, encoding='utf-8')
    print(f"Classification dataset: {len(df_classification)} records across {df_classification['job_role_label'].nunique()} classes.")
    
    # 3.2 Regression Dataset: valid numerical salary target + non-empty descriptions
    reg_mask = (labeled_df['salary_parse_status'] == 'valid') & \
               (labeled_df['job_description'].str.strip() != '')
    df_regression = labeled_df[reg_mask].copy().reset_index(drop=True)
    reg_path = config.PROCESSED_DATA_DIR / "regression_data.csv"
    df_regression.to_csv(reg_path, index=False, encoding='utf-8')
    print(f"Salary regression dataset: {len(df_regression)} records with valid numerical salaries.")
    
    # 3.3 Clustering Dataset: non-empty composite text
    composite_texts, cluster_ids = prepare_clustering_text(labeled_df)
    labeled_df['composite_text'] = composite_texts
    cluster_mask = labeled_df['composite_text'].str.strip() != ''
    df_clustering = labeled_df[cluster_mask].copy().reset_index(drop=True)
    cluster_path = config.PROCESSED_DATA_DIR / "clustering_data.csv"
    df_clustering.to_csv(cluster_path, index=False, encoding='utf-8')
    print(f"Clustering dataset: {len(df_clustering)} records.")
    
    # -------------------------------------------------------------
    # STEP 4: SBERT EMBEDDING GENERATION
    # -------------------------------------------------------------
    print("\n>>> STEP 4: Generating / Validating SBERT Multilingual Embeddings...")
    
    # SBERT for Master / Classification / Regression (job_description text)
    master_texts, master_ids = prepare_classification_regression_text(labeled_df)
    sbert_cache = config.EMBEDDINGS_DIR / "sbert_embeddings.npy"
    master_embeddings, _ = encode_texts(master_texts, master_ids, sbert_cache, batch_size=config.SBERT_BATCH_SIZE)
    
    # Index alignment mapping using unique_job_id
    id_to_idx = {uid: idx for idx, uid in enumerate(master_ids)}
    
    cls_indices = [id_to_idx[uid] for uid in df_classification['unique_job_id']]
    cls_embeddings = master_embeddings[cls_indices]
    
    reg_indices = [id_to_idx[uid] for uid in df_regression['unique_job_id']]
    reg_embeddings = master_embeddings[reg_indices]
    
    # SBERT for Clustering (composite text: title + desc + reqs)
    cluster_texts_sub, cluster_ids_sub = prepare_clustering_text(df_clustering)
    cluster_sbert_cache = config.EMBEDDINGS_DIR / "sbert_clustering_embeddings.npy"
    cluster_embeddings, _ = encode_texts(cluster_texts_sub, cluster_ids_sub, cluster_sbert_cache, batch_size=config.SBERT_BATCH_SIZE)
    
    # Save Metadata
    save_embedding_metadata(
        classification_ids=df_classification['unique_job_id'].tolist(),
        clustering_ids=cluster_ids_sub,
        cls_shape=cls_embeddings.shape,
        cluster_shape=cluster_embeddings.shape
    )
    
    # -------------------------------------------------------------
    # STEP 5: TASK 1 - JOB ROLE CLASSIFICATION
    # -------------------------------------------------------------
    print("\n>>> STEP 5: Executing Task 1 - Job Role Classification (80/10/10 Stratified Split)...")
    cls_results, best_cls_model, cls_meta, cls_split_manifest = run_classification_experiments(
        cls_embeddings,
        df_classification['job_role_label'],
        df_classification['job_description'].tolist(),
        df_classification['unique_job_id'].tolist()
    )
    print("\nClassification Comparison Table (Test Set with Val Selection):")
    print(cls_results.to_string(index=False))
    
    # -------------------------------------------------------------
    # STEP 6: TASK 2 - SALARY PREDICTION (REGRESSION)
    # -------------------------------------------------------------
    print("\n>>> STEP 6: Executing Task 2 - Salary Regression, Reconciled Ablation & Temporal Validation...")
    reg_results, ablation_results, temporal_results, best_reg_model, reg_meta, reg_split_manifest = run_regression_experiments(
        df_regression, reg_embeddings
    )
    print("\nRegression Model Comparison on Test Set (Full Feature E4):")
    print(reg_results.to_string(index=False))
    print("\nReconciled Ablation Study Summary (E1, E2, E3, E4):")
    print(ablation_results.to_string(index=False))
    if not temporal_results.empty:
        print("\nTemporal Validation (Train <=2025, Test 2026):")
        print(temporal_results.to_string(index=False))
        
    # -------------------------------------------------------------
    # STEP 7: TASK 3 - JOB PROFILE CLUSTERING (UNSUPERVISED)
    # -------------------------------------------------------------
    print("\n>>> STEP 7: Executing Task 3 - Unsupervised Job Clustering & Fair Benchmark...")
    clustering_results, assignments, best_km_model, cluster_meta = run_clustering_experiments(
        df_clustering, cluster_embeddings, cluster_texts_sub
    )
    print("\nClustering Performance Benchmark Table:")
    print(clustering_results.to_string(index=False))
    
    # -------------------------------------------------------------
    # STEP 8: SAVE CONSOLIDATED SPLIT MANIFEST
    # -------------------------------------------------------------
    print("\n>>> STEP 8: Saving Consolidated Split Manifest...")
    consolidated_manifest = pd.concat([cls_split_manifest, reg_split_manifest], ignore_index=True)
    manifest_path = config.OUTPUTS_DIR / "split_manifest.csv"
    consolidated_manifest.to_csv(manifest_path, index=False)
    
    manifest_summary = consolidated_manifest.groupby(['task', 'split']).size().reset_index(name='count')
    summary_path = config.REPORTS_DIR / "tables" / "split_manifest_summary.csv"
    manifest_summary.to_csv(summary_path, index=False)
    print(f"Split manifest saved to {manifest_path} (Summary saved to {summary_path})")
    
    print("\n" + "=" * 75)
    print(" ALL PIPELINE STEPS COMPLETED AND RESULTS OVERWRITTEN SUCCESSFULLY! ")
    print("=" * 75)

if __name__ == "__main__":
    main()
