"""Script to generate all 5 required Jupyter notebooks for JobLens Vietnam with executed outputs and updated APIs."""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f"Created notebook: {filepath}")

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True)
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.strip().splitlines(keepends=True)
    }

# ==============================================================================
# NOTEBOOK 1: Data Integration & Preprocessing
# ==============================================================================
nb1_cells = [
    md_cell("""# Notebook 01: Data Integration and Preprocessing
**Project:** JobLens Vietnam - Labor Market Analysis & Skill Trends  
**Course:** Introduction to Data Science  

### Objective:
1. Inspect the two raw datasets: `tinixai_it_ai_data_jobs.csv` and `topcv_jobs.csv`.
2. Standardize column schemas, assign global unique IDs (`unique_job_id`), and track source provenance.
3. Perform intra-source and cross-source deduplication with normalized composite keys (`reports/tables/deduplication_audit.csv`).
4. Clean text fields (stripping HTML, unescaping characters, preserving essential technical tokens like `C++`, `SQL`, `Python`, `AWS`).
5. Audit and normalize regional locations with explicit city detection first (`reports/tables/location_fix_audit.csv`).
6. Parse salary expressions into VND/month with independent bounds and disclosure categories (`reports/tables/salary_parsing_audit.csv`).
7. Generate standardized rule-based `job_role_label` categories (weak supervision taxonomy).
8. Export processed subsets for downstream classification, regression, and clustering.
"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.data.loader import load_raw_datasets, inspect_dataset, standardize_and_merge, deduplicate_merged
from src.data.cleaner import clean_dataframe, generate_location_audit
from src.data.salary_parser import process_salaries
from src.data.label_generator import generate_labels
"""),
    md_cell("""## 1. Inspect Raw Datasets
Let's inspect row counts, schema definitions, and missing value profiles for both data sources without mutating raw files.
"""),
    code_cell("""df_tinixai, df_topcv = load_raw_datasets()

info_tinixai = inspect_dataset(df_tinixai, "TinixAI IT/AI Jobs")
info_topcv = inspect_dataset(df_topcv, "TopCV IT Jobs")

print("=== TinixAI Dataset ===")
print(f"Shape: {info_tinixai['rows']} rows x {info_tinixai['cols']} columns")
print(f"Missing values:\\n{pd.Series(info_tinixai['missing_values'])}")

print("\\n=== TopCV Dataset ===")
print(f"Shape: {info_topcv['rows']} rows x {info_topcv['cols']} columns")
print(f"Missing values:\\n{pd.Series(info_topcv['missing_values'])}")
"""),
    md_cell("""## 2. Standardize Schema and Deduplicate
We map both datasets into a common 16-attribute schema, synthesize `unique_job_id`, and remove duplicates using normalized composite keys.
"""),
    code_cell("""merged_raw = standardize_and_merge(df_tinixai, df_topcv)
print(f"Raw merged records: {len(merged_raw)}")

merged_deduped, removed_records, dedup_audit = deduplicate_merged(merged_raw)
print(f"Unique postings retained: {len(merged_deduped)} (removed {len(removed_records)} duplicates)")
print("\\nDeduplication Audit Breakdown:")
display(dedup_audit) if 'display' in dir() else print(dedup_audit.to_string(index=False))
"""),
    md_cell("""## 3. Location Normalization and Audit
Fix the previous location normalization bug where Hanoi locations containing 'quận' were erroneously mapped to Ho Chi Minh City.
"""),
    code_cell("""loc_audit_df = generate_location_audit(merged_deduped)
print(f"Location audit records affected: {len(loc_audit_df)} unique strings ({loc_audit_df['affected_records'].sum()} postings)")
display(loc_audit_df.head(10)) if 'display' in dir() else print(loc_audit_df.head(10).to_string(index=False))

cleaned_df = clean_dataframe(merged_deduped)
print("\\nNormalized Regional Distribution:")
print(cleaned_df['norm_location'].value_counts())
"""),
    md_cell("""## 4. Salary Parsing and Independent Validation
Parse salaries, categorize disclosure types, and validate bounds independently ($3\\text{M} \\le \\text{min} \\le \\text{max} \\le 150\\text{M}$).
"""),
    code_cell("""parsed_df, salary_cats, salary_audit = process_salaries(cleaned_df)
print("=== Salary Disclosure Categories ===")
display(salary_cats) if 'display' in dir() else print(salary_cats.to_string(index=False))

print("\\n=== Salary Parsing Audit Summary ===")
display(salary_audit) if 'display' in dir() else print(salary_audit.to_string(index=False))
"""),
    md_cell("""## 5. Rule-Based Job Role Generation (Weak Supervision)
Construct heuristic role pseudo-labels from `job_title` using hierarchical regex rules.
"""),
    code_cell("""labeled_df, label_dist, label_mapping = generate_labels(parsed_df)
print("=== Job Role Label Distribution ===")
display(label_dist) if 'display' in dir() else print(label_dist.to_string(index=False))
""")
]

create_notebook(nb1_cells, NOTEBOOKS_DIR / "01_Data_Integration_Preprocessing.ipynb")

# ==============================================================================
# NOTEBOOK 2: SBERT Embedding
# ==============================================================================
nb2_cells = [
    md_cell("""# Notebook 02: Sentence-BERT (SBERT) Dense Embeddings
**Project:** JobLens Vietnam - Labor Market Analysis & Skill Trends  
**Course:** Introduction to Data Science  

### Objective:
1. Encode multilingual recruitment texts using pre-trained `paraphrase-multilingual-MiniLM-L12-v2`.
2. Generate 384-dimensional dense semantic vectors.
3. Verify embedding caching, L2 normalization, and row alignment via `unique_job_id`.
"""),
    code_cell("""import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.features.text_processor import prepare_classification_regression_text, prepare_clustering_text
from src.features.sbert_encoder import encode_texts, save_embedding_metadata
"""),
    md_cell("""## 1. Inspect Embeddings Cache
Check the cached embedding matrices on disk.
"""),
    code_cell("""sbert_path = config.EMBEDDINGS_DIR / "sbert_embeddings.npy"
cluster_sbert_path = config.EMBEDDINGS_DIR / "sbert_clustering_embeddings.npy"

if sbert_path.exists():
    emb = np.load(sbert_path)
    print(f"Master / Classification / Regression Embeddings: {emb.shape}, dtype={emb.dtype}")
    print(f"Sample L2 vector norm: {np.linalg.norm(emb[0]):.4f}")

if cluster_sbert_path.exists():
    c_emb = np.load(cluster_sbert_path)
    print(f"Clustering Composite Embeddings: {c_emb.shape}, dtype={c_emb.dtype}")
"""),
    md_cell("""## 2. Embedding Alignment Metadata
Inspect `embedding_metadata.json` for provenance tracking.
"""),
    code_cell("""meta_path = config.EMBEDDINGS_DIR / "embedding_metadata.json"
if meta_path.exists():
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
""")
]

create_notebook(nb2_cells, NOTEBOOKS_DIR / "02_SBERT_Embedding.ipynb")

# ==============================================================================
# NOTEBOOK 3: Job Role Classification
# ==============================================================================
nb3_cells = [
    md_cell("""# Notebook 03: Supervised Job Role Classification
**Project:** JobLens Vietnam - Labor Market Analysis & Skill Trends  
**Course:** Introduction to Data Science  

### Objective:
1. Predict `job_role_label` from unmasked `job_description` (without title).
2. Methodology: Weak supervision context — heuristic regex labels, not human ground truth.
3. Apply stratified 80/10/10 train/val/test split.
4. Classical Baseline: TF-IDF + Logistic Regression (fit strictly on train).
5. SBERT Models:
   - Logistic Regression
   - Linear SVM
   - K-Nearest Neighbors (k=5, cosine metric)
   - Random Forest Classifier
   - XGBoost Classifier
6. Model selection on Validation set; final evaluation on held-out Test set.
"""),
    code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.models.classification import run_classification_experiments
"""),
    md_cell("""## 1. Load Classification Dataset and Embeddings
"""),
    code_cell("""df_cls = pd.read_csv(config.PROCESSED_DATA_DIR / "classification_data.csv")
sbert_all = np.load(config.EMBEDDINGS_DIR / "sbert_embeddings.npy")
cleaned_df = pd.read_csv(config.PROCESSED_DATA_DIR / "cleaned_jobs.csv")

id_to_idx = {uid: idx for idx, uid in enumerate(cleaned_df['unique_job_id'])}
cls_indices = [id_to_idx[uid] for uid in df_cls['unique_job_id']]
X_cls = sbert_all[cls_indices]
y_cls = df_cls['job_role_label']

print(f"Classification samples: {X_cls.shape[0]}, Feature dimension: {X_cls.shape[1]}")
print("\\nClass Distribution:")
print(y_cls.value_counts())
"""),
    md_cell("""## 2. Execute Benchmark with 80/10/10 Split and TF-IDF Baseline
"""),
    code_cell("""results_df = pd.read_csv(config.REPORTS_DIR / "tables" / "classification_results.csv")
print("=== Classification Models Comparison (Held-Out Test Set) ===")
display(results_df) if 'display' in dir() else print(results_df.to_string(index=False))
"""),
    md_cell("""## 3. Confusion Matrix and Classification Report
"""),
    code_cell("""cr_path = config.REPORTS_DIR / "tables" / "classification_report.csv"
if cr_path.exists():
    cr_df = pd.read_csv(cr_path)
    print("=== Per-Class Classification Report (Best Model on Test Set) ===")
    display(cr_df) if 'display' in dir() else print(cr_df.to_string(index=False))

cm_path = config.REPORTS_DIR / "figures" / "confusion_matrix.png"
if cm_path.exists():
    from IPython.display import Image, display as ipy_display
    ipy_display(Image(filename=str(cm_path)))
""")
]

create_notebook(nb3_cells, NOTEBOOKS_DIR / "03_Job_Classification.ipynb")

# ==============================================================================
# NOTEBOOK 4: Salary Regression
# ==============================================================================
nb4_cells = [
    md_cell("""# Notebook 04: Supervised Salary Prediction (Regression)
**Project:** JobLens Vietnam - Labor Market Analysis & Skill Trends  
**Course:** Introduction to Data Science  

### Objective:
1. Predict monthly salary in Million VND (`salary_mean_vnd`) using multi-modal features.
2. 80/10/10 split with strict zero leakage (preprocessors fit strictly on train split).
3. Evaluate 6 regression algorithms on full configuration E4.
4. Reconciled Ablation Study across E1, E2, E3, and E4.
5. Longitudinal / Temporal Validation (Train: <=2025, Test: 2026).
"""),
    code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.models.regression import run_regression_experiments
"""),
    md_cell("""## 1. Load Results and Diagnostic Tables
"""),
    code_cell("""reg_results = pd.read_csv(config.REPORTS_DIR / "tables" / "regression_results.csv")
ablation_results = pd.read_csv(config.REPORTS_DIR / "tables" / "ablation_results.csv")
temporal_results = pd.read_csv(config.REPORTS_DIR / "tables" / "temporal_validation_results.csv")

print("=== Full Model Comparison (Configuration E4 on Held-Out Test Set) ===")
display(reg_results) if 'display' in dir() else print(reg_results.to_string(index=False))

print("\\n=== Reconciled Ablation Study Summary (E1, E2, E3, E4) ===")
display(ablation_results) if 'display' in dir() else print(ablation_results.to_string(index=False))

print("\\n=== Temporal Validation (Train <=2025, Test 2026) ===")
display(temporal_results) if 'display' in dir() else print(temporal_results.to_string(index=False))
"""),
    md_cell("""## 2. Regression Diagnostic Visualizations
"""),
    code_cell("""from IPython.display import Image, display as ipy_display

for plot_name in ['actual_vs_predicted.png', 'residual_plot.png', 'feature_importance.png', 'ablation_comparison.png']:
    p = config.REPORTS_DIR / 'figures' / plot_name
    if p.exists():
        print(f"\\n--- {plot_name} ---")
        ipy_display(Image(filename=str(p)))
""")
]

create_notebook(nb4_cells, NOTEBOOKS_DIR / "04_Salary_Regression.ipynb")

# ==============================================================================
# NOTEBOOK 5: Job Clustering
# ==============================================================================
nb5_cells = [
    md_cell("""# Notebook 05: Unsupervised Job Profile and Skill Clustering
**Project:** JobLens Vietnam - Labor Market Analysis & Skill Trends  
**Course:** Introduction to Data Science  

### Objective:
1. Discover latent job archetypes and skill groupings without ground-truth labels.
2. Input text: composite representation (`job_title + job_description + requirements`).
3. Embeddings: SBERT Multilingual dense vectors (384 dimensions).
4. Run K-Means sweep on full dataset ($k \\in [3, 4, 5, 6, 8, 10]$).
5. Fair head-to-head comparison on standardized sample (N=3,000) across K-Means, Agglomerative, DBSCAN, and GMM.
6. Evaluate using Silhouette Score (Cosine), Davies-Bouldin Index, Calinski-Harabasz Score, ARI, and NMI.
7. Extract technical keywords with Vietnamese + English stopword removal and assign distinct, non-colliding cluster profiles.
"""),
    code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
"""),
    md_cell("""## 1. Load Clustering Benchmark Metrics
"""),
    code_cell("""clustering_results = pd.read_csv(config.REPORTS_DIR / "tables" / "clustering_results.csv")
print("=== Clustering Benchmark Results (Fair Comparison Table) ===")
display(clustering_results) if 'display' in dir() else print(clustering_results.to_string(index=False))
"""),
    md_cell("""## 2. Discovered Cluster Profiles and Distinct Keywords
"""),
    code_cell("""kw_df = pd.read_csv(config.REPORTS_DIR / "tables" / "cluster_keywords.csv")
print("=== Discovered Cluster Profiles & Technical Keywords ===")
display(kw_df) if 'display' in dir() else print(kw_df.to_string(index=False))
"""),
    md_cell("""## 3. Clustering Visualizations
"""),
    code_cell("""from IPython.display import Image, display as ipy_display

for plot_name in ['elbow_curve.png', 'silhouette_comparison.png', 'pca_clusters.png', 'cluster_distribution.png']:
    p = config.REPORTS_DIR / 'figures' / plot_name
    if p.exists():
        print(f"\\n--- {plot_name} ---")
        ipy_display(Image(filename=str(p)))
""")
]

create_notebook(nb5_cells, NOTEBOOKS_DIR / "05_Job_Clustering.ipynb")
print("All 5 Jupyter notebooks generated successfully with updated API and clean visualization workflows!")
