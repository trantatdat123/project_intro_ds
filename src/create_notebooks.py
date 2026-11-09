"""Script to generate all 5 required Jupyter notebooks for JobLens Vietnam."""
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
2. Standardize column schemas, assign global unique IDs (`unique_job_id`), and tag source provenance.
3. Perform intra-source and cross-source deduplication.
4. Clean text fields (stripping HTML, unescaping characters, preserving essential technical tokens like `C++`, `SQL`, `Python`, `AWS`).
5. Parse salary expressions into VND/month (`salary_min_vnd`, `salary_max_vnd`, `salary_mean_vnd`) with currency conversion and validation.
6. Generate standardized `job_role_label` categories using hierarchical keyword matching.
7. Export processed subsets for downstream classification, regression, and clustering.
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
from src.data.cleaner import clean_dataframe
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
We map both datasets into a common 16-attribute schema, synthesize `unique_job_id`, and check for duplicates within and across platforms.
"""),
    code_cell("""merged_raw = standardize_and_merge(df_tinixai, df_topcv)
print(f"Raw merged records: {len(merged_raw)}")

merged_deduped, removed = deduplicate_merged(merged_raw)
print(f"Records after deduplication: {len(merged_deduped)} (removed {len(removed)} duplicate postings)")

merged_deduped[['unique_job_id', 'source', 'job_title', 'company_name', 'salary', 'location']].head(5)
"""),
    md_cell("""## 3. Text Cleaning & Normalization
Apply `clean_dataframe` to strip HTML tags, normalize whitespace, and preserve tech terms (e.g., C++, .NET, Python, SQL). Also normalize structured features: location regions, experience years, job position hierarchy, and year.
"""),
    code_cell("""cleaned_df = clean_dataframe(merged_deduped)
print("Sample cleaned text snippet:")
print(cleaned_df[['job_title', 'norm_location', 'exp_years', 'norm_experience', 'year_clean']].head(5))
"""),
    md_cell("""## 4. Salary Parsing and Auditing
Convert salary strings into `salary_min_vnd`, `salary_max_vnd`, and `salary_mean_vnd`. Apply USD conversion (1 USD = 25,000 VND), parse millions/triệu, and filter implausible bounds (3M - 150M VND/month).
"""),
    code_cell("""parsed_df, salary_audit = process_salaries(cleaned_df)
print("=== Salary Parsing Audit Status ===")
print(salary_audit.to_string(index=False))

valid_salaries = parsed_df[parsed_df['salary_parse_status'] == 'valid']['salary_mean_vnd'] / 1_000_000
print("\\nValid Salary Statistics (Million VND/month):")
print(valid_salaries.describe().round(2))
"""),
    md_cell("""## 5. Job Role Label Generation
Apply hierarchical regex rules to `job_title` to assign ground truth roles for classification. Note: `job_title` is strictly withheld from the classification feature set.
"""),
    code_cell("""labeled_df, label_dist, label_mapping = generate_labels(parsed_df)
print("=== Job Role Label Distribution ===")
print(label_dist.to_string(index=False))
"""),
    md_cell("""## 6. Export Task-Specific Subsets
Save the finalized data files to `data/processed/` for downstream modeling:
- `merged_jobs.csv`: Integrated raw jobs.
- `cleaned_jobs.csv`: Master cleaned dataset.
- `classification_data.csv`: Valid job role labels.
- `regression_data.csv`: Valid numeric salaries.
- `clustering_data.csv`: Valid composite texts.
"""),
    code_cell("""# Save master files
labeled_df.to_csv(config.PROCESSED_DATA_DIR / "cleaned_jobs.csv", index=False)

# Classification subset
cls_df = labeled_df[(labeled_df['job_role_label'] != 'Other / Unassigned') & (labeled_df['job_description'].str.strip() != '')]
cls_df.to_csv(config.PROCESSED_DATA_DIR / "classification_data.csv", index=False)

# Regression subset
reg_df = labeled_df[(labeled_df['salary_parse_status'] == 'valid') & (labeled_df['job_description'].str.strip() != '')]
reg_df.to_csv(config.PROCESSED_DATA_DIR / "regression_data.csv", index=False)

# Clustering subset
cl_df = labeled_df[labeled_df['job_description'].str.strip() != '']
cl_df.to_csv(config.PROCESSED_DATA_DIR / "clustering_data.csv", index=False)

print(f"Exported:\\n- Cleaned: {len(labeled_df)}\\n- Classification: {len(cls_df)}\\n- Regression: {len(reg_df)}\\n- Clustering: {len(cl_df)}")
""")
]

create_notebook(nb1_cells, NOTEBOOKS_DIR / "01_Data_Integration_Preprocessing.ipynb")

# ==============================================================================
# NOTEBOOK 2: SBERT Embedding Generation
# ==============================================================================
nb2_cells = [
    md_cell("""# Notebook 02: Sentence-BERT (SBERT) Multilingual Embeddings
**Project:** JobLens Vietnam - Labor Market Analysis & Skill Trends  
**Course:** Introduction to Data Science  

### Objective:
1. Explain the selection of `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
2. Demonstrate batch encoding for Vietnamese/English technical job descriptions.
3. Validate embedding dimensions (384-dimensional dense vectors) and unit normalization.
4. Align embeddings with tabular datasets using `unique_job_id` keys.
5. Save cached `.npy` arrays and metadata tracking file.
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
from src.features.sbert_encoder import load_sbert_model, encode_texts, save_embedding_metadata
"""),
    md_cell("""## 1. Load Pretrained SBERT Model
We utilize `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, an efficient 12-layer Transformer supporting 50+ languages including Vietnamese and English. It balances embedding quality with CPU inference efficiency.
"""),
    code_cell("""model = load_sbert_model()
print(f"Loaded SBERT model: {config.SBERT_MODEL_NAME}")
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
print(f"Max sequence length: {model.max_seq_length}")
"""),
    md_cell("""## 2. Load Processed Datasets and Text Representations
We extract:
1. `job_description` for Classification and Regression.
2. `job_title + job_description + requirements` for Unsupervised Clustering.
"""),
    code_cell("""cleaned_df = pd.read_csv(config.PROCESSED_DATA_DIR / "cleaned_jobs.csv")
print(f"Cleaned master records: {len(cleaned_df)}")

# Prepare text lists
desc_texts, desc_ids = prepare_classification_regression_text(cleaned_df)
print(f"Sample description text (first 150 chars):\\n{desc_texts[0][:150]}...")
"""),
    md_cell("""## 3. Verify or Compute Embeddings
Load cached embeddings from `embeddings/sbert_embeddings.npy` and `embeddings/sbert_clustering_embeddings.npy` if already generated.
"""),
    code_cell("""sbert_path = config.EMBEDDINGS_DIR / "sbert_embeddings.npy"
embeddings, _ = encode_texts(desc_texts, desc_ids, sbert_path, batch_size=config.SBERT_BATCH_SIZE)

print(f"SBERT Embeddings shape: {embeddings.shape}")
print(f"Vector dtype: {embeddings.dtype}")
print(f"Sample vector norm (should be ~1.0 if normalized): {np.linalg.norm(embeddings[0]):.4f}")
"""),
    md_cell("""## 4. Embedding Metadata and Diagnostics
Inspect `embedding_metadata.json` which tracks dimensionality, sample identifiers, and row alignment.
"""),
    code_cell("""meta_path = config.EMBEDDINGS_DIR / "embedding_metadata.json"
if meta_path.exists():
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
else:
    print("Metadata file will be created by the main pipeline.")
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
1. Formulate a multi-class text classification problem predicting `job_role_label` from `job_description`.
2. Address class imbalance across 8 job role categories.
3. Apply stratified 80/20 train/test split (random seed = 42).
4. Train and compare 5 distinct algorithm families on SBERT embeddings:
   - Logistic Regression (Balanced)
   - Linear Support Vector Machine (Linear SVM)
   - K-Nearest Neighbors (KNN with Cosine metric)
   - Random Forest Classifier
   - XGBoost Classifier
5. Evaluate models using Accuracy, Precision, Recall, Macro-F1 (primary metric), and Weighted-F1.
6. Display Confusion Matrix and analyze misclassifications.
"""),
    code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.models.classification import run_classification_experiments
"""),
    md_cell("""## 1. Load Classification Dataset and Embeddings
We load `classification_data.csv` and slice the corresponding SBERT embedding matrix using aligned `unique_job_id` keys.
"""),
    code_cell("""df_cls = pd.read_csv(config.PROCESSED_DATA_DIR / "classification_data.csv")
sbert_all = np.load(config.EMBEDDINGS_DIR / "sbert_embeddings.npy")
cleaned_df = pd.read_csv(config.PROCESSED_DATA_DIR / "cleaned_jobs.csv")

# Map unique_job_id to embedding row index
id_to_idx = {uid: idx for idx, uid in enumerate(cleaned_df['unique_job_id'])}
cls_indices = [id_to_idx[uid] for uid in df_cls['unique_job_id']]
X_cls = sbert_all[cls_indices]
y_cls = df_cls['job_role_label']

print(f"Classification samples: {X_cls.shape[0]}, Feature dimension: {X_cls.shape[1]}")
print("\\nClass Distribution:")
print(y_cls.value_counts())
"""),
    md_cell("""## 2. Execute Model Training and Evaluation
We execute all 5 models. Class weighting is incorporated to prevent majority classes (e.g., Software Engineer) from dominating specialized roles (Data Scientist, AI Engineer).
"""),
    code_cell("""results_df, best_model, metadata = run_classification_experiments(X_cls, y_cls)
print("=== Classification Model Comparison Table ===")
display(results_df) if 'display' in dir() else print(results_df)
"""),
    md_cell("""## 3. Detailed Classification Report and Confusion Matrix
Inspect per-class precision, recall, and F1 scores for the top-performing model.
"""),
    code_cell("""cr_path = config.CLASSIFICATION_OUTPUTS / 'metrics' / 'classification_report.csv'
if cr_path.exists():
    cr_df = pd.read_csv(cr_path)
    print("=== Per-Class Classification Report (Best Model) ===")
    display(cr_df) if 'display' in dir() else print(cr_df.to_string(index=False))

# Display Confusion Matrix Plot
cm_path = config.CLASSIFICATION_OUTPUTS / 'plots' / 'confusion_matrix.png'
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
2. Combine 384-dimensional SBERT text embeddings with structured features:
   - `job_role_label`
   - `exp_years` (numerical experience)
   - `norm_location` (regional location)
   - `norm_position` (job position hierarchy)
   - `norm_job_type` (employment type)
   - `year_clean` (mandatory year feature)
3. Evaluate 6 regression algorithms:
   - Dummy Regressor (Median baseline)
   - Linear Regression
   - Ridge Regression
   - Random Forest Regressor
   - Extra Trees Regressor
   - XGBoost Regressor
4. Conduct required **Ablation Study**:
   - E1: Structured features without Year
   - E2: Structured features with Year
   - E3: SBERT + Structured features without Year
   - E4: SBERT + Structured features with Year
5. Analyze Actual vs Predicted salary, residuals, and feature importances.
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
    md_cell("""## 1. Load Regression Dataset and Embeddings
Filter records with valid numerical monthly salaries and retrieve corresponding SBERT description embeddings.
"""),
    code_cell("""df_reg = pd.read_csv(config.PROCESSED_DATA_DIR / "regression_data.csv")
sbert_all = np.load(config.EMBEDDINGS_DIR / "sbert_embeddings.npy")
cleaned_df = pd.read_csv(config.PROCESSED_DATA_DIR / "cleaned_jobs.csv")

id_to_idx = {uid: idx for idx, uid in enumerate(cleaned_df['unique_job_id'])}
reg_indices = [id_to_idx[uid] for uid in df_reg['unique_job_id']]
sbert_reg = sbert_all[reg_indices]

print(f"Regression samples: {len(df_reg)}")
print("Salary target summary (Million VND/month):")
print((df_reg['salary_mean_vnd'] / 1_000_000).describe().round(2))
"""),
    md_cell("""## 2. Model Training and Ablation Study
We train all 6 regressors on configuration E4 (full multi-modal feature set) and evaluate ablation configurations E1 through E4.
"""),
    code_cell("""reg_results, ablation_results, best_model, metadata = run_regression_experiments(df_reg, sbert_reg)

print("=== Full Model Comparison (Configuration E4) ===")
display(reg_results) if 'display' in dir() else print(reg_results)

print("\\n=== Ablation Study Summary (E1, E2, E3, E4) ===")
display(ablation_results) if 'display' in dir() else print(ablation_results)
"""),
    md_cell("""## 3. Regression Diagnostic Plots
Examine the Actual vs Predicted scatter plot, residual error distributions, feature importances, and ablation comparison.
"""),
    code_cell("""from IPython.display import Image, display as ipy_display

for plot_name in ['actual_vs_predicted.png', 'residual_plot.png', 'feature_importance.png', 'ablation_comparison.png']:
    p = config.REGRESSION_OUTPUTS / 'plots' / plot_name
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
    md_cell("""# Notebook 05: Unsupervised Job and Skill Clustering
**Project:** JobLens Vietnam - Labor Market Analysis & Skill Trends  
**Course:** Introduction to Data Science  

### Objective:
1. Discover latent job archetypes and skill groupings without ground-truth labels.
2. Input text: composite representation (`job_title + job_description + requirements`).
3. Embeddings: SBERT Multilingual dense vectors (384 dimensions).
4. Run 4 clustering algorithm families:
   - K-Means across cluster counts $K \\in [3, 4, 5, 6, 8, 10]$
   - Agglomerative Hierarchical Clustering (Cosine metric)
   - DBSCAN (density-based clustering with noise identification)
   - Gaussian Mixture Model (GMM on PCA-reduced dimensions)
5. Evaluate using Silhouette Score (Cosine), Davies-Bouldin Index, and Calinski-Harabasz Score.
6. Extract top discriminating keywords and assign human-interpretable cluster profiles.
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
from src.features.text_processor import prepare_clustering_text
from src.models.clustering import run_clustering_experiments
"""),
    md_cell("""## 1. Load Clustering Dataset and Composite Embeddings
We load `clustering_data.csv` and the corresponding `sbert_clustering_embeddings.npy` matrix.
"""),
    code_cell("""df_cluster = pd.read_csv(config.PROCESSED_DATA_DIR / "clustering_data.csv")
cluster_embeddings = np.load(config.EMBEDDINGS_DIR / "sbert_clustering_embeddings.npy")
composite_texts, _ = prepare_clustering_text(df_cluster)

print(f"Clustering samples: {cluster_embeddings.shape[0]}, Dimensions: {cluster_embeddings.shape[1]}")
print(f"Sample composite text (first 150 chars):\\n{composite_texts[0][:150]}...")
"""),
    md_cell("""## 2. Execute Clustering Algorithms and Evaluation
Compute clustering metrics across K-Means, Agglomerative, DBSCAN, and GMM.
"""),
    code_cell("""clustering_results, assignments, best_km_model, metadata = run_clustering_experiments(
    df_cluster, cluster_embeddings, composite_texts
)

print("=== Clustering Evaluation Metrics Comparison ===")
display(clustering_results) if 'display' in dir() else print(clustering_results)
"""),
    md_cell("""## 3. Discovered Cluster Profiles and Top Keywords
Inspect the interpreted cluster archetypes, their top 10 keywords, and job volumes.
"""),
    code_cell("""kw_path = config.CLUSTERING_OUTPUTS / 'metrics' / 'cluster_keywords.csv'
if kw_path.exists():
    kw_df = pd.read_csv(kw_path)
    print("=== Discovered Cluster Profiles & Keywords ===")
    display(kw_df) if 'display' in dir() else print(kw_df.to_string(index=False))

from IPython.display import Image, display as ipy_display

for plot_name in ['elbow_curve.png', 'silhouette_comparison.png', 'pca_clusters.png', 'cluster_distribution.png']:
    p = config.CLUSTERING_OUTPUTS / 'plots' / plot_name
    if p.exists():
        print(f"\\n--- {plot_name} ---")
        ipy_display(Image(filename=str(p)))
""")
]

create_notebook(nb5_cells, NOTEBOOKS_DIR / "05_Job_Clustering.ipynb")
print("All 5 Jupyter notebooks generated successfully!")
