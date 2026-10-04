# 📋 JobLens Vietnam: Implementation Plan

A streamlined 5-step implementation plan designed around the core technical specification modules.

---

## 📌 Implementation Steps

### Step 1: Project Setup & TinixAI Data Acquisition
- [x] Initialize project directory structure (`data/`, `src/`, `configs/`, `notebooks/`, `tests/`).
- [x] Create `requirements.txt` for TinixAI download/filtering: `pandas`, `pyarrow`, `huggingface_hub`, `tqdm`, `ipykernel`. Add modeling dependencies when those stages are implemented.
- [x] Construct `configs/skill_dictionary.json` defining canonical skill names and synonym mappings.
- [x] Download TinixAI `data.parquet` into `data/raw/tinixai_vietnamese_job_descriptions/`.
- [x] Filter IT/AI/Data Science titles using `notebooks/tinix_download_filter.ipynb`, exporting CSV, Parquet and metadata to `data/processed/`.

### Step 2: Data Preprocessing Pipeline (`src/preprocessing/`)
- [x] **`cleaner.py`**: Implement HTML stripping (`clean_html`), Vietnamese text normalization (`normalize_text`), and minimum experience parsing (`parse_experience`).
- [x] **`salary_parser.py`**: Implement salary range extraction and VND conversion (`parse_salary_range`), followed by outlier filtering (`remove_salary_outliers`).
- [x] **`skill_extractor.py`**: Implement FlashText keyword processor to extract deduplicated skills from job descriptions (`extract_skills`).
- [x] **`role_normalizer.py`**: Map unstructured job titles into 5 standard roles (`map_standard_role`).
- [x] Export cleaned dataset to `data/processed/jobs_cleaned.csv`.

### Step 3: Feature Engineering & Exploratory Data Analysis (EDA)
- [x] **`build_features.py`**:
  - Generate multi-hot binary skill matrix with frequency thresholding (`create_skill_matrix`).
  - Compute dense semantic embeddings using multilingual Sentence-Transformers (`generate_text_embeddings`).
- [x] Create EDA notebook (`notebooks/01_exploratory_data_analysis.ipynb`): Analyze role distributions, salary spreads, required experience levels, and top demanding skills.

### Step 4: Machine Learning Modeling & Evaluation (`src/models/`)
- [x] **`classifier.py`**: Train role classifiers (TF-IDF + Logistic Regression / LinearSVC / MLP), evaluate via Macro-F1, Precision, Recall, and Confusion Matrix.
- [x] **`regressor.py`**: Train salary regression models (LightGBM / Random Forest / Ridge) on $\log(\text{salary})$, evaluate via MAE & $R^2$, and perform SHAP feature importance analysis.
- [x] **`cluster.py`**:
  - Build skill co-occurrence matrix and construct NetworkX graph for topology & centrality analysis.
  - Apply UMAP dimensionality reduction and cluster job postings via K-Means / HDBSCAN.

### Step 5: Pipeline Integration & Reporting
- [x] Build an end-to-end execution script (`main.py` or `src/pipeline.py`) to automate data ingestion through model evaluation.
- [x] Compile the final course project report summarizing classification performance, salary insights, and skill ecosystem trends.

---

## 🎯 Deliverables
1. Complete, modular code under `src/` executing without errors.
2. Cleaned and structured dataset at `data/processed/jobs_cleaned.csv`.
3. Jupyter notebooks demonstrating EDA, model evaluation, and visual analytics.
4. Comprehensive project report and presentation slides for IntroDS.


## Completed artifacts

- Runner: `python main.py --stage all`.
- Cleaned data: `data/processed/jobs_cleaned.csv` and Parquet (14,034 rows).
- Executed notebooks: `01_exploratory_data_analysis.ipynb`, `02_modeling_evaluation.ipynb`, `03_skills_and_clustering.ipynb`.
- Report: [project_report.md](reports/project_report.md); schema: [data_dictionary.md](reports/data_dictionary.md).
- Slides: [joblens_vietnam_v3.pptx](reports/slides/joblens_vietnam_v3.pptx) (13 slides).
- Models and features: `artifacts/models/`, `data/features/`; metrics and provenance: `reports/`.
- Actual test results: selected LinearSVC macro-F1 0.839 (weak title labels), selected LightGBM MAE 6.12 million VND/month, R² 0.387.
- Semantic analysis: KMeans 8 clusters; HDBSCAN 44 clusters with 5,060 noise rows. Silhouette is measured in UMAP space.
