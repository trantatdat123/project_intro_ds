# 📋 JobLens Vietnam: Implementation Plan

A streamlined 5-step implementation plan designed around the core technical specification modules.

---

## 📌 Implementation Steps

### Step 1: Project Setup & TinixAI Data Acquisition
- [ ] Initialize project directory structure (`data/`, `src/`, `configs/`, `notebooks/`, `tests/`).
- [x] Create `requirements.txt` for TinixAI download/filtering: `pandas`, `pyarrow`, `huggingface_hub`, `tqdm`, `ipykernel`. Add modeling dependencies when those stages are implemented.
- [ ] Construct `configs/skill_dictionary.json` defining canonical skill names and synonym mappings.
- [x] Download TinixAI `data.parquet` into `data/raw/tinixai_vietnamese_job_descriptions/`.
- [x] Filter IT/AI/Data Science titles using `notebooks/tinix_download_filter.ipynb`, exporting CSV, Parquet and metadata to `data/processed/`.

### Step 2: Data Preprocessing Pipeline (`src/preprocessing/`)
- [ ] **`cleaner.py`**: Implement HTML stripping (`clean_html`), Vietnamese text normalization (`normalize_text`), and minimum experience parsing (`parse_experience`).
- [ ] **`salary_parser.py`**: Implement salary range extraction and VND conversion (`parse_salary_range`), followed by outlier filtering (`remove_salary_outliers`).
- [ ] **`skill_extractor.py`**: Implement FlashText keyword processor to extract deduplicated skills from job descriptions (`extract_skills`).
- [ ] **`role_normalizer.py`**: Map unstructured job titles into 5 standard roles (`map_standard_role`).
- [ ] Export cleaned dataset to `data/processed/jobs_cleaned.csv`.

### Step 3: Feature Engineering & Exploratory Data Analysis (EDA)
- [ ] **`build_features.py`**:
  - Generate multi-hot binary skill matrix with frequency thresholding (`create_skill_matrix`).
  - Compute dense semantic embeddings using multilingual Sentence-Transformers (`generate_text_embeddings`).
- [ ] Create EDA notebook (`notebooks/01_exploratory_data_analysis.ipynb`): Analyze role distributions, salary spreads, required experience levels, and top demanding skills.

### Step 4: Machine Learning Modeling & Evaluation (`src/models/`)
- [ ] **`classifier.py`**: Train role classifiers (TF-IDF + Logistic Regression / LinearSVC / MLP), evaluate via Macro-F1, Precision, Recall, and Confusion Matrix.
- [ ] **`regressor.py`**: Train salary regression models (LightGBM / Random Forest / Ridge) on $\log(\text{salary})$, evaluate via MAE & $R^2$, and perform SHAP feature importance analysis.
- [ ] **`cluster.py`**:
  - Build skill co-occurrence matrix and construct NetworkX graph for topology & centrality analysis.
  - Apply UMAP dimensionality reduction and cluster job postings via K-Means / HDBSCAN.

### Step 5: Pipeline Integration & Reporting
- [ ] Build an end-to-end execution script (`main.py` or `src/pipeline.py`) to automate data ingestion through model evaluation.
- [ ] Compile the final course project report summarizing classification performance, salary insights, and skill ecosystem trends.

---

## 🎯 Deliverables
1. Complete, modular code under `src/` executing without errors.
2. Cleaned and structured dataset at `data/processed/jobs_cleaned.csv`.
3. Jupyter notebooks demonstrating EDA, model evaluation, and visual analytics.
4. Comprehensive project report and presentation slides for IntroDS.
