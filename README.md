# JobLens Vietnam: Data Science Pipeline & Machine Learning Project

**Course:** Introduction to Data Science  
**Topic:** Vietnamese IT, AI, and Data Science Labor Market & Skill Trends Analysis  
**Academic Checkpoint:** Initial Results on Applying Machine Learning Models (Comprehensive Bug-Fixed & Audited Edition)  

---

## 1. Project Overview

**JobLens Vietnam** is an end-to-end Data Science project analyzing recruitment dynamics, technical skill requirements, and salary determinants in Vietnam's technology sector. The project processes real-world job postings gathered from TopCV and Vietnamese IT/AI portals, transforms raw textual job descriptions using multilingual Sentence-BERT (SBERT) embeddings, and executes three core Machine Learning tasks:

1. **Job Role Classification (Supervised Multiclass NLP):** Predicts standardized career roles (*AI/ML Engineer, Data Scientist, Data Engineer, Data Analyst / BI, Software Engineer, QA/Tester, DevOps/Cloud, IT Support*) from unstructured job descriptions, featuring a TF-IDF baseline and an **80/10/10 Stratified Train/Val/Test split**.
2. **Salary Prediction (Supervised Multi-modal Regression):** Estimates monthly salaries in Million VND by combining dense 384-dimensional SBERT text embeddings with tabular job features (*experience level, location, position hierarchy, employment type, and year*), featuring a 4-configuration ablation study and a **temporal validation split** ($\le 2025$ vs $2026$).
3. **Job & Skill Clustering (Unsupervised Representation Learning):** Discovers latent market archetypes and skill co-occurrence patterns from composite job profiles (*title + description + requirements*) using K-Means, Agglomerative Clustering, DBSCAN, and Gaussian Mixture Models, benchmarked fairly with external ARI/NMI validation.

---

## 2. Workspace Directory Structure

```text
Project Data Science/
├── data/
│   ├── raw/                               # Untouched original raw CSV datasets
│   │   ├── tinixai_it_ai_data_jobs.csv    # 13,964 raw job postings
│   │   └── topcv_jobs.csv                 # 1,503 raw job postings
│   └── processed/                         # Standardized and task-specific datasets
│       ├── merged_jobs.csv                # Schema-standardized & deduplicated master data (14,037)
│       ├── cleaned_jobs.csv               # Text-cleaned & normalized master data
│       ├── classification_data.csv        # Sub-dataset for Job Role Classification (7,126)
│       ├── regression_data.csv            # Sub-dataset with valid numeric salaries (10,922)
│       └── clustering_data.csv            # Sub-dataset for Unsupervised Clustering (14,037)
│
├── embeddings/                            # Precomputed dense SBERT representations
│   ├── sbert_embeddings.npy               # 384-d normalized embeddings (job_description)
│   ├── sbert_clustering_embeddings.npy    # 384-d embeddings (composite text)
│   └── embedding_metadata.json            # Embedding tracking & unique_job_id alignment
│
├── notebooks/                             # 5 academic Jupyter Notebooks
│   ├── 01_Data_Integration_Preprocessing.ipynb
│   ├── 02_SBERT_Embedding.ipynb
│   ├── 03_Job_Classification.ipynb
│   ├── 04_Salary_Regression.ipynb
│   └── 05_Job_Clustering.ipynb
│
├── src/                                   # Modular Python source package
│   ├── __init__.py
│   ├── config.py                          # Global paths, seeds, rates, hyperparameters
│   ├── data/
│   │   ├── loader.py                      # Schema alignment & deduplication audit
│   │   ├── cleaner.py                     # Text cleaning & location normalization audit
│   │   ├── salary_parser.py               # Independent bounds salary parser & audit
│   │   └── label_generator.py             # Rule-based job role generator (weak supervision)
│   ├── features/
│   │   ├── text_processor.py              # Text composite builder
│   │   ├── sbert_encoder.py               # SBERT batch encoder with disk caching
│   │   └── feature_builder.py             # Multi-modal ablation matrix builder (80/10/10)
│   ├── models/
│   │   ├── classification.py              # TF-IDF baseline + 5 SBERT classifiers (80/10/10)
│   │   ├── regression.py                  # 6 regressors, reconciled ablation & temporal validation
│   │   └── clustering.py                  # K-Means, Agglomerative, DBSCAN, GMM fair benchmark
│   └── evaluation/
│       ├── classification_metrics.py      # Macro-F1, accuracy & confusion matrix
│       ├── regression_metrics.py          # MAE, RMSE, R2, actual vs predicted
│       └── clustering_metrics.py          # Silhouette, Davies-Bouldin, ARI, NMI, PCA 2D plot
│
├── tests/                                 # Automated Unit Test Suite (17 tests, 100% pass)
│   ├── __init__.py
│   ├── test_location_normalization.py     # Verifies Hanoi 'quận' bug fix
│   ├── test_salary_parser.py              # Verifies bounds, currency & categories
│   ├── test_deduplication.py              # Verifies composite deduplication
│   └── test_data_splitting.py             # Verifies 80/10/10 split & zero leakage
│
├── outputs/                               # Artifacts, models, metrics & figures
│   ├── split_manifest.csv                 # Complete sample allocation (train/val/test)
│   ├── classification/ (metrics/, plots/, models/)
│   ├── regression/     (metrics/, plots/, models/)
│   └── clustering/     (metrics/, plots/, models/)
│
├── reports/                               # Academic deliverables
│   ├── figures/                           # High-resolution experiment plots
│   ├── tables/                            # Performance comparison & audit CSV tables
│   ├── Initial_Results_Report.md          # Comprehensive academic checkpoint report
│   └── report_draft.md                    # Detailed technical explanation report (Vietnamese)
│
├── demo_predict.py                        # Interactive live prediction demo
├── requirements.txt                       # Pinned Python package dependencies
├── README.md                              # Project documentation
├── .gitignore                             # Ignored cache, models & large binaries
└── main.py                                # End-to-end reproducible pipeline runner
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.11 / 3.13)
- Windows / macOS / Linux

### Virtual Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

---

## 4. Reproducing the Experiments

### 1. Run Complete Automated Unit Tests
Verify data processing logic and split policies:
```bash
python -m unittest discover -s tests
```
*(17 tests run in 0.02s: OK - 100% Pass)*

### 2. Execute End-to-End Pipeline
Run data ingestion, location audit, salary parsing, SBERT encoding, 80/10/10 training, and evaluation:
```bash
python main.py
```

### 3. Test Live Inference Demo
Predict role, estimate salary, and identify cluster archetype for any job posting:
```bash
python demo_predict.py
```

### 4. Interactive Jupyter Notebooks
Run the notebooks sequentially:
```bash
jupyter notebook notebooks/
```
1. `01_Data_Integration_Preprocessing.ipynb`: Inspect raw datasets, deduplicate with audit, audit location fixes, parse salaries with independent bounds, and generate role labels.
2. `02_SBERT_Embedding.ipynb`: Verify dense 384-dimensional multilingual SBERT representations and metadata tracking.
3. `03_Job_Classification.ipynb`: Evaluate TF-IDF baseline and SBERT classifiers under stratified 80/10/10 split.
4. `04_Salary_Regression.ipynb`: Evaluate regressors on E4, run reconciled ablation study (E1–E4), and test temporal validation.
5. `05_Job_Clustering.ipynb`: Perform K-Means sweeps, fair sample benchmark, and visualize distinct cluster archetypes with stopword filtering.

---

## 5. Machine Learning Summary (Audited Results)

| Task | Primary Features | Target | Primary Metric | Best Baseline / Model | Best Performance (Held-Out Test Set) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Task 1: Classification** | SBERT (384-d) / TF-IDF | `job_role_label` (8 classes) | Macro-F1 / Accuracy | SBERT + KNN / TF-IDF + Logistic Reg | **SBERT KNN: Acc 87.52%, Macro-F1 0.7704**<br>*(TF-IDF Baseline: Acc 91.58%, Macro-F1 0.8505)* |
| **Task 2: Regression** | SBERT + Tabular + Year | `salary_mean_vnd` (Million VND) | MAE (Million VND) | XGBoost Regressor (E4) | **Test MAE: 5.24M VND, $R^2$: 0.3604**<br>*(Baseline Median MAE: 8.35M VND)* |
| **Task 3: Clustering** | SBERT(Title + Desc + Reqs) | Latent Skill Archetypes | Silhouette (Cosine) / NMI | K-Means ($k=8$) | **Silhouette: 0.1224 (Full) / 0.1245 (Sample)**<br>*(NMI vs Roles: 0.3133)* |

Detailed experimental tables, location fix audits, residual analyses, and academic interpretations are documented in [`reports/Initial_Results_Report.md`](file:///c:/Users/Admin/OneDrive/Desktop/Project%20Data%20Science/reports/Initial_Results_Report.md) and [`reports/report_draft.md`](file:///c:/Users/Admin/OneDrive/Desktop/Project%20Data%20Science/reports/report_draft.md).
