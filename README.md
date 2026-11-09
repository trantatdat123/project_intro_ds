# JobLens Vietnam: Data Science Pipeline & Machine Learning Project

**Course:** Introduction to Data Science  
**Topic:** Vietnamese IT, AI, and Data Science Labor Market & Skill Trends Analysis  

---

## 1. Project Overview

**JobLens Vietnam** is an end-to-end Data Science project analyzing recruitment dynamics, technical skill requirements, and salary determinants in Vietnam's technology sector. The project processes real-world job postings gathered from TopCV and Vietnamese IT/AI portals, transforms raw textual job descriptions using multilingual Sentence-BERT (SBERT) embeddings, and executes three core Machine Learning tasks:

1. **Job Role Classification (Supervised Multiclass NLP):** Predicts standardized career roles (*AI/ML Engineer, Data Scientist, Data Engineer, Data Analyst / BI, Software Engineer, QA/Tester, DevOps/Cloud, IT Support*) from unstructured job descriptions.
2. **Salary Prediction (Supervised Multi-modal Regression):** Estimates monthly salaries in VND by combining dense 384-dimensional SBERT text embeddings with tabular job features (*experience level, location, position hierarchy, employment type, and year*), featuring a 4-configuration ablation study.
3. **Job & Skill Clustering (Unsupervised Representation Learning):** Discovers latent market archetypes and skill co-occurrence patterns from composite job profiles (*title + description + requirements*) using K-Means, Agglomerative Clustering, DBSCAN, and Gaussian Mixture Models.

---

## 2. Workspace Directory Structure

```text
Project Data Science/
├── data/
│   ├── raw/                               # Untouched original raw CSV datasets
│   │   ├── tinixai_it_ai_data_jobs.csv     # 13,964 raw job postings
│   │   └── topcv_jobs.csv                 # 1,503 raw job postings
│   └── processed/                         # Standardized and task-specific datasets
│       ├── merged_jobs.csv                # Schema-standardized & deduplicated master data
│       ├── cleaned_jobs.csv               # Text-cleaned & normalized master data
│       ├── classification_data.csv        # Sub-dataset for Job Role Classification
│       ├── regression_data.csv            # Sub-dataset with valid numeric salaries
│       └── clustering_data.csv            # Sub-dataset for Unsupervised Clustering
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
│   │   ├── loader.py                      # Schema alignment & deduplication
│   │   ├── cleaner.py                     # Text cleaning & categorical normalizers
│   │   ├── salary_parser.py               # Robust salary regex parser & audit
│   │   └── label_generator.py             # Hierarchical job role rule generator
│   ├── features/
│   │   ├── text_processor.py              # Text composite builder
│   │   ├── sbert_encoder.py               # SBERT batch encoder with disk caching
│   │   └── feature_builder.py             # Multi-modal ablation matrix builder
│   ├── models/
│   │   ├── classification.py              # 5 classification algorithm families
│   │   ├── regression.py                  # 6 regression algorithms & ablation study
│   │   └── clustering.py                  # 4 clustering algorithm families
│   └── evaluation/
│       ├── classification_metrics.py      # Macro-F1, accuracy & confusion matrix
│       ├── regression_metrics.py          # MAE, RMSE, R2, actual vs predicted
│       └── clustering_metrics.py          # Silhouette, Davies-Bouldin, PCA 2D plot
│
├── outputs/                               # Artifacts, models, metrics & figures
│   ├── classification/ (metrics/, plots/, models/)
│   ├── regression/     (metrics/, plots/, models/)
│   └── clustering/     (metrics/, plots/, models/)
│
├── reports/                               # Academic deliverables
│   ├── figures/                           # High-resolution experiment plots
│   ├── tables/                            # Performance comparison CSV tables
│   └── Initial_Results_Report.md          # Comprehensive academic checkpoint report
│
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

You can execute the entire Data Science pipeline from data loading to model training and evaluation using one single command:

```bash
python main.py
```

### Running Interactive Jupyter Notebooks
Alternatively, open each notebook in order:
```bash
jupyter notebook notebooks/
```
1. `01_Data_Integration_Preprocessing.ipynb`: Inspect raw datasets, perform deduplication, text normalization, salary parsing, and role labeling.
2. `02_SBERT_Embedding.ipynb`: Generate 384-dimensional dense multilingual SBERT embeddings.
3. `03_Job_Classification.ipynb`: Train and evaluate Logistic Regression, Linear SVM, KNN, Random Forest, and XGBoost.
4. `04_Salary_Regression.ipynb`: Train regression models and evaluate the 4 ablation configurations (E1..E4).
5. `05_Job_Clustering.ipynb`: Perform K-Means sweeps, Agglomerative, DBSCAN, and GMM clustering, and visualize 2D PCA clusters.

---

## 5. Machine Learning Summary

| Task | Primary Input | Target | Primary Metric | Best Baseline Model |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1: Classification** | SBERT(Job Description) | `job_role_label` (8 classes) | Macro-F1 | Linear SVM / Logistic Regression |
| **Task 2: Regression** | SBERT + Tabular + Year | `salary_mean_vnd` (Million VND) | MAE (Million VND) | Random Forest / Extra Trees |
| **Task 3: Clustering** | SBERT(Title + Desc + Reqs) | Latent Archetypes | Silhouette (Cosine) | K-Means ($k=4$) |

Detailed experimental tables, residual analyses, and academic interpretations are documented in [`reports/Initial_Results_Report.md`](file:///c:/Users/Admin/OneDrive/Desktop/Project%20Data%20Science/reports/Initial_Results_Report.md).

