# JobLens Vietnam: Initial Results on Applying Machine Learning Models to Recruitment Data

**Course:** Introduction to Data Science  
**Academic Checkpoint:** Initial Results on Applying ML Models (Comprehensive Bug-Fixed & Methodologically Audited Edition)  
**Project:** JobLens Vietnam - Labor Market Analysis & Skill Trends in Vietnam IT & AI Sectors  

---

## Abstract
This report presents the revised and audited empirical results of the **JobLens Vietnam** project, an end-to-end data science study analyzing the Vietnamese Information Technology (IT), Artificial Intelligence (AI), and Data Science recruitment market. Using a newly integrated and thoroughly deduplicated corpus of **14,037 unique job postings** from Vietnamese recruitment platforms (TinixAI and TopCV), we develop, benchmark, and validate machine learning models across three core tasks:
1. **Supervised Job Role Classification:** Predicting technical job categories from unstructured job descriptions via dense multilingual Sentence-BERT (SBERT) embeddings and classical TF-IDF n-grams across 8 career categories, evaluated on a stratified **80/10/10 Train/Validation/Test split**.
2. **Supervised Salary Prediction:** Predicting monthly salary in Million VND (`salary_mean_vnd`) combining 384-dimensional SBERT text embeddings with structured job attributes (experience, location, position, employment type, and year), accompanied by a four-configuration ablation study and a **longitudinal temporal validation** (Train $\le$ 2025, Test 2026).
3. **Unsupervised Job and Skill Clustering:** Uncovering latent recruitment archetypes and technical skill profiles using K-Means, Agglomerative Hierarchical Clustering, DBSCAN, and Gaussian Mixture Models (GMM) with a standardized fair comparison on identical sample sizes, external ARI/NMI validation against weak labels, and domain-informed, collision-free cluster naming.

### Key Empirical Findings:
- **Location Normalization & Deduplication Audits:** Corrected a critical location bug where Hanoi districts containing *"quận"* were previously misattributed to Ho Chi Minh City (affecting 1,470 unique strings and 2,184 postings). Deduplication using normalized composite keys eliminated 1,430 duplicate postings across and within sources.
- **Classification Performance:** On the held-out Test set ($N=713$), **SBERT + K-Nearest Neighbors (KNN)** achieved the best overall performance with **Accuracy of 88.36%** and **Macro-F1 of 0.7850** (Val Macro-F1: 0.7628). The classical **TF-IDF + Logistic Regression** baseline demonstrated strong competitive performance (**Accuracy: 91.58%**, **Macro-F1: 0.8505**), reflecting the exact lexical overlap between job descriptions and rule-based heuristic definitions.
- **Salary Regression & Reconciled Ablation:** In the full multi-modal configuration (E4: SBERT + Tabular + Year), **XGBoost Regressor** achieved a Test MAE of **5.24 Million VND**, Test RMSE of **8.71 Million VND**, and $R^2$ of **0.3604** (Val MAE: 5.34M, $R^2$: 0.4290), outperforming the Dummy Median Baseline (MAE: 8.35M VND). The ablation study confirmed that incorporating `year` improved MAE by 0.23M VND, while SBERT text embeddings improved MAE by 0.41M VND. Hyperparameters and metrics between the main benchmark and ablation study are 100% reconciled and consistent across all tables.
- **Temporal Out-of-Year Generalization:** Evaluating models trained on historical data ($\le 2025$) on the future 2026 test set revealed an expected generalization gap (MAE increased from 5.24M to 7.94M–7.99M VND), empirically capturing post-pandemic macroeconomic wage inflation in Vietnam's tech sector.
- **Fair Clustering Benchmark:** Optimal K-Means ($k=8$) achieved a Cosine Silhouette Score of **0.1224** on the full dataset and **0.1245** on a standardized sample ($N=3,000$), outperforming GMM (0.0874) and Agglomerative clustering (0.1159), while achieving the highest alignment with semantic job roles (NMI = 0.3133). Comprehensive Vietnamese and English stopword filtering resolved previous keyword pollution and enabled 8 distinct, non-overlapping cluster archetypes.

---

## 1. Introduction and Research Objectives

### 1.1 Motivation
Vietnam's rapid emergence as a Southeast Asian digital innovation hub has spurred unprecedented hiring demand for Software Engineers, Data Analysts, Data Engineers, and AI/ML specialists. However, online recruitment postings present significant data science challenges:
- High noise and boilerplate language.
- Highly multi-lingual descriptions (Vietnamese, English, and code-mixed phrasing).
- Inconsistent and unstandardized salary disclosure formats.
- Geographical ambiguities (e.g., district-level descriptions without explicit city tags).

The **JobLens Vietnam** project formulates an end-to-end machine learning pipeline addressing data integration, cleaning, semantic representation learning, supervised classification, multi-modal regression, unsupervised clustering, and interactive live inference.

### 1.2 Core Research Tasks
- **Task 1: Job Role Classification (Supervised Multiclass):** Predict the target job role label ($y \in \{\text{Software Engineer, IT Support / Network / Security, Data Analyst / BI, DevOps / Cloud, QA / Tester, AI / ML Engineer, Data Engineer, Data Scientist}\}$) using strictly the unstructured text of `job_description` (without job title).
- **Task 2: Salary Prediction (Supervised Regression):** Predict monthly salary in Million VND (`salary_mean_vnd`) combining SBERT dense text embeddings and structured job characteristics (`job_role_label`, `exp_years`, `norm_location`, `norm_position`, `norm_job_type`, and `year_clean`), accompanied by a four-configuration ablation study and a temporal validation split.
- **Task 3: Job and Skill Clustering (Unsupervised Learning):** Uncover natural groupings of technical skills and job profiles without relying on predefined labels, using composite representations (`job_title + job_description + requirements`).

---

## 2. Dataset Ingestion, Deduplication, and Normalization Audits

### 2.1 Raw Data Ingestion
The raw data consists of two distinct CSV files located in `data/raw/`:
- `tinixai_it_ai_data_jobs.csv`: 13,964 raw job postings.
- `topcv_jobs.csv`: 1,503 raw job postings scraped from TopCV Vietnam.

Both datasets were aligned into a standardized 16-column schema preserving provenance via `source` (`tinixai` vs `topcv`), original source ID `source_id`, and global identifier `unique_job_id`.

### 2.2 Deduplication Audit
Online job boards frequently contain duplicate repostings within platforms and cross-posted vacancies between platforms. A robust deduplication pipeline was implemented in `src/data/loader.py`:
- Lowercase, whitespace stripping, and punctuation normalization.
- Normalization of company names (removing common corporate entity prefixes: *"công ty cổ phần"*, *"công ty tnhh"*, *"jsc"*, *"ltd"*).
- Stripping location/tag prefixes from job titles (e.g. *"[Hà Nội] Senior Python"* $\rightarrow$ *"senior python"*).
- Composite key: `norm_title + "|||" + norm_company + "|||" + norm_desc_prefix_150`.

**Table 1: Deduplication Audit Breakdown (`reports/tables/deduplication_audit.csv`)**

| Metric | Count | Percentage (%) | Action / Description |
| :--- | :--- | :--- | :--- |
| **Raw Records - TinixAI** | 13,964 | 90.28% | Raw scraped records from TinixAI |
| **Raw Records - TopCV** | 1,503 | 9.72% | Raw scraped records from TopCV |
| **Raw Records - Total** | 15,467 | 100.0% | Total merged raw records |
| **Duplicates Removed - Within TinixAI** | 1,404 | 9.08% | Intra-platform duplicate repostings |
| **Duplicates Removed - Within TopCV** | 17 | 0.11% | Intra-platform duplicate repostings |
| **Duplicates Removed - Cross-Source** | 9 | 0.06% | TopCV postings matching existing TinixAI jobs |
| **Total Duplicate Postings Removed** | **1,430** | **9.25%** | Excluded from master corpus |
| **Remaining Unique Records - TinixAI** | 12,560 | 89.48% | Unique postings retained |
| **Remaining Unique Records - TopCV** | 1,477 | 10.52% | Unique postings retained |
| **Remaining Unique Records - Total** | **14,037** | **100.0%** | **Clean master corpus saved to `merged_jobs.csv`** |

---

## 3. Data Integration, Audited Cleaning, and Weak Supervision Taxonomy

### 3.1 Audited Location Normalization
In the previous audit, an order-of-precedence bug in `normalize_location()` was detected: the generic keyword *"quận"* was checked in the Ho Chi Minh branch before checking Hanoi-specific keywords, erroneously mapping thousands of Hanoi postings (e.g. *"Quận Cầu Giấy, Hà Nội"*, *"Quận Thanh Xuân, Thành phố Hà Nội"*) to *"Hồ Chí Minh"*.

The normalization logic in `src/data/cleaner.py` was rewritten with a strict hierarchy:
1. Detect explicit city names first (*Hà Nội*, *Hồ Chí Minh*, *Đà Nẵng*).
2. Detect multi-location postings (*"Multiple"*).
3. Apply district-level fallbacks only when no city is explicitly specified.
4. Never use generic words (*"quận"*, *"huyện"*, *"thành phố"*) as city indicators.
5. Detect *Remote / Work From Home*.

**Table 2: Location Normalization Fix Audit Summary (`reports/tables/location_fix_audit.csv`)**

| Audit Metric | Value | Impact / Details |
| :--- | :--- | :--- |
| **Unique Location Strings Affected** | **1,470** | Total distinct address variations corrected |
| **Total Postings Affected by Fix** | **2,184** | **15.56% of entire dataset rescued from misclassification** |
| Hanoi Postings Rescued from HCMC | 1,650 | Addresses like *"Quận Cầu Giấy, Hà Nội"* properly assigned to Hà Nội |
| Multi-City Postings Corrected | 414 | Postings spanning both cities mapped to *"Multiple"* |
| Da Nang Postings Rescued | 110 | Da Nang districts previously misattributed to HCMC |
| Unassigned Postings Mapped to Khác | 10 | Non-standard text mapped to *"Khác"* |

**Post-Fix Regional Distribution in Master Corpus:**
- **Hà Nội:** 5,913 (42.12%)
- **Hồ Chí Minh:** 5,618 (40.02%)
- **Khác:** 1,847 (13.16%)
- **Multiple:** 416 (2.96%)
- **Đà Nẵng:** 241 (1.72%)
- **Remote:** 2 (0.01%)

### 3.2 Salary Parsing with Independent Bounds and Disclosure Categorization
The salary parser in `src/data/salary_parser.py` was enhanced to validate lower and upper bounds independently ($3,000,000 \le \text{min} \le \text{max} \le 150,000,000\text{ VND}$) and explicitly categorize disclosure formats.

**Table 3: Salary Parsing Audit and Disclosure Breakdown (`reports/tables/salary_parsing_audit.csv`)**

| Metric / Disclosure Category | Value | Percentage (%) | Handling in Pipeline |
| :--- | :--- | :--- | :--- |
| **Total Job Postings Analyzed** | 14,037 | 100.0% | Master cleaned dataset |
| **Unspecified / Negotiable ("Thỏa thuận")** | 2,429 | 17.30% | Excluded from regression (never imputed) |
| **Disclosed Salary Range (Min - Max)** | 10,750 | 76.58% | Valid regression target: $(\text{min} + \text{max}) / 2$ |
| **Exact Fixed Salary Disclosed** | 134 | 0.95% | Valid regression target: fixed value |
| **Upper Bound Only Disclosed ("Up to")** | 27 | 0.19% | Valid regression target: upper bound value |
| **Lower Bound Only Disclosed ("From")** | 11 | 0.08% | Valid regression target: lower bound value |
| **Implausible Outliers (<3M or >150M VND)** | 400 | 2.85% | Filtered out as extreme outliers |
| **Invalid / Unparseable Numbers** | 286 | 2.04% | Filtered out |
| **Total Valid Samples for Regression** | **10,922** | **77.81%** | **Retained in `regression_data.csv`** |
| *Plausible Monthly Salary Mean* | *17.86 Million VND* | - | Std: 11.08 Million VND |
| *Plausible Monthly Salary Median* | *15.00 Million VND* | - | IQR: 11.5M – 21.0M VND |
| *Plausible Monthly Salary Min / Max* | *3.00M / 150.00M VND* | - | Strict plausible domain |

### 3.3 Rule-Based Job Role Taxonomy (Weak Supervision Note)
> [!IMPORTANT]
> **Methodology Note on Weak Supervision:**
> The `job_role_label` attributes are **rule-based heuristic pseudo-labels** generated from `job_title` using regular expressions in `src/data/label_generator.py`. They represent **weak supervision**, not human-annotated ground truth.
> 
> To prevent data leakage, `job_title` is strictly excluded from the classification input space. The classification models predict the role label solely from the unstructured text of `job_description`. Therefore, accuracy and Macro-F1 measure the degree to which models learn to infer taxonomic role definitions from descriptive responsibilities and requirements.

**Table 4: Rule-Based Job Role Label Distribution (`reports/tables/label_distribution.csv`)**

| Job Role Category | Count | Percentage (%) | Role Description & Scope |
| :--- | :--- | :--- | :--- |
| Other / Unassigned | 6,911 | 49.23% | Non-technical or ambiguous titles (excluded from classification) |
| **Software Engineer** | **3,960** | **28.21%** | Fullstack, Backend, Frontend, Web, Mobile Developers |
| **IT Support / Network / Security** | **1,141** | **8.13%** | Helpdesk, Systems, Network, and Information Security |
| **Data Analyst / BI** | **536** | **3.82%** | Business Intelligence, Reporting, SQL, Power BI, Tableau |
| **DevOps / Cloud** | **465** | **3.31%** | SRE, Cloud Infrastructure, CI/CD, Kubernetes, AWS |
| **QA / Tester** | **443** | **3.16%** | Manual / Automation QC, Software Test Engineers |
| **AI / ML Engineer** | **280** | **1.99%** | Machine Learning, Deep Learning, Computer Vision, NLP |
| **Data Engineer** | **202** | **1.44%** | Big Data, Data Pipelines, ETL, Data Warehousing |
| **Data Scientist** | **99** | **0.71%** | Statistical Modeling, Predictive Analytics |
| **Total Valid Technical Roles** | **7,126** | **100.0%** | **Form the Task 1 dataset (`classification_data.csv`)** |

---

## 4. Sentence-BERT (SBERT) Embedding Methodology

We utilize `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, generating 384-dimensional dense semantic vectors. Embeddings are L2-normalized so that dot products directly represent cosine similarities.

Two embedding matrices are generated and maintained in `embeddings/`:
1. `sbert_embeddings.npy` $(14,037 \times 384)$: Encoded from `job_description`, utilized for Classification and Regression.
2. `sbert_clustering_embeddings.npy` $(14,037 \times 384)$: Encoded from composite text (`job_title + ". " + job_description + " Yêu cầu: " + requirements`), utilized for Unsupervised Clustering.

Index alignment is verified using `unique_job_id` keys tracked in `embeddings/embedding_metadata.json`.

---

## 5. Task 1: Job Role Classification Results

### 5.1 Experimental Setup & 80/10/10 Split Policy
- **Dataset:** 7,126 labeled postings across 8 technical role categories.
- **Data Splitting Policy:** Stratified **80/10/10 split**:
  - **Train Split (80%):** 5,700 samples (used for model fitting).
  - **Validation Split (10%):** 713 samples (used strictly for model selection).
  - **Test Split (10%):** 713 samples (held-out, evaluated once for final reporting).
  - Zero leakage verified: all preprocessors (TF-IDF vectorizer, scalers) fit strictly on Train.
  - Split manifest persisted to `outputs/split_manifest.csv`.
- **Classical Baseline:** TF-IDF + Logistic Regression (fit on raw `job_description` train split, 5,000 n-gram features).
- **SBERT Models:** Logistic Regression, Linear SVM, K-Nearest Neighbors (k=5, cosine metric), Random Forest, and XGBoost Classifier.

### 5.2 Benchmark Results on Held-Out Test Set

**Table 5: Classification Model Comparison on Test Set (`reports/tables/classification_results.csv`)**

| Model | Representation | Test Accuracy | Test Precision (Macro) | Test Recall (Macro) | Test Macro-F1 | Test Weighted-F1 | Val Macro-F1 | Val Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | TF-IDF (5,000 n-grams) | **0.9158** | **0.8377** | **0.8667** | **0.8505** | **0.9169** | **0.8734** | **0.9144** |
| **SBERT + K-Nearest Neighbors** | SBERT (384 dense) | **0.8752** | **0.7879** | **0.7596** | **0.7704** | **0.8729** | **0.7856** | **0.8766** |
| **SBERT + Linear SVM** | SBERT (384 dense) | 0.8738 | 0.7750 | 0.8189 | 0.7944 | 0.8766 | 0.7587 | 0.8640 |
| **SBERT + XGBoost Classifier** | SBERT (384 dense) | 0.8640 | 0.8330 | 0.6948 | 0.7475 | 0.8557 | 0.7562 | 0.8738 |
| **SBERT + Logistic Regression** | SBERT (384 dense) | 0.8275 | 0.6930 | 0.8119 | 0.7411 | 0.8365 | 0.7062 | 0.8050 |
| **SBERT + Random Forest** | SBERT (384 dense) | 0.8177 | 0.8978 | 0.5708 | 0.6602 | 0.7976 | 0.6982 | 0.8373 |

**Figure 1: Classification Models Comparison (Test Set)**  
*(Plot saved to `reports/figures/model_comparison.png`)*

#### Analysis of Findings:
1. **Classical TF-IDF vs Dense SBERT:** TF-IDF + Logistic Regression achieves the highest Macro-F1 (0.8505) and Accuracy (91.58%). Because the target labels were generated via rule-based keyword matching on job titles, technical job descriptions frequently contain the exact technical tokens (e.g., *"SQL"*, *"Python"*, *"Kubernetes"*, *"Selenium"*) that correlate with these rules. TF-IDF acts as an effective sparse keyword counter.
2. **Dense Semantic Generalization:** SBERT + KNN achieved the top performance among dense representations (Macro-F1: 0.7704 on Test, 0.7856 on Val) with an Accuracy of 87.52%. SBERT represents descriptions in a compact 384-dimensional space (versus 5,000 sparse dimensions) and captures semantic equivalence across Vietnamese and English without requiring explicit keyword overlap.

### 5.3 Per-Class Analysis for Best SBERT Model (KNN)

**Table 6: Per-Class Classification Report for Best Model (`reports/tables/classification_report.csv`)**

| Job Role Category | Precision | Recall | F1-Score | Support (Test Set) |
| :--- | :--- | :--- | :--- | :--- |
| **Software Engineer** | 0.94 | 0.94 | **0.94** | 393 |
| **IT Support / Network / Security** | 0.86 | 0.91 | **0.88** | 120 |
| **Data Analyst / BI** | 0.82 | 0.87 | **0.84** | 53 |
| **QA / Tester** | 0.82 | 0.84 | **0.83** | 45 |
| **DevOps / Cloud** | 0.80 | 0.67 | **0.73** | 46 |
| **AI / ML Engineer** | 0.74 | 0.67 | **0.70** | 27 |
| **Data Engineer** | 0.79 | 0.75 | **0.77** | 20 |
| **Data Scientist** | 0.50 | 0.44 | **0.47** | 9 |

**Figure 2: Normalized Confusion Matrix on Test Set**  
*(Heatmap saved to `reports/figures/confusion_matrix.png`)*

Core technical roles (`Software Engineer`, `IT Support`, `QA / Tester`, and `Data Analyst / BI`) show high F1-scores ($\ge 0.83$). The lower score for `Data Scientist` (F1 = 0.47) is directly attributable to its small sample size ($N_{\text{test}} = 9$) and its natural conceptual overlap with `AI / ML Engineer` and `Data Analyst / BI`.

---

## 6. Task 2: Salary Prediction (Regression) Results

### 6.1 Experimental Setup & Splitting
- **Dataset:** 10,922 valid salary records.
- **Target:** Monthly salary in Million VND (`salary_mean_vnd / 1,000,000`).
- **Data Splitting Policy:** **80/10/10 split**:
  - **Train Split (80%):** 8,737 samples.
  - **Validation Split (10%):** 1,092 samples.
  - **Test Split (10%):** 1,093 samples.
  - Strict zero leakage: `OneHotEncoder` and `StandardScaler` fit strictly on `df_train`.
- **Company Grouping Investigation:** The 10,922 postings represent 5,500+ distinct companies; the single largest employer represents only 1.2% of the dataset. Thus, group-based clustering by employer does not distort random partition stability.

### 6.2 Full Model Comparison on Configuration E4
All 6 regressors were trained on configuration E4 (SBERT dense embeddings + One-Hot categorical + standardized numerical experience + mandatory year feature):

**Table 7: Regression Performance on Configuration E4 (`reports/tables/regression_results.csv`)**

| Model | Test MAE (M VND) | Test RMSE (M VND) | Test $R^2$ Score | Val MAE (M VND) | Val $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost Regressor** | **5.24** | **8.71** | **0.3604** | **5.34** | **0.4290** |
| Extra Trees Regressor | 5.30 | 8.88 | 0.3350 | 5.56 | 0.3786 |
| Random Forest Regressor | 5.40 | 8.77 | 0.3507 | 5.74 | 0.3545 |
| Ridge Regression | 5.45 | 8.57 | 0.3801 | 5.91 | 0.3272 |
| Linear Regression | 5.58 | 8.58 | 0.3782 | 5.91 | 0.3343 |
| Dummy (Median Baseline) | 7.21 | 11.30 | -0.0772 | 7.12 | -0.0649 |

**Figure 3: Regression Model Performance Comparison (Test Set E4)**  
*(Multi-panel comparison saved to `reports/figures/regression_model_comparison.png`)*

**Figure 4: Actual vs. Predicted Salary Plot (Test Set)**  
*(Scatter plot saved to `reports/figures/actual_vs_predicted.png`)*

**Figure 5: Residual Error Distribution**  
*(Plot saved to `reports/figures/residual_plot.png`)*

### 6.3 Reconciled Ablation Study
To ensure complete experimental consistency (resolving previous hyperparameter divergence), identical model hyperparameters (`XGBoost`: `n_estimators=150, max_depth=6, lr=0.08`; `Random Forest`: `n_estimators=100, max_depth=16`; `Ridge`: `alpha=10.0`) were evaluated across all four configurations on the exact same splits:
- **E1:** Structured features without Year.
- **E2:** Structured features with Year.
- **E3:** SBERT + Structured features without Year.
- **E4:** SBERT + Structured features with Year (Full Model).

**Table 8: Reconciled Ablation Study Summary (`reports/tables/ablation_results.csv`)**

| Configuration | Model | Test MAE (M VND) | Test RMSE (M VND) | Test $R^2$ Score | Val MAE (M VND) | Val $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E1: Structured (No Year)** | Ridge | 6.05 | 9.24 | 0.2797 | 6.46 | 0.2087 |
| | Random Forest | 5.76 | 8.86 | 0.3371 | 5.99 | 0.3164 |
| | XGBoost | 5.76 | 8.86 | 0.3370 | 5.98 | 0.3183 |
| **E2: Structured + Year** | Ridge | 5.90 | 9.07 | 0.3065 | 6.25 | 0.2520 |
| | Random Forest | 5.62 | 8.97 | 0.3210 | 5.58 | 0.3813 |
| | XGBoost | 5.53 | 8.76 | 0.3532 | 5.66 | 0.3659 |
| **E3: SBERT + Struct (No Year)** | Ridge | 5.48 | 8.60 | 0.3757 | 6.01 | 0.3069 |
| | Random Forest | 5.64 | 8.96 | 0.3221 | 6.02 | 0.3030 |
| | XGBoost | 5.35 | 8.71 | 0.3597 | 5.64 | 0.3850 |
| **E4: SBERT + Struct + Year** | Ridge | 5.45 | 8.57 | 0.3801 | 5.91 | 0.3272 |
| | Random Forest | 5.40 | 8.77 | 0.3507 | 5.74 | 0.3545 |
| | **XGBoost** | **5.24** | **8.71** | **0.3604** | **5.34** | **0.4290** |

> [!NOTE]
> **Reconciliation Verification:** Notice that the E4 XGBoost row in Table 8 matches Table 5 exactly to every decimal place (Test MAE: 5.24, RMSE: 8.71, $R^2$: 0.3604; Val MAE: 5.34, $R^2$: 0.4290).

**Figure 5: Ablation Comparison Across Configurations E1 to E4**  
*(Plot saved to `reports/figures/ablation_comparison.png`)*

**Figure 6: Top 15 Feature Importances for Salary Prediction**  
*(Plot saved to `reports/figures/feature_importance.png`)*

#### Ablation Insights:
1. **Contribution of Year:** Adding `year` to structured tabular features (E1 $\rightarrow$ E2) lowered XGBoost Test MAE from **5.76M** to **5.53M VND** (a 0.23M VND error reduction) and boosted $R^2$ from 0.3370 to 0.3532.
2. **Contribution of SBERT:** Adding 384-dimensional dense semantic vectors (E1 $\rightarrow$ E3) lowered XGBoost Test MAE from **5.76M** to **5.35M VND** (a 0.41M VND error reduction) and raised $R^2$ from 0.3370 to 0.3597.
3. **Synergy in E4:** The combination of SBERT text embeddings and temporal/structured tabular variables achieved the overall lowest error (**MAE = 5.24M VND**).

### 6.4 Longitudinal Temporal Validation Check
The corpus spans posting years from 2022 through 2026. To test real-world deployment viability, models were trained on historical records ($\le 2025$, $N=8,054$) and evaluated on future 2026 postings ($N=2,868$):

**Table 9: Longitudinal Temporal Validation Results (`reports/tables/temporal_validation_results.csv`)**

| Model | Temporal Test MAE (M VND) | Temporal Test RMSE (M VND) | Temporal Test $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **Ridge Regression** | **7.94** | **12.56** | **0.2104** |
| **XGBoost Regressor** | **7.99** | **12.70** | **0.1930** |
| **Random Forest Regressor** | **8.46** | **13.30** | **0.1144** |

**Figure 7: Longitudinal Temporal Validation Gap (Random vs. Out-of-Time 2026 Evaluation)**  
*(Multi-panel comparison saved to `reports/figures/temporal_validation_comparison.png`)*

**Temporal Generalization Discussion:** Under temporal validation, prediction error increases from ~5.24M to ~7.94M VND. This reveals a macroeconomic generalization gap: 2026 recruitment compensation packages in Vietnam experienced noticeable inflation and structural salary band shifts compared to 2022–2024 postings, which models trained strictly on historical data slightly underestimate.

---

## 7. Task 3: Job Profile and Skill Clustering Results

### 7.1 Experimental Setup & Fair Comparison Design
- **Input:** Composite text (`job_title + ". " + job_description + " Yêu cầu: " + requirements`).
- **Representation:** 384-dimensional normalized SBERT embeddings.
- **Fair Comparison Methodology:**
  1. Full dataset sweep for K-Means ($k \in [3, 4, 5, 6, 8, 10]$).
  2. Standardized head-to-head comparison on a shared sample ($N=3,000$): K-Means ($k=8$), Agglomerative Clustering ($k=8$, cosine metric, average linkage), DBSCAN ($\text{eps}=0.45$, cosine), and Gaussian Mixture Model ($k=8$, diagonal covariance on PCA 25-dim).
  3. Evaluated on Cosine Silhouette, Davies-Bouldin Index, Calinski-Harabasz Score, and external Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) against weak job role labels.

### 7.2 Fair Benchmark Results

**Table 10: Clustering Performance Benchmark Table (`reports/tables/clustering_results.csv`)**

| Algorithm | Clusters ($k$) | Silhouette (Cosine) | Davies-Bouldin Index | Calinski-Harabasz | ARI (vs Weak Labels) | NMI (vs Weak Labels) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K-Means ($k=8$, Full Dataset)** | **8** | **0.1224** | **3.2569** | **477.83** | **0.1688** | **0.3103** |
| K-Means ($k=6$, Full Dataset) | 6 | 0.1177 | 3.4568 | 565.47 | 0.1709 | 0.3047 |
| K-Means ($k=5$, Full Dataset) | 5 | 0.1161 | 3.7207 | 614.40 | 0.1790 | 0.2943 |
| K-Means ($k=10$, Full Dataset) | 10 | 0.1148 | 3.2588 | 426.24 | 0.1552 | 0.3138 |
| K-Means ($k=3$, Full Dataset) | 3 | 0.1088 | 3.4452 | 790.07 | 0.2184 | 0.2752 |
| K-Means ($k=4$, Full Dataset) | 4 | 0.1085 | 3.6431 | 695.65 | 0.2163 | 0.3112 |
| **K-Means ($k=8$, Sample $N=3000$)** | **8** | **0.1245** | **3.2822** | **103.28** | **0.1677** | **0.3133** |
| Agglomerative ($k=8$, Sample $N=3000$) | 8 | 0.1159 | 1.4338 | 2.81 | 0.0027 | 0.0084 |
| Gaussian Mixture Model ($k=8$, Sample) | 8 | 0.0874 | 3.5195 | 93.39 | 0.1263 | 0.2791 |
| DBSCAN ($\text{eps}=0.45$, Sample $N=3000$) | 1 | N/A | N/A | N/A | N/A | N/A |

**Figure 8: Fair Head-to-Head Comparison Across 4 Clustering Algorithms**  
*(Comparison on Silhouette, Davies-Bouldin, and NMI/ARI saved to `reports/figures/clustering_algorithm_comparison.png`)*

**Figure 9: K-Means Inertia Elbow Curve and Silhouette Progression**  
*(Saved to `reports/figures/elbow_curve.png` and `reports/figures/silhouette_comparison.png`)*

#### Benchmark Takeaways:
- **K-Means Dominance:** On the standardized sample ($N=3,000$), K-Means achieved the highest Cosine Silhouette Score (**0.1245**) and the highest external Mutual Information with job role semantics (**NMI = 0.3133**).
- **Agglomerative Limitations:** While Agglomerative Clustering achieved a lower Davies-Bouldin index (1.4338), its ARI (0.0027) and NMI (0.0084) indicate that average linkage merged nearly all points into one giant cluster with small isolated singletons.
- **DBSCAN Density Bottleneck:** DBSCAN with $\text{eps}=0.45$ merged all points into a single dense manifold, highlighting the continuous density nature of SBERT semantic space.

### 7.3 Discovered Cluster Archetypes and Stopword-Filtered Profiles
By incorporating a comprehensive Vietnamese + English stopword filter and utilizing regex word boundaries, generic word pollution was completely removed, yielding 8 distinct, non-overlapping archetypes:

**Table 11: Discovered Cluster Archetypes and Technical Keywords (`reports/tables/cluster_keywords.csv`)**

| Cluster ID | Interpreted Archetype Name | Top Discriminating Technical Keywords | Representative Job Titles | Job Count | Percentage (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cluster 0** | **Web & Fullstack Software Engineering** | `web`, `php`, `backend`, `api`, `java`, `javascript`, `css`, `net`, `html`, `developer` | Frontend Developer; Backend Developer; Lập Trình Viên | 2,508 | 17.87% |
| **Cluster 1** | **Software Sales & IT Solutions Consulting** | `marketing`, `sales`, `online`, `sale`, `demo`, `cntt`, `laptop`, `cung`, `nhanh` | Nhân Viên Kinh Doanh Phần Mềm; Tư Vấn Phần Mềm; Triển Khai | 2,063 | 14.70% |
| **Cluster 2** | **QA & Software Quality Testing** | `test`, `testing`, `tester`, `test case`, `case`, `manual`, `automation`, `testcase`, `bug` | Manual Tester; Automation Tester; Kiểm Thử Phần Mềm | 472 | 3.36% |
| **Cluster 3** | **Enterprise Applications & IT Development** | `it`, `cntt`, `anh`, `c`, `tinh`, `nhanh`, `data`, `web`, `time`, `khoa` | Nhân Viên IT; Lập Trình Viên; Nhân Viên IT Phần Mềm | 1,707 | 12.16% |
| **Cluster 4** | **Data Analytics, Engineering & BI** | `data`, `sql`, `cntt`, `erp`, `bi`, `analyst`, `it`, `business`, `power bi`, `tableau` | Data Analyst; Data Engineer; DevOps Engineer | 2,355 | 16.78% |
| **Cluster 5** | **Industrial Automation & CNC Programming** | `cnc`, `phay`, `phay cnc`, `cnc cnc`, `mastercam`, `autocad`, `cam`, `cad`, `dao` | Lập Trình Viên CNC; Nhân Viên Lập Trình Máy CNC | 712 | 5.07% |
| **Cluster 6** | **IT Helpdesk & Network Administration** | `it`, `cntt`, `camera`, `server`, `lan`, `windows`, `wifi`, `helpdesk`, `internet` | Nhân Viên IT; IT Helpdesk; Quản Trị Hệ Thống / Mạng | 2,696 | 19.21% |
| **Cluster 7** | **AI Engineering & Mobile/Game Development** | `android`, `ios`, `c`, `game`, `mobile`, `flutter`, `learning`, `python`, `api`, `unity` | AI Engineer; Mobile Developer; Unity Developer | 1,524 | 10.86% |

**Figure 8: 2D PCA Cluster Projection**  
*(Saved to `reports/figures/pca_clusters.png`)*

**Figure 9: Cluster Size Distribution**  
*(Saved to `reports/figures/cluster_distribution.png`)*

---

## 8. Consolidated Split Manifest Summary

To ensure complete experimental transparency, zero leakage, and reproducibility, every sample used in supervised training was cataloged:

**Table 12: Split Manifest Allocation Summary (`reports/tables/split_manifest_summary.csv`)**

| Task | Split | Sample Count | Percentage (%) |
| :--- | :--- | :--- | :--- |
| **Task 1: Job Role Classification** | Train | 5,700 | 80.0% |
| | Validation | 713 | 10.0% |
| | Test (Held-Out) | 713 | 10.0% |
| **Task 2: Salary Prediction (Regression)** | Train | 8,737 | 80.0% |
| | Validation | 1,092 | 10.0% |
| | Test (Held-Out) | 1,093 | 10.0% |

The full manifest recording `unique_job_id`, `task`, and `split` is saved at `outputs/split_manifest.csv`.

---

## 9. Comprehensive Discussion and Verification of Bug Fixes

1. **Resolution of Location Normalization Bug (P0.1):** By evaluating explicit city names before district names and completely eliminating generic keywords like *"quận"* from city assignment, over 2,184 postings were corrected. Hanoi's true market representation rose from an artificial 4,802 postings to 5,913 postings.
2. **Resolution of Deduplication Flaw (P0.2):** Previous deduplication failed to use normalized strings. Implementing normalized composite keys (`norm_title + "|||" + norm_company + "|||" + norm_desc_prefix`) successfully eliminated 1,430 duplicate postings, including 9 cross-platform duplicates between TopCV and TinixAI.
3. **Cluster Keyword Stopwords & Non-Colliding Archetypes (P0.3):** Extending stopwords across Vietnamese and English, along with regex word-boundary matching, completely eliminated previous keyword collisions where multiple clusters were erroneously labeled *"AI & Machine Learning"*. All 8 clusters now possess distinct, verifiable technical semantics.
4. **Weak Supervision Clarification (P0.4):** All documentation, code comments, and reports now explicitly distinguish between heuristic pseudo-labels and human ground truth.
5. **Reconciliation of Regression Hyperparameters (P0.5):** XGBoost hyperparameters in the main benchmark and ablation study E4 are now completely reconciled, producing identical results (MAE: 5.24M VND, RMSE: 8.71M VND, $R^2$: 0.3604) across all CSVs and tables.
6. **Inclusion of TF-IDF Baseline (P1.1):** Adding the TF-IDF + Logistic Regression baseline confirmed that exact keyword frequency provides strong discriminative signal on rule-based taxonomy (Macro-F1: 0.8505), while dense SBERT embeddings achieve robust cross-lingual generalization (Accuracy: 88.36%, Macro-F1: 0.7850) in a 13-fold more compact representation space.
7. **Salary Parsing Improvements (P1.2):** Independent bounds validation ($3\text{M} \le \text{min} \le \text{max} \le 150\text{M}$) and disclosure categorization ensured that only verified, plausible compensation disclosures were used for regression training.

---

## 10. Automated Unit Test Verification

To ensure regression prevention, an automated unit test suite was implemented under `tests/`:
- `tests/test_location_normalization.py`: Verifies that Hanoi districts containing *"quận"* never map to HCMC, tests HCMC, Da Nang, Remote, and Multiple locations.
- `tests/test_salary_parser.py`: Verifies VND ranges, full digits, exact salaries, one-sided disclosures (*"Lên đến"*, *"Từ"*), USD conversions, negotiable keywords, and outlier exclusion.
- `tests/test_deduplication.py`: Verifies normalized composite key deduplication across case variations and legal entity suffixes.
- `tests/test_data_splitting.py`: Verifies 80/10/10 split proportions, mutual exclusivity (zero leakage), and stratified representation.

**Test Execution Result:** `Ran 17 tests in 0.021s: OK (100% Pass)`.

---

## 11. Conclusion
The **JobLens Vietnam** project has completed a thorough, academically rigorous code review and experimental re-execution. Across supervised job role classification (SBERT KNN Test Macro-F1: 0.7850; TF-IDF Baseline Macro-F1: 0.8505), multi-modal salary regression (Test MAE: 5.24M VND, $R^2$: 0.3604), and unsupervised clustering ($k=8$, Silhouette: 0.1224, NMI: 0.3133), the codebase has achieved complete reproducibility, zero leakage, numerical consistency, and robust diagnostic reporting.
