# JobLens Vietnam: Initial Results on Applying Machine Learning Models to Recruitment Data

**Course:** Introduction to Data Science  
**Academic Checkpoint:** Initial Results on Applying ML Models  
**Project:** JobLens Vietnam - Labor Market Analysis & Skill Trends in Vietnam IT & AI Sectors  

---

## Abstract
This report presents the initial empirical results of the **JobLens Vietnam** project, an end-to-end data science study examining the Vietnamese Information Technology (IT), Artificial Intelligence (AI), and Data Science job recruitment market. Using a newly integrated corpus of 14,307 deduplicated job postings from Vietnamese recruitment platforms (TinixAI and TopCV), we develop and benchmark machine learning models across three core tasks:
1. **Supervised Job Role Classification** from unstructured job descriptions via dense multilingual Sentence-BERT (SBERT) embeddings across 8 distinct career categories;
2. **Supervised Salary Prediction** combining 384-dimensional SBERT text embeddings with structured job attributes (experience, location, position, employment type, and year), accompanied by a four-configuration ablation study;
3. **Unsupervised Job and Skill Clustering** exploring natural recruitment archetypes using K-Means, Agglomerative Hierarchical Clustering, DBSCAN, and Gaussian Mixture Models (GMM).

Our empirical findings demonstrate that dense semantic representations provide substantial predictive leverage over traditional heuristics. In classification, K-Nearest Neighbors and Linear SVM achieved Macro-F1 scores of **0.7759** and **0.7525** with overall accuracies exceeding **88%**. In salary regression, combining SBERT embeddings with tabular features and the mandatory `year` feature reduced Mean Absolute Error (MAE) from a median baseline of **6.93 Million VND** down to **5.24 Million VND** ($R^2 = 0.3683$) via an XGBoost regressor. In clustering, K-Means ($k=10$, Silhouette = **0.1368**) successfully isolated specialized archetypes such as AI/ML, Data Analytics, QA/Testing, and Hardware/Infrastructure.

---

## 1. Introduction and Research Objectives

### 1.1 Motivation
The digital transformation in Vietnam has led to rapid growth in technical hiring across Software Engineering, Data Analytics, Data Engineering, and Artificial Intelligence. However, recruitment data posted on online job boards is characterized by high noise, inconsistent titles, multi-language descriptions (Vietnamese, English, and code-mixed formats), and unstandardized salary expressions. 

The primary goal of the **JobLens Vietnam** project is to formulate an end-to-end data science pipeline that moves from raw web scraping to data cleaning, semantic representation learning, supervised classification, multi-modal regression, unsupervised clustering, and an interactive decision-support interface for job seekers and educational institutions.

### 1.2 Core Research Tasks
- **Task 1: Job Role Classification (Supervised Multiclass):** Predict the target job role label ($y \in \{\text{Software Engineer, Data Analyst / BI, Data Scientist, Data Engineer, AI / ML Engineer, QA / Tester, DevOps / Cloud, IT Support / Network / Security}\}$) using only the unstructured text of `job_description`.
- **Task 2: Salary Prediction (Supervised Regression):** Predict the monthly salary in VND (`salary_mean_vnd`) by combining textual embeddings and tabular job characteristics (`job_role_label`, `exp_years`, `norm_location`, `norm_position`, `norm_job_type`, and `year_clean`), rigorously measuring the incremental contribution of the mandatory `year` feature and textual embeddings via an ablation study.
- **Task 3: Job and Skill Clustering (Unsupervised Learning):** Uncover natural groupings of technical skills and job profiles without relying on predefined labels, using composite representations (`job_title + job_description + requirements`).

---

## 2. Dataset Description and Schema Standardization

### 2.1 Raw Data Ingestion
The raw data consists of two distinct CSV files located in `data/raw/`:
- `tinixai_it_ai_data_jobs.csv`: 13,964 raw job postings with preliminary matched groups.
- `topcv_jobs.csv`: 1,503 raw job postings scraped from TopCV Vietnam.

Both datasets were inspected prior to merging. The schemas were aligned to a unified 16-column layout:
- Primary Identifiers: `unique_job_id` (prefixed with source), `source_id`, `source` (`tinixai` vs `topcv`).
- Job Attributes: `job_title`, `company_name`, `salary`, `location`, `job_type`, `job_industry`.
- Qualifications & Requirements: `experience_level`, `education_level`, `job_position`, `requirements`, `benefits`.
- Textual Description: `job_description`.
- Temporal Metadata: `year`.

### 2.2 Deduplication
Job postings frequently suffer from duplicate repostings. Deduplication was executed across composite normalized keys (`job_title`, `company_name`, `location`, and the first 100 characters of `job_description`). Out of 15,467 initial merged records, 1,160 redundant postings were removed, yielding a clean master set of **14,307 unique job postings** saved to `data/processed/merged_jobs.csv`.

---

## 3. Data Integration and Preprocessing

### 3.1 Text Cleaning Pipeline
A robust cleaning function was implemented in `src/data/cleaner.py`:
- HTML tags (`<br>`, `<div>`, `<li>`) were stripped using regex.
- HTML character entities (`&amp;`, `&nbsp;`) were unescaped.
- Unicode normalization (NFKC) was applied to reconcile accented Vietnamese characters.
- **Critical Token Preservation:** High-value programming tokens such as `C++`, `C#`, `.NET`, `Python`, `SQL`, `AWS`, `CI/CD`, `Docker`, `PyTorch`, and `Vue.js` were explicitly preserved by avoiding blanket punctuation removal.

### 3.2 Salary Parsing and Normalization
Salary information was parsed using regular expressions in `src/data/salary_parser.py`:
1. **Unspecified Handling:** Expressions indicating non-numeric negotiations (*"Thỏa thuận"*, *"Thương lượng"*, *"Đang cập nhật"*, *"Negotiable"*) were flagged as `unspecified` and excluded from regression training (never imputed with median).
2. **Currency Conversion:** USD salaries were converted to VND using a documented reference rate of $1\text{ USD} = 25,000\text{ VND}$ (early 2026 baseline).
3. **Unit Scaling:** Values stated in millions (*"triệu"*, *"tr"*, *"M"*) were scaled to VND ($10^6$).
4. **Range Averaging:** Lower and upper bounds were extracted, setting $\text{salary\_mean\_vnd} = (\text{min} + \text{max}) / 2$.
5. **Plausibility Bounds:** Monthly salaries outside the realistic range $[3,000,000\text{ VND}, 150,000,000\text{ VND}]$ were flagged as outliers.

The audit distribution of salary parsing is summarized in **Table 1**.

**Table 1: Salary Parsing Audit Diagnostic (`reports/tables/salary_parsing_report.csv`)**

| Parsing Status | Count | Percentage (%) | Action / Description |
| :--- | :--- | :--- | :--- |
| **valid** | **11,292** | **78.93%** | Retained for regression training |
| unspecified | 2,507 | 17.52% | Excluded (negotiable / unstated) |
| implausible_low | 207 | 1.45% | Filtered (< 3M VND/month) |
| invalid_numbers | 143 | 1.00% | Filtered (malformed strings) |
| missing | 112 | 0.78% | Filtered (null fields) |
| implausible_high | 28 | 0.20% | Filtered (> 150M VND/month) |
| no_numbers_found | 18 | 0.13% | Filtered |
| **Total** | **14,307** | **100.0%** | **11,292 valid training samples** |

### 3.3 Job Role Label Generation
To establish ground truth for classification, hierarchical regular expression rules were applied to `job_title` in `src/data/label_generator.py`. Specific roles took precedence over general ones:
1. `AI / ML Engineer`
2. `Data Scientist`
3. `Data Engineer`
4. `Data Analyst / BI`
5. `QA / Tester`
6. `DevOps / Cloud`
7. `IT Support / Network / Security`
8. `Software Engineer`

Postings mapped to unassigned or non-technical categories were marked `Other / Unassigned`. To prevent target leakage, `job_title` was strictly excluded from the classification input space, leaving only `job_description`. The resulting classification dataset comprises **7,184 verified records** across 8 classes (`data/processed/classification_data.csv`).

**Table 2: Ground Truth Job Role Label Distribution (`reports/tables/label_distribution.csv`)**

| Job Role Label | Count | Percentage (%) | Notes |
| :--- | :--- | :--- | :--- |
| Other / Unassigned | 7,123 | 49.79% | General IT / Management (excluded from classification) |
| **Software Engineer** | **3,960** | **27.68%** | Web, Mobile, Fullstack, Backend |
| **IT Support / Network / Security** | **1,208** | **8.44%** | Helpdesk, Systems, Infrastructure |
| **Data Analyst / BI** | **533** | **3.73%** | BI Analyst, Reporting, Power BI, SQL |
| **DevOps / Cloud** | **469** | **3.28%** | SRE, Cloud Architecture, CI/CD |
| **QA / Tester** | **452** | **3.16%** | Manual / Automation QC, Test Engineer |
| **AI / ML Engineer** | **274** | **1.92%** | Deep Learning, NLP, Computer Vision |
| **Data Engineer** | **202** | **1.41%** | ETL, Big Data, Pipeline Architectures |
| **Data Scientist** | **86** | **0.60%** | Statistical Modeling, Predictive Analysis |
| **Total Valid Technical Roles** | **7,184** | **100.0%** | **Included in Task 1 modeling** |

---

## 4. Sentence-BERT (SBERT) Embedding Methodology

### 4.1 Model Selection
We selected `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. This 12-layer multilingual Transformer maps sentences and paragraphs into a 384-dimensional dense vector space. Key reasons for this selection:
- **Multilingual Support:** Trained on parallel corpora covering Vietnamese and English, essential for IT job descriptions that mix Vietnamese prose with English technical jargon.
- **Computational Efficiency:** 12-layer MiniLM architecture allows rapid batch inference on standard CPUs (~55-200 texts/sec) without requiring GPU acceleration.
- **Unit Normalization:** Embeddings are normalized such that dot products equal cosine similarities.

### 4.2 Encoding and Alignment
Two separate embedding sets were generated and persisted as `.npy` matrices in `embeddings/`:
1. `sbert_embeddings.npy` $(14307 \times 384)$: Encoded solely from `job_description`, utilized for Classification and Regression.
2. `sbert_clustering_embeddings.npy` $(14307 \times 384)$: Encoded from composite text (`job_title + ". " + job_description + " Yêu cầu: " + requirements`), utilized for Unsupervised Clustering.

Index alignment is verified using `unique_job_id` keys tracked in `embeddings/embedding_metadata.json`.

---

## 5. Task 1: Job Role Classification Results

### 5.1 Experimental Setup
- **Input:** 384-dimensional SBERT embeddings of `job_description`.
- **Target:** 8-class `job_role_label`.
- **Split:** Stratified 80/20 train/test split (random seed 42, $N_{\text{train}} = 5,747$, $N_{\text{test}} = 1,437$).
- **Primary Metric:** Macro-F1 (to equitably measure performance across imbalanced classes).

### 5.2 Performance Comparison
We benchmarked 5 algorithm families: Logistic Regression, Linear SVM, K-Nearest Neighbors (KNN with cosine metric), Random Forest, and XGBoost Classifier.

**Table 3: Classification Model Comparison (`reports/tables/classification_results.csv`)**

| Model | Accuracy | Precision (Macro) | Recall (Macro) | Macro-F1 | Weighted-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **K-Nearest Neighbors** | **0.8845** | **0.7966** | **0.7600** | **0.7759** | **0.8825** |
| XGBoost Classifier | 0.8824 | 0.8395 | 0.7107 | 0.7625 | 0.8769 |
| Linear SVM | 0.8685 | 0.7308 | 0.7790 | 0.7525 | 0.8712 |
| Logistic Regression | 0.8281 | 0.6692 | 0.7864 | 0.7164 | 0.8375 |
| Random Forest | 0.8253 | 0.8697 | 0.5819 | 0.6751 | 0.8109 |

**Figure 1: Classification Models Comparison**  
*(Visual comparison available in `reports/figures/model_comparison.png`)*

### 5.3 Per-Class Analysis and Confusion Matrix
Examining the per-class performance of the best model (KNN with cosine metric) reveals high fidelity across core domains:

**Table 4: Per-Class Classification Report for Best Model (`reports/tables/classification_report.csv`)**

| Class | Precision | Recall | F1-Score | Support (Test Set) |
| :--- | :--- | :--- | :--- | :--- |
| Software Engineer | 0.93 | 0.94 | **0.93** | 792 |
| IT Support / Network / Security | 0.86 | 0.91 | **0.88** | 242 |
| Data Analyst / BI | 0.84 | 0.86 | **0.85** | 107 |
| QA / Tester | 0.83 | 0.86 | **0.84** | 90 |
| Data Engineer | 0.81 | 0.75 | **0.78** | 40 |
| DevOps / Cloud | 0.81 | 0.66 | **0.73** | 94 |
| AI / ML Engineer | 0.76 | 0.69 | **0.72** | 55 |
| Data Scientist | 0.54 | 0.41 | **0.47** | 17 |

**Figure 2: Normalized Confusion Matrix**  
*(Heatmap saved to `reports/figures/confusion_matrix.png`)*

The confusion matrix indicates that `Software Engineer` and `IT Support` have distinct, highly separable lexical profiles. The lower score for `Data Scientist` (F1 = 0.47) stems primarily from its small sample size ($N_{\text{test}} = 17$) and lexical overlap with `AI / ML Engineer` and `Data Analyst / BI`.

---

## 6. Task 2: Salary Prediction (Regression) Results

### 6.1 Experimental Setup
- **Target:** Monthly salary in Million VND (`salary_mean_vnd / 1,000,000`).
- **Samples:** 11,292 valid records, 80/20 train/test split ($N_{\text{train}} = 9,033$, $N_{\text{test}} = 2,259$).
- **Features:** 384-dimensional SBERT text embeddings combined with One-Hot Encoded categorical variables (`job_role_label`, `norm_location`, `norm_position`, `norm_job_type`) and standardized numerical variables (`exp_years`, `year_clean`).

### 6.2 Full Model Comparison (Configuration E4)
We evaluated 6 regression algorithms on the complete multi-modal feature set (E4):

**Table 5: Regression Performance on Configuration E4 (`reports/tables/regression_results.csv`)**

| Model | MAE (Million VND) | RMSE (Million VND) | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **XGBoost Regressor** | **5.24** | **8.67** | **0.3683** |
| Extra Trees Regressor | 5.29 | 8.88 | 0.3386 |
| Random Forest Regressor | 5.48 | 9.05 | 0.3119 |
| Ridge Regression | 5.59 | 8.94 | 0.3288 |
| Linear Regression | 5.77 | 9.01 | 0.3177 |
| Dummy (Median Baseline) | 6.93 | 11.23 | -0.0590 |

**Figure 3: Actual vs. Predicted Salary Plot**  
*(Scatter plot saved to `reports/figures/actual_vs_predicted.png`)*

**Figure 4: Residual Distribution and Error Analysis**  
*(Plot saved to `reports/figures/residual_plot.png`)*

The XGBoost model outperforms the median baseline by reducing MAE from **6.93M VND** down to **5.24M VND**, achieving an $R^2$ of **0.3683**.

### 6.3 Required Ablation Study
To rigorously quantify the individual and synergistic contributions of text embeddings and the mandatory `year` feature, we conducted four controlled ablation experiments across identical train/test splits:
- **E1:** Structured features only (without Year).
- **E2:** Structured features with Year.
- **E3:** SBERT embeddings + Structured features (without Year).
- **E4:** SBERT embeddings + Structured features with Year (Full Model).

**Table 6: Comprehensive Ablation Study Results (`reports/tables/ablation_results.csv`)**

| Configuration | Model | MAE (Million VND) | RMSE (Million VND) | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- |
| **E1: Structured (No Year)** | Ridge | 6.06 | 9.54 | 0.2364 |
| | Random Forest | 5.81 | 9.38 | 0.2616 |
| | XGBoost | 5.80 | 9.30 | 0.2736 |
| **E2: Structured + Year** | Ridge | 5.97 | 9.38 | 0.2620 |
| | Random Forest | 5.65 | 9.16 | 0.2959 |
| | XGBoost | 5.56 | 8.99 | 0.3217 |
| **E3: SBERT + Struct (No Year)**| Ridge | 5.63 | 8.99 | 0.3218 |
| | Random Forest | 5.67 | 9.17 | 0.2935 |
| | XGBoost | 5.41 | 8.90 | 0.3354 |
| **E4: SBERT + Struct + Year** | Ridge | 5.59 | 8.94 | 0.3288 |
| | Random Forest | 5.55 | 9.09 | 0.3063 |
| | **XGBoost** | **5.34** | **8.81** | **0.3480** |

**Figure 5: Ablation Comparison Across Configurations E1 to E4**  
*(Comparison bar chart saved to `reports/figures/ablation_comparison.png`)*

#### Ablation Insights:
1. **Contribution of Year:** Incorporating `year` into tabular features (moving from E1 to E2) improved XGBoost $R^2$ from **0.2736** to **0.3217** and lowered MAE from **5.80M** to **5.56M VND**. This confirms significant temporal shifts in compensation packages across posting years.
2. **Contribution of SBERT:** Adding dense 384-dimensional text embeddings (moving from E1 to E3) reduced MAE by **0.39M VND** and boosted $R^2$ by **6.2%**, proving that qualitative nuances in job descriptions contain valuable compensation signals.
3. **Synergy in E4:** The combination of SBERT semantic vectors and temporal/structured features achieves the highest predictive accuracy across all configurations.

**Figure 6: Top 15 Feature Importances for Salary Prediction**  
*(Feature importance chart saved to `reports/figures/feature_importance.png`)*  
Key drivers identified by the tree ensemble include `exp_years`, `norm_position_Quản lý / Leader`, `norm_location_Hồ Chí Minh`, `job_role_label_AI / ML Engineer`, and specific high-weight SBERT semantic dimensions.

---

## 7. Task 3: Job and Skill Clustering Results

### 7.1 Experimental Setup
- **Input:** Composite representation (`job_title + job_description + requirements`).
- **Representation:** 384-dimensional normalized SBERT embeddings.
- **Algorithms:** K-Means ($k \in [3, 4, 5, 6, 8, 10]$), Agglomerative Hierarchical (cosine linkage), DBSCAN ($\text{eps}=0.45$), and Gaussian Mixture Model (GMM on 25 PCA dimensions).
- **Metrics:** Cosine Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score.

### 7.2 Clustering Evaluation
**Table 7: Clustering Algorithm Benchmark (`reports/tables/clustering_results.csv`)**

| Algorithm | Clusters ($k$) | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Score |
| :--- | :--- | :--- | :--- | :--- |
| K-Means ($k=3$) | 3 | 0.1122 | 3.4378 | 813.74 |
| K-Means ($k=4$) | 4 | 0.1121 | 3.6346 | 714.26 |
| K-Means ($k=5$) | 5 | 0.1170 | 3.7046 | 629.41 |
| K-Means ($k=6$) | 6 | 0.1200 | 3.4601 | 579.95 |
| K-Means ($k=8$) | 8 | 0.1084 | 3.2895 | 489.39 |
| **K-Means ($k=10$)** | **10** | **0.1368** | **3.1640** | **437.16** |
| Agglomerative ($k=10$)| 10 | 0.1184 | 1.5705 | 3.27 |
| DBSCAN ($\text{eps}=0.45$)| 1 | N/A (noise threshold) | N/A | N/A |
| GMM ($k=10$) | 10 | 0.0868 | 3.3338 | 403.51 |

**Figure 7: K-Means Elbow Curve and Silhouette Progression**  
*(Saved to `reports/figures/elbow_curve.png` and `reports/figures/silhouette_comparison.png`)*

### 7.3 Discovered Cluster Archetypes and Interpretation
To interpret the 10 discovered clusters without predefined bias, we extracted the top 10 TF-IDF keywords for each cluster:

**Table 8: Cluster Archetype Interpretation & Top Keywords (`reports/tables/cluster_keywords.csv`)**

| Cluster ID | Interpreted Archetype | Top Discriminating Keywords | Job Postings Count |
| :--- | :--- | :--- | :--- |
| **Cluster 0** | **QA / QC & Testing** | `test`, `kiểm thử`, `thử`, `kiểm`, `testing`, `tester`, `automation` | 471 |
| **Cluster 2** | **Network & System Hardware**| `máy`, `mạng`, `hệ thống`, `thiết bị`, `cài đặt`, `phần cứng` | 2,725 |
| **Cluster 3** | **IT Sales & Business Consulting**| `khách hàng`, `kinh doanh`, `tư vấn`, `bán hàng`, `dịch vụ` | 2,120 |
| **Cluster 4** | **AI & Machine Learning** | `ai`, `learning`, `mô hình`, `python`, `deep learning`, `trí tuệ nhân tạo` | 631 |
| **Cluster 5** | **Mobile & Game Development** | `android`, `ios`, `game`, `mobile`, `ứng dụng`, `flutter`, `react native` | 630 |
| **Cluster 6** | **CAD / CNC Embedded Programming**| `cnc`, `gia công`, `máy`, `bản vẽ`, `lập trình`, `phay` | 683 |
| **Cluster 9** | **Data Analytics & BI** | `dữ liệu`, `data`, `phân tích`, `sql`, `báo cáo`, `power bi`, `bi` | 969 |
| Clusters 1,7,8| General Software / Development | `phát triển`, `phần mềm`, `backend`, `java`, `spring`, `api`, `web` | 6,078 |

**Figure 8: 2D PCA Cluster Projection**  
*(Saved to `reports/figures/pca_clusters.png`)*

**Figure 9: Cluster Size Distribution**  
*(Saved to `reports/figures/cluster_distribution.png`)*

The unsupervised clustering accurately grouped technical roles into coherent sub-industries: Machine Learning (Cluster 4), Data Analytics (Cluster 9), Testing (Cluster 0), Mobile/Game (Cluster 5), and System/Hardware (Cluster 2) separated clearly from general software engineering.

---

## 8. Discussion and Key Findings

1. **SBERT vs Traditional Bag-of-Words:** Dense SBERT multilingual vectors capture cross-lingual synonyms seamlessly (e.g., mapping *"lập trình viên"* and *"software engineer"* into adjacent vector regions). This enables high classification accuracy (88.45%) using solely job descriptions.
2. **Predictability of Salary:** Monthly salary in Vietnam IT recruitment has a predictable structural floor dictated by experience level and location (Hanoi vs HCMC vs Remote). However, high residual variance ($R^2 \approx 0.37$) reflects the prevalent practice of unstated variable compensation (performance bonuses, 13th-month salary, stock options).
3. **Role of Temporal Features:** The mandatory `year` feature significantly improved regression performance, corroborating post-pandemic market recalibration in tech hiring budgets.
4. **Natural Skill Partitioning:** Unsupervised clustering proved that data roles naturally split into analytical (BI/Data Analyst) and algorithmic (AI/ML) clusters, confirming that market demand distinguishes business-oriented reporting from mathematical modeling.

---

## 9. Limitations

1. **Unstated Salaries:** 17.52% of job postings listed salaries as *"Thỏa thuận"*, introducing potential survivorship bias toward mid-level positions that disclose wage ranges.
2. **Class Imbalance in Specialized Roles:** While Software Engineers accounted for 27.68% of postings, Data Scientists represented only 0.60% (86 postings). This limited test samples for rare roles ($N=17$), reducing the empirical stability of their per-class metrics.
3. **Noisy Text Elements:** Certain postings contained employer branding boilerplate (*"Môi trường năng động, teambuilding hàng quý"*), adding non-functional noise to textual representations.

---

## 10. Future Work

For the final project submission, the following enhancements are planned:
1. **Vietnamese Language Fine-Tuning:** Evaluate domain-specific pretrained encoders such as PhoBERT or ViDeBERTa.
2. **Imbalance Mitigation:** Incorporate Focal Loss or SMOTE oversampling in embedding space.
3. **Interactive Streamlit Dashboard:** Deploy the trained models (`best_classification_model.joblib`, `best_regression_model.joblib`, `best_clustering_model.joblib`) into an interactive web application allowing users to paste a JD to get role predictions, estimated salary ranges, and recommended skills.

---

## 11. Conclusion
The **JobLens Vietnam** project has successfully completed its initial milestone by establishing a clean, reproducible, end-to-end data science pipeline on over 14,000 real-world recruitment postings. Across supervised classification (Macro-F1: 0.7759), multi-modal salary regression (MAE: 5.24M VND), and unsupervised clustering ($k=10$, Silhouette: 0.1368), the experimental results validate the efficacy of SBERT representations and provide a solid empirical foundation for final project delivery.

---
*Report generated and validated on 14,307 real records from `data/raw/` in the JobLens Vietnam workspace.*
