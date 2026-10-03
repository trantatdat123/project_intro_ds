# JobLens Vietnam 🎯
> **Course Project:** Introduction to Data Science (IntroDS)  
> **Repository:** `project_intro_ds`  
> **Topic:** Vietnam Tech & Data Job Market Intelligence, Skill Extraction, and Predictive Modeling Pipeline.

---

## 📖 1. Project Overview

### 1.1. Context & Problem Statement
The Information Technology (IT) and Data ecosystem (Data Science, Data Analytics, Data Engineering, AI/ML) in Vietnam is experiencing rapid growth, creating an unprecedented demand for skilled professionals. However, job postings scraped from recruitment platforms (e.g., TopCV, VietnamWorks, ITviec, CareerViet) present significant challenges for job seekers, recruiters, and market researchers:

1. **Unstructured and Noisy Text:**
   - Raw Job Descriptions (JDs) are stored as unstructured text contaminated with HTML boilerplate tags (`<p>`, `<div>`, `<script>`), broken character encodings, and inconsistent code-switching between Vietnamese and English.
2. **Fragmented and Opaque Salary Information:**
   - Compensation is reported in inconsistent currencies and patterns: VND, USD, millions/month, ranges (`"15 - 30 triệu"`, `"20-25M"`, `"Up to $2000"`), or labeled vaguely as `"Thỏa thuận"` (Negotiable). This makes empirical salary benchmarking difficult without systematic normalization.
3. **Job Role Ambiguity:**
   - Employers often use non-standardized job titles for equivalent responsibilities (e.g., *"Senior BI Analyst"*, *"Data Mining Specialist"*, *"Big Data Engineer"*, *"AI Research Specialist"*), creating friction in macro-level role categorization.
4. **Latent Technical Skills:**
   - Crucial technical requirements (e.g., Python, SQL, Docker, PyTorch, AWS, Spark) are buried across narrative requirement sections without structured taxonomy tags, making it difficult to gauge market demand programmatically.

---

### 1.2. Project Objectives
**JobLens Vietnam** is developed as an **End-to-End Data Science Pipeline** that automates the transition from raw web scrapes to actionable market insights and production-grade machine learning models:

* **Automated Preprocessing & Normalization:** Transform messy job posts into structured datasets by parsing salary ranges into standardized VND values, extracting minimum required experience in years, and mapping arbitrary job titles into **5 core industry archetypes** (`Data Analyst`, `Data Scientist`, `Data Engineer`, `AI/ML Engineer`, `Software Engineer`).
* **High-Speed Skill Extraction Engine:** Implement a dictionary-backed FlashText / Aho-Corasick algorithm paired with a canonical synonym map to reliably detect 150+ technology keywords without word-boundary errors.
* **Applied Machine Learning & Explainable Modeling:**
  - **Role Classification:** Automatically classify raw job descriptions into standardized job roles using text representations (TF-IDF and dense embeddings).
  - **Salary Prediction & Marginal Impact Analysis:** Predict average salaries using tree-based regression (LightGBM / Random Forest) on log-transformed salaries, coupled with **SHAP (SHapley Additive exPlanations)** to answer: *What is the marginal salary premium associated with specific skills (e.g., Docker, Spark, PyTorch) or additional years of experience?*
  - **Skill Network & Market Clustering:** Construct a Skill Co-occurrence Graph using NetworkX to identify complementary technology clusters, and apply semantic clustering (UMAP + K-Means / HDBSCAN) to discover latent job posting archetypes.

---

### 1.3. Practical Impact & Target Audience
* **Students & Job Seekers:** Gain empirical market transparency, identify the highest-paying technical skills, and negotiate compensation with confidence backed by data.
* **Recruiters & Companies:** Benchmark compensation packages against market standards and optimize JD phrasing for better candidate targeting.
* **Educational Institutions:** Align academic curricula and training programs with real-world tech stack requirements.

---

## 📁 2. Project Directory Structure

```text
project_intro_ds/
├── configs/
│   └── skill_dictionary.json     # Standardized skills & synonym dictionary
├── data/
│   ├── raw/                      # Raw scraped job posting datasets
│   └── processed/                # Cleaned, standardized datasets ready for modeling
├── notebooks/                    # Jupyter notebooks for EDA & experimental analysis
├── src/
│   ├── crawler/                  # TopCV, CareerViet, VietnamWorks crawlers
│   │   ├── topcv_crawler.py      # Core TopCVCrawler class
│   │   ├── careerviet_crawler.py # CareerVietCrawler: live + Wayback
│   │   ├── vietnamworks_crawler.py # VietnamWorksCrawler: live + Wayback
│   │   └── run_crawl.py          # CLI runner script
│   ├── preprocessing/            # Data cleaning & normalization pipeline
│   │   ├── cleaner.py            # HTML removal, text normalization, experience parser
│   │   ├── salary_parser.py      # Salary range parsing, currency conversion, outlier removal
│   │   ├── skill_extractor.py    # FlashText keyword extraction engine
│   │   └── role_normalizer.py    # Rule-based mapping to 5 core roles
│   ├── features/
│   │   └── build_features.py     # Binary skill matrix & sentence embeddings
│   └── models/
│       ├── classifier.py         # Role classification models & evaluation
│       ├── regressor.py          # Salary prediction & SHAP feature explanation
│       └── cluster.py            # Skill co-occurrence network & job clustering
├── requirements.txt
├── README.md
└── plan.md
```

---

## ⚙️ 3. Core Functions Specification

### 3.1. Data Preprocessing (`src/preprocessing/`)
* **`cleaner.py`**:
  - `clean_html(raw_html: str) -> str`: Strips HTML tags (`<p>`, `<div>`, `<script>`, `<style>`) and unescapes HTML entities.
  - `normalize_text(text: str) -> str`: Normalizes Vietnamese Unicode (NFC), strips redundant newlines, punctuation artifacts, and extra whitespace.
  - `parse_experience(raw_exp: str) -> float`: Extracts numeric minimum required experience in years (e.g., `"2 - 3 năm"` $\rightarrow$ `2.0`, Fresher $\rightarrow$ `0.0`).
* **`salary_parser.py`**:
  - `parse_salary_range(raw_salary: str, usd_to_vnd_rate: float = 25500.0) -> Dict[str, Optional[float]]`: Parses salary text into `salary_min`, `salary_max`, `salary_mean` (in VND), currency, and `is_negotiable` flag.
  - `remove_salary_outliers(df, salary_col: str = "salary_mean", lower_quantile=0.01, upper_quantile=0.99) -> pd.DataFrame`: Filters anomalous salary records using quantile boundaries / IQR.
* **`skill_extractor.py`**:
  - Class `SkillExtractor(skill_dict_path: str)`: Loads a JSON mapping of canonical skills and aliases into a FlashText `KeywordProcessor`.
  - `extract_skills(text: str) -> List[str]`: Extracts a deduplicated list of standardized skills present in the text.
* **`role_normalizer.py`**:
  - `map_standard_role(job_title: str) -> str`: Maps unstructured titles into 5 standard categories: `Data Analyst`, `Data Scientist`, `Data Engineer`, `AI/ML Engineer`, `Software Engineer` (or `Other`).

### 3.2. Feature Engineering (`src/features/`)
* **`build_features.py`**:
  - `create_skill_matrix(df: pd.DataFrame, skill_col="extracted_skills", min_frequency=10) -> pd.DataFrame`: Generates a binary multi-hot matrix (`skill_<name>`) for skills with occurrence frequency $\ge$ `min_frequency`.
  - `generate_text_embeddings(texts: List[str], model_name="paraphrase-multilingual-MiniLM-L12-v2") -> np.ndarray`: Produces dense semantic embeddings `(n_samples, embedding_dim)` from JD text using multilingual Sentence-Transformers.

### 3.3. Machine Learning Modeling (`src/models/`)
* **`classifier.py`**:
  - `train_role_classifier(X_train, y_train, model_type="tfidf_logistic") -> Pipeline`: Trains a multiclass role classifier (`tfidf_logistic`, `tfidf_svm`, `embedding_mlp`).
  - `evaluate_classifier(model, X_test, y_test) -> Dict[str, Any]`: Computes Macro-F1, Weighted-F1, Precision, Recall, and Confusion Matrix.
* **`regressor.py`**:
  - `train_salary_regressor(X_train, y_train, model_name="lightgbm")`: Fits regression models (`ridge`, `random_forest`, `lightgbm`) to predict log-salary $\log(\text{salary\_mean})$.
  - `evaluate_regressor(model, X_test, y_test) -> Dict[str, float]`: Transforms predictions back to original VND scale and evaluates MAE, RMSE, and $R^2$.
  - `explain_salary_features(model, X_sample, feature_names: List[str])`: Computes SHAP values to quantify the contribution of individual skills and experience.
* **`cluster.py`**:
  - `build_skill_cooccurrence_matrix(skill_lists: List[List[str]]) -> pd.DataFrame`: Builds a symmetric co-occurrence frequency matrix across all job posts.
  - `create_skill_network(cooccurrence_df: pd.DataFrame, min_edge_weight=15) -> nx.Graph`: Generates a weighted NetworkX graph and calculates Degree Centrality for skill clustering.
  - `cluster_job_embeddings(embeddings: np.ndarray, n_clusters=5, use_umap=True) -> Tuple[np.ndarray, np.ndarray]`: Performs 2D UMAP projection and clusters job postings via K-Means or HDBSCAN.

---

## 🚀 4. Installation & Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows PowerShell
# source .venv/bin/activate # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt
```

### Crawl VietnamWorks

Mở `notebooks/03_vietnamworks_crawler.ipynb` và chạy lần lượt. Mặc định crawl 50 URL IT/Data hiện tại mỗi lượt, một luồng bằng `curl_cffi`, có checkpoint. Chọn `MODE = "history"` hoặc `"both"` để lấy snapshot Wayback; xem `src/crawler/README.md` về nguồn ngày đăng và giới hạn dữ liệu lịch sử.

---
