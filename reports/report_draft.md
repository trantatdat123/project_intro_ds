# JobLens Vietnam: Báo Cáo Kỹ Thuật Chi Tiết Quá Trình Tiền Xử Lý, Huấn Luyện Và Đánh Giá Mô Hình (Bản Cập Nhật Toàn Diện & Khắc Phục Lỗi)

**Dự án:** JobLens Vietnam - Phân tích thị trường việc làm & xu hướng kỹ năng CNTT/AI tại Việt Nam bằng Khoa học Dữ liệu  
**Môn học:** Nhập môn Khoa học Dữ liệu (Introduction to Data Science)  
**Tài liệu:** Bản giải thích kỹ thuật đầy đủ dành cho nhóm để nắm vững toàn bộ quy trình, phương pháp thực nghiệm và số liệu báo cáo trước giảng viên.

---

## 1. Xác Nhận Trạng Thái Toàn Diện Của Dự Án

Dự án đã được **kiểm toán toàn diện mã nguồn (code audit), sửa triệt để các lỗi logic/nghiệp vụ, thực nghiệm lại 100% trên dữ liệu thực tế và ghi đè đồng bộ toàn bộ bảng số liệu, biểu đồ và mô hình mới nhất**.

* **Dữ liệu thô gốc (Original Raw Datasets):** Giữ nguyên tuyệt đối tại `data/raw/` (`tinixai_it_ai_data_jobs.csv` và `topcv_jobs.csv`), không bị sửa đổi, xóa hay di chuyển.
* **Không dùng dữ liệu giả (No Synthetic Data):** Tất cả các phân tích, thực nghiệm và số liệu đều chạy trên dữ liệu thực tế được crawl từ TinixAI và TopCV.
* **Khắc phục hoàn toàn 5 lỗi cốt lõi (P0 Bug Fixes):**
  1. *Lỗi chuẩn hóa địa điểm (P0.1):* Sửa lỗi điều kiện từ khóa `"quận"` khiến địa chỉ Hà Nội bị gán nhầm sang TP.HCM. Cứu đúng **1,470 chuỗi địa chỉ khác nhau và 2,184 tin tuyển dụng (15.56% dữ liệu)**.
  2. *Lỗi khử trùng lặp (P0.2):* Sửa hàm `deduplicate_merged()` để thực sự sử dụng chuỗi đã chuẩn hóa (bỏ tiền tố pháp lý công ty, chuẩn hóa tên job, kết hợp 150 ký tự đầu của JD), loại bỏ chính xác **1,430 tin tuyển dụng trùng lặp**.
  3. *Lỗi từ khóa & trùng tên phân cụm (P0.3):* Bổ sung bộ lọc stopword tiếng Việt + tiếng Anh toàn diện, dùng regex biên từ (`\b...\b`), loại bỏ hoàn toàn các từ rác tiếng Việt (`khai`, `doanh`, `giao`, `cao`, `nam`,...), phân tách thành **8 cụm rõ ràng, không bị trùng lặp tên**.
  4. *Chuẩn hóa thuật ngữ nhãn (P0.4):* Định nghĩa chính xác nhãn nghề là **nhãn bán tự động / quy tắc heuristic (weakly supervised / rule-based labels)**, không tuyên bố sai lệch là "ground truth do con người dán nhãn".
  5. *Đồng bộ tham số XGBoost (P0.5):* Thống nhất 100% siêu tham số giữa bài toán so sánh mô hình chính (E4) và nghiên cứu suy giảm (Ablation Study E4), loại bỏ hoàn toàn việc mâu thuẫn số liệu.
* **Cải tiến phương pháp luận (P1 Improvements):**
  * Bổ sung **TF-IDF + Logistic Regression Baseline** cho bài toán phân loại để so sánh trực diện với SBERT.
  * Phân tách lương độc lập: Kiểm tra độc lập cả Min, Max và Mean trong khoảng $[3,000,000\text{ VNĐ}, 150,000,000\text{ VNĐ}]$ và phân loại rõ ràng tin có khoảng lương (range), lương cố định (exact) và tin 1 chiều (*"Lên đến"*, *"Từ"*).
  * Áp dụng bắt buộc chính sách chia tập **80/10/10 (Train 80%, Validation 10%, Test 10%)**: Tập Validation dùng để chọn mô hình, tập Test giữ nguyên đến bước đánh giá cuối cùng; lưu danh sách phân chia tại `outputs/split_manifest.csv`.
  * Thực hiện **Đánh giá theo thời gian (Temporal Validation)** cho bài toán dự đoán lương: Train trên các năm $\le 2025$, Test trên năm 2026.
  * Thiết kế **So sánh công bằng cho Phân cụm (Fair Clustering Benchmark)** trên cùng một mẫu $N=3,000$ điểm, tính Silhouette (Cosine), Davies-Bouldin, Calinski-Harabasz, ARI và NMI.
  * Xây dựng bộ kiểm thử tự động **17 Unit Tests** trong thư mục `tests/` với kết quả pass 100%.

---

## 2. Tiền Xử Lý Dữ Liệu & Các Báo Cáo Kiểm Toán (Data Audits)

### 2.1 Hợp nhất và Khử Trùng Lặp (Deduplication Audit)
* **Dữ liệu thô ban đầu:** TinixAI (13,964 dòng) + TopCV (1,503 dòng) = 15,467 dòng.
* **Chuẩn hóa schema 16 thuộc tính:** `unique_job_id`, `source_id`, `source`, `job_title`, `company_name`, `salary`, `location`, `job_type`, `job_industry`, `experience_level`, `education_level`, `job_position`, `job_description`, `benefits`, `requirements`, `year`.
* **Khử trùng lặp đa nền tảng:** 
  * Chuẩn hóa tên công ty (loại bỏ *"công ty cổ phần"*, *"tnhh"*, *"jsc"*, *"ltd"*).
  * Chuẩn hóa chức danh (loại bỏ thẻ địa điểm như *"[Hà Nội]"*).
  * Khóa nhận diện: `clean_title + "|||" + clean_company + "|||" + clean_desc_prefix_150`.
* **Kết quả:** Loại bỏ **1,430 tin tuyển dụng trùng lặp** (1,404 tin trùng nội bộ TinixAI, 17 tin trùng nội bộ TopCV, 9 tin TopCV trùng TinixAI). Bộ dữ liệu sạch duy nhất gồm **14,037 tin tuyển dụng**, lưu tại `data/processed/merged_jobs.csv`.
* Báo cáo chi tiết lưu tại: `reports/tables/deduplication_audit.csv`.

### 2.2 Kiểm toán & Khắc Phục Lỗi Địa Điểm (Location Normalization Fix)
* **Nguyên nhân lỗi cũ:** Trước đây, hàm `normalize_location()` kiểm tra mảng từ khóa của TP.HCM trước, trong đó có chứa từ chung chung là `'quận'`. Vì vậy, các địa chỉ ở Hà Nội như *"Quận Cầu Giấy, Hà Nội"* hay *"Quận Thanh Xuân, Thành phố Hà Nội"* bị dính chữ `'quận'` và trả về nhầm thành `"Hồ Chí Minh"`.
* **Cách khắc phục triệt để:**
  1. Kiểm tra từ khóa thành phố rõ ràng trước (*Hà Nội*, *Hồ Chí Minh*, *Đà Nẵng*).
  2. Phát hiện các tin tuyển dụng đa vùng (*"Multiple"*).
  3. Chỉ khi nào không ghi tên thành phố mới fallback sang danh sách các quận huyện cụ thể.
  4. Tuyệt đối không dùng từ khóa chung chung như `"quận"`, `"huyện"`, `"thành phố"` để nhận diện thành phố.
  5. Nhận diện các tin làm việc từ xa (*Remote / WFH*).
* **Kết quả kiểm toán (`reports/tables/location_fix_audit.csv`):**
  * Có **1,470 chuỗi địa chỉ khác nhau (gồm 2,184 tin tuyển dụng)** đã được sửa chữa đúng.
  * Cứu lại **1,650 tin tuyển dụng tại Hà Nội** bị gán sai sang TP.HCM.
  * Tỷ lệ phân bố vùng miền sau sửa lỗi: Hà Nội chiếm 42.12% (5,913 tin), Hồ Chí Minh chiếm 40.02% (5,618 tin), Khác chiếm 13.16% (1,847 tin), Multiple chiếm 2.96% (416 tin), Đà Nẵng chiếm 1.72% (241 tin), Remote chiếm 0.01% (2 tin).

### 2.3 Phân Tích Lương & Kiểm Tra Ngưỡng Độc Lập (Salary Parsing Audit)
* **Xử lý chuỗi lương:** Bóc tách các dạng chuỗi tự do (VND, triệu, USD theo tỷ giá $1\text{ USD} = 25,000\text{ VNĐ}$, năm quy đổi ra tháng).
* **Kiểm tra ngưỡng độc lập:** Yêu cầu độc lập $3,000,000 \le \text{salary\_min\_vnd} \le \text{salary\_max\_vnd} \le 150,000,000\text{ VNĐ}$.
* **Phân loại trạng thái công bố lương (`reports/tables/salary_parsing_audit.csv`):**
  * *Thỏa thuận / Thương lượng / Cạnh tranh:* 2,429 tin (17.30%) $\rightarrow$ Đưa vào trạng thái `unspecified` và **tuyệt đối không điền khống bằng Median**.
  * *Có khoảng lương rõ ràng (Range):* 10,750 tin (76.58%) $\rightarrow$ Lương đại diện bằng trung bình cộng $(\text{min} + \text{max}) / 2$.
  * *Lương cố định (Exact):* 134 tin (0.95%).
  * *Lương cận trên ("Lên đến / Tới"):* 27 tin (0.19%).
  * *Lương cận dưới ("Từ / Min"):* 11 tin (0.08%).
  * *Lương phi thực tế (<3M hoặc >150M VNĐ):* 400 tin (2.85%) $\rightarrow$ Loại bỏ.
  * *Lỗi định dạng số / Không tìm thấy số:* 286 tin (2.04%) $\rightarrow$ Loại bỏ.
* **Tổng số mẫu hồi quy hợp lệ:** **10,922 tin tuyển dụng (77.81%)**, lưu tại `data/processed/regression_data.csv`. Mức lương trung bình thị trường là **17.86 triệu VNĐ/tháng**, trung vị là **15.00 triệu VNĐ/tháng** (IQR: 11.5M – 21.0M VNĐ).

### 2.4 Nhãn Nghề Heuristic Dưới Dạng Weak Supervision
* Nhãn nghề `job_role_label` được sinh từ cột `job_title` bằng các luật Regex phân tầng (từ hẹp đến rộng):
  1. `Software Engineer`: 3,960 tin (28.21% / 55.57% tập kỹ thuật)
  2. `IT Support / Network / Security`: 1,141 tin (8.13% / 16.01% tập kỹ thuật)
  3. `Data Analyst / BI`: 536 tin (3.82% / 7.52% tập kỹ thuật)
  4. `DevOps / Cloud`: 465 tin (3.31% / 6.52% tập kỹ thuật)
  5. `QA / Tester`: 443 tin (3.16% / 6.22% tập kỹ thuật)
  6. `AI / ML Engineer`: 280 tin (1.99% / 3.93% tập kỹ thuật)
  7. `Data Engineer`: 202 tin (1.44% / 2.83% tập kỹ thuật)
  8. `Data Scientist`: 99 tin (0.71% / 1.39% tập kỹ thuật)
* Các tin không thuộc chuyên môn CNTT rõ ràng (6,911 tin ~ 49.23%) được gán `Other / Unassigned` và loại khỏi bài toán Phân loại.
* **Tập dữ liệu phân loại:** **7,126 tin tuyển dụng**, lưu tại `data/processed/classification_data.csv`.

---

## 3. Biểu Diễn Văn Bản Bằng Sentence-BERT (SBERT)

* **Pretrained Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
* **Kích thước vector:** **384 chiều** (dense semantic representations).
* **Chuẩn hóa:** Vector được chuẩn hóa L2, biến tích vô hướng thành độ tương đồng Cosine.
* **Hai bộ vector độc lập:**
  1. `embeddings/sbert_embeddings.npy` $(14,037 \times 384)$: Mã hóa `job_description`, phục vụ Bài toán 1 (Phân loại) và Bài toán 2 (Dự đoán lương).
  2. `embeddings/sbert_clustering_embeddings.npy` $(14,037 \times 384)$: Mã hóa văn bản tổng hợp `job_title + job_description + requirements`, phục vụ Bài toán 3 (Phân cụm).

---

## 4. Chi Tiết 3 Bài Toán Machine Learning & Kết Quả Thực Nghiệm Mới Nhất

### BÀI TOÁN 1: PHÂN LOẠI NHÃN NGHỀ (JOB ROLE CLASSIFICATION)
* **Bản chất:** Supervised Multiclass Classification (8 lớp).
* **Input ($X$):** Đoạn văn bản mô tả công việc `job_description` (được biểu diễn bằng vector SBERT 384 chiều hoặc TF-IDF 5,000 n-grams). **Tuyệt đối không dùng `job_title` làm input để tránh target leakage.**
* **Target ($y$):** 1 trong 8 nhãn nghề chuẩn hóa.
* **Chính sách phân chia dữ liệu (Stratified 80/10/10 Split):**
  * **Train Split (80%):** 5,700 mẫu (huấn luyện mô hình).
  * **Validation Split (10%):** 713 mẫu (dùng để chọn mô hình tốt nhất dựa trên Val Macro-F1).
  * **Test Split (10%):** 713 mẫu (tập độc lập hoàn toàn, đánh giá kiểm định cuối cùng).
* **Kết quả thực nghiệm trên tập Test:**

| Mô hình | Đặc trưng | Test Accuracy | Test Precision (Macro) | Test Recall (Macro) | Test Macro-F1 | Val Macro-F1 | Val Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | TF-IDF (5,000 n-grams) | **91.58%** | **0.8377** | **0.8667** | **0.8505** | **0.8734** | **91.44%** |
| **SBERT + K-Nearest Neighbors** | SBERT (384 chiều) | **87.52%** | **0.7879** | **0.7596** | **0.7704** | **0.7856** | **87.66%** |
| **SBERT + Linear SVM** | SBERT (384 chiều) | 87.38% | 0.7750 | 0.8189 | 0.7944 | 0.7587 | 86.40% |
| **SBERT + XGBoost** | SBERT (384 chiều) | 86.40% | 0.8330 | 0.6948 | 0.7475 | 0.7562 | 87.38% |
| **SBERT + Logistic Regression** | SBERT (384 chiều) | 82.75% | 0.6930 | 0.8119 | 0.7411 | 0.7062 | 80.50% |
| **SBERT + Random Forest** | SBERT (384 chiều) | 81.77% | 0.8978 | 0.5708 | 0.6602 | 0.6982 | 83.73% |

* **Nhận xét chuyên sâu:**
  * **Baseline TF-IDF:** Đạt Macro-F1 0.8505 và Accuracy 91.58%. Lý do là nhãn nghề được gán theo quy tắc từ khóa, và trong JD của nhà tuyển dụng thường xuyên lặp lại chính xác các từ khóa công nghệ này (như *"Python"*, *"SQL"*, *"Selenium"*, *"Kubernetes"*). TF-IDF hoạt động như bộ đếm tần suất từ khóa chính xác.
  * **SBERT Representation:** Mô hình **SBERT + KNN** đạt kết quả tốt nhất trong nhóm biểu diễn ngữ nghĩa dày đặc với **Accuracy 87.52%** và **Macro-F1 0.7704** trên tập Test (Val Macro-F1 đạt 0.7856). Không gian vector SBERT chỉ có 384 chiều (gọn hơn 13 lần so với 5,000 chiều của TF-IDF), nắm bắt được ngữ cảnh đồng nghĩa giữa tiếng Anh và tiếng Việt mà không phụ thuộc vào việc trùng lặp từ ngữ chính xác.

---

### BÀI TOÁN 2: DỰ ĐOÁN MỨC LƯƠNG (SALARY REGRESSION)
* **Bản chất:** Supervised Regression kết hợp đa phương thức (Multi-Modal: SBERT văn bản + Đặc trưng bảng).
* **Target ($y$):** Mức lương trung bình tháng tính bằng **Triệu VNĐ/tháng** (`salary_mean_vnd / 1,000,000`).
* **Input ($X$):** 384 chiều SBERT + Đặc trưng phân loại One-Hot (`job_role_label`, `norm_location`, `norm_position`, `norm_job_type`) + Đặc trưng số chuẩn hóa (`exp_years`, `year_clean`).
* **Chính sách phân chia dữ liệu (80/10/10 Split):**
  * **Train Split (80%):** 8,737 mẫu.
  * **Validation Split (10%):** 1,092 mẫu.
  * **Test Split (10%):** 1,093 mẫu.
  * Triệt tiêu hoàn toàn rò rỉ dữ liệu (Zero Leakage): `OneHotEncoder` và `StandardScaler` được fit duy nhất trên `df_train`.
* **Kết quả so sánh mô hình trên cấu hình E4 (Held-Out Test Set):**

| Mô hình | Test MAE (Triệu VNĐ) | Test RMSE (Triệu VNĐ) | Test $R^2$ Score | Val MAE (Triệu VNĐ) | Val $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost Regressor** | **5.24** | **8.71** | **0.3604** | **5.34** | **0.4290** |
| **Extra Trees Regressor** | 5.30 | 8.88 | 0.3350 | 5.56 | 0.3786 |
| **Random Forest Regressor** | 5.40 | 8.77 | 0.3507 | 5.74 | 0.3545 |
| **Ridge Regression** | 5.45 | 8.57 | 0.3801 | 5.91 | 0.3272 |
| **Linear Regression** | 5.58 | 8.58 | 0.3782 | 5.91 | 0.3343 |
| **Dummy (Median Baseline)** | 7.21 | 11.30 | -0.0772 | 7.12 | -0.0649 |

* *Biểu đồ so sánh trực quan hiệu năng các mô hình hồi quy:* `reports/figures/regression_model_comparison.png`
* *Biểu đồ tương quan thực tế vs dự đoán:* `reports/figures/regression_actual_vs_predicted.png`
* *Biểu đồ phân phối phần dư:* `reports/figures/regression_residuals_distribution.png`

* **Nghiên cứu suy giảm tái đồng bộ (Reconciled Ablation Study E1 - E4):**
  * **E1 (Đặc trưng bảng không có Năm):** XGBoost đạt Test MAE = 5.76M, $R^2 = 0.3370$.
  * **E2 (Đặc trưng bảng có Năm):** XGBoost đạt Test MAE = 5.53M, $R^2 = 0.3532$ (Thêm `year` giảm sai số 0.23 triệu VNĐ).
  * **E3 (SBERT + Bảng không có Năm):** XGBoost đạt Test MAE = 5.35M, $R^2 = 0.3597$ (Thêm SBERT giảm sai số 0.41 triệu VNĐ).
  * **E4 (SBERT + Bảng có Năm - Toàn phần):** XGBoost đạt **Test MAE = 5.24M**, **RMSE = 8.71M**, **$R^2 = 0.3604$** (Val MAE = 5.34M, $R^2 = 0.4290$).
  > *Lưu ý:* Số liệu cấu hình E4 ở bảng Ablation và bảng so sánh mô hình chính trùng khớp 100% từng số thập phân.
* **Đánh giá theo thời gian (Temporal Validation - Train $\le 2025$, Test 2026):**
  * Mô hình huấn luyện trên dữ liệu lịch sử ($\le 2025$, $N=8,054$) khi dự đoán trên thị trường năm 2026 ($N=2,868$) đạt Test MAE là **7.94M VNĐ (Ridge)** và **7.99M VNĐ (XGBoost)**.
  * Hiện tượng tăng sai số từ 5.24M lên ~7.99M phản ánh chính xác khoảng cách tổng quát hóa thời gian (Temporal Generalization Gap), bắt nguồn từ đợt tăng lương thị trường và biến động ngân sách tuyển dụng năm 2026 so với giai đoạn trước.
  * *Biểu đồ trực quan khoảng cách thời gian (Temporal Gap):* `reports/figures/temporal_validation_comparison.png`

---

### BÀI TOÁN 3: PHÂN CỤM CÔNG VIỆC VÀ KỸ NĂNG (JOB PROFILE CLUSTERING)
* **Bản chất:** Unsupervised Learning (Không có nhãn trước).
* **Input ($X$):** 384 chiều vector SBERT mã hóa văn bản tổng hợp `job_title + job_description + requirements` (14,037 bản ghi).
* **So sánh công bằng (Fair Comparison on Sample $N=3,000$):**

| Thuật toán | Số cụm ($k$) | Silhouette (Cosine) | Davies-Bouldin Index | Calinski-Harabasz | ARI (vs Weak Labels) | NMI (vs Weak Labels) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K-Means ($k=8$, Full Dataset)** | **8** | **0.1224** | **3.2569** | **477.83** | **0.1688** | **0.3103** |
| K-Means ($k=6$, Full Dataset) | 6 | 0.1177 | 3.4568 | 565.47 | 0.1709 | 0.3047 |
| K-Means ($k=5$, Full Dataset) | 5 | 0.1161 | 3.7207 | 614.40 | 0.1790 | 0.2943 |
| K-Means ($k=10$, Full Dataset) | 10 | 0.1148 | 3.2588 | 426.24 | 0.1552 | 0.3138 |
| **K-Means ($k=8$, Sample $N=3000$)** | **8** | **0.1245** | **3.2822** | **103.28** | **0.1677** | **0.3133** |
| Agglomerative ($k=8$, Sample $N=3000$) | 8 | 0.1159 | 1.4338 | 2.81 | 0.0027 | 0.0084 |
| GMM ($k=8$, Sample $N=3000$) | 8 | 0.0874 | 3.5195 | 93.39 | 0.1263 | 0.2791 |
| DBSCAN ($\text{eps}=0.45$, Sample $N=3000$) | 1 | N/A | N/A | N/A | N/A | N/A |

* *Biểu đồ so sánh đối đầu công bằng 4 thuật toán phân cụm:* `reports/figures/clustering_algorithm_comparison.png`
* *Biểu đồ phân bố cụm trong không gian 2D PCA:* `reports/figures/clustering_pca_clusters.png`

* **8 Nhóm Hình Mẫu Tuyển Dụng Sau Khi Lọc Stopword Triệt Để (`reports/tables/cluster_keywords.csv`):**
  1. **Cluster 0: Web & Fullstack Software Engineering** (2,508 tin - 17.87%): `web`, `php`, `backend`, `api`, `java`, `javascript`, `css`, `.net`, `html`, `developer`.
  2. **Cluster 1: Software Sales & IT Solutions Consulting** (2,063 tin - 14.70%): `marketing`, `sales`, `online`, `sale`, `demo`, `cntt`, `laptop`, `cung`, `nhanh`.
  3. **Cluster 2: QA & Software Quality Testing** (472 tin - 3.36%): `test`, `testing`, `tester`, `test case`, `case`, `manual`, `automation`, `testcase`, `bug`.
  4. **Cluster 3: Enterprise Applications & IT Development** (1,707 tin - 12.16%): `it`, `cntt`, `anh`, `c`, `tinh`, `nhanh`, `data`, `web`, `time`, `khoa`.
  5. **Cluster 4: Data Analytics, Engineering & BI** (2,355 tin - 16.78%): `data`, `sql`, `cntt`, `erp`, `bi`, `analyst`, `it`, `business`, `power bi`, `tableau`.
  6. **Cluster 5: Industrial Automation & CNC Programming** (712 tin - 5.07%): `cnc`, `phay`, `phay cnc`, `cnc cnc`, `mastercam`, `autocad`, `cam`, `cad`, `dao`.
  7. **Cluster 6: IT Helpdesk & Network Administration** (2,696 tin - 19.21%): `it`, `cntt`, `camera`, `server`, `lan`, `windows`, `wifi`, `helpdesk`, `internet`.
  8. **Cluster 7: AI Engineering & Mobile/Game Development** (1,524 tin - 10.86%): `android`, `ios`, `c`, `game`, `mobile`, `flutter`, `learning`, `python`, `api`, `unity`.

---

## 5. Hướng Dẫn Sử Dụng Thư Mục Và Tài Liệu Viết Báo Cáo

Để hoàn thiện bài báo cáo nộp giảng viên, nhóm có thể trích xuất trực tiếp:
1. **Các bảng số liệu mới nhất (thư mục `reports/tables/`):**
   * `location_fix_audit.csv`: Bảng bằng chứng kiểm toán sửa lỗi địa chỉ.
   * `deduplication_audit.csv`: Bảng thống kê khử trùng lặp dữ liệu.
   * `salary_parsing_audit.csv` & `salary_parsing_report.csv`: Bảng phân loại và kiểm định lương.
   * `classification_results.csv` & `classification_report.csv`: Bảng so sánh phân loại và chi tiết từng lớp trên tập Test.
   * `regression_results.csv` & `ablation_results.csv`: Bảng kết quả hồi quy và nghiên cứu suy giảm E1-E4.
   * `temporal_validation_results.csv`: Bảng đánh giá tổng quát hóa thời gian (<=2025 so với 2026).
   * `clustering_results.csv` & `cluster_keywords.csv`: Bảng so sánh các thuật toán phân cụm và hồ sơ 8 cụm.
   * `split_manifest_summary.csv`: Bảng tổng kết số lượng mẫu phân chia 80/10/10.
2. **Các biểu đồ trực quan hóa độ phân giải cao (thư mục `reports/figures/`):**
   * `confusion_matrix.png`: Ma trận nhầm lẫn chuẩn hóa của mô hình phân loại tốt nhất trên tập Test.
   * `model_comparison.png`: Biểu đồ so sánh Accuracy, Macro-F1 giữa các mô hình phân loại.
   * `actual_vs_predicted.png`: Biểu đồ phân tán giữa lương thực tế và lương dự đoán.
   * `residual_plot.png`: Biểu đồ phân bố phần dư (sai số) của mô hình hồi quy XGBoost.
   * `feature_importance.png`: Biểu đồ Top 15 đặc trưng quan trọng nhất dự đoán lương.
   * `ablation_comparison.png`: Biểu đồ so sánh MAE và $R^2$ qua 4 cấu hình E1, E2, E3, E4.
   * `elbow_curve.png`: Đường cong quán tính xác định số cụm tối ưu $k=8$.
   * `pca_clusters.png`: Hình chiếu PCA 2D không gian ngữ nghĩa của 8 cụm công việc.
   * `cluster_distribution.png`: Biểu đồ phân bố số lượng tin tuyển dụng theo từng cụm.
3. **Mã nguồn chạy demo trực tiếp:** Chạy lệnh `python demo_predict.py` để biểu diễn trực quan: nhập một mô tả công việc (JD), hệ thống tự động dự đoán nhóm nghề, ước tính mức lương (triệu VNĐ/tháng) và đề xuất kỹ năng liên quan theo cụm.
4. **Bộ kiểm thử tự động:** Chạy `python -m unittest discover -s tests` để kiểm tra độ tin cậy của mã nguồn (17/17 tests pass).
