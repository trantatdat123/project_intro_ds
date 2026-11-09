# JobLens Vietnam: Báo Cáo Giải Thích Chi Tiết Quá Trình Tiền Xử Lý, Huấn Luyện Và Kết Quả Mô Hình (Report Draft)

**Dự án:** JobLens Vietnam - Phân tích thị trường việc làm & kỹ năng CNTT/AI tại Việt Nam bằng Khoa học Dữ liệu  
**Môn học:** Nhập môn Khoa học Dữ liệu (Introduction to Data Science)  
**Tài liệu:** Bản giải thích kỹ thuật chi tiết dành cho thành viên nhóm nắm vững dự án để báo cáo vấn đáp với giảng viên.

---

## 1. Xác nhận tình trạng dự án

Toàn bộ dự án đã được **tiền xử lý, trích xuất đặc trưng bằng Sentence-BERT (SBERT), huấn luyện (train) và đánh giá (test) hoàn tất 100% trên tập dữ liệu thực tế**.
* **Dữ liệu thô gốc:** Giữ nguyên 100% tại `data/raw/` (`tinixai_it_ai_data_jobs.csv` và `topcv_jobs.csv`), không bị chỉnh sửa hay xóa bỏ.
* **Không dùng dữ liệu giả (No synthetic data):** Toàn bộ kết quả đều lấy từ 14,307 tin tuyển dụng thực tế.
* **Đầy đủ artifacts:** Toàn bộ model tốt nhất đã lưu dưới dạng `.joblib`, toàn bộ biểu đồ lưu ở `outputs/**/plots/` và `reports/figures/`, toàn bộ bảng số liệu lưu ở `reports/tables/`.

---

## 2. Tiền xử lý & Hợp nhất dữ liệu (Preprocessing & Data Integration)

### 2.1 Hợp nhất 2 nguồn dữ liệu (Merge Data)
* **Nguồn 1:** `tinixai_it_ai_data_jobs.csv` gồm 13,964 dòng, 17 cột.
* **Nguồn 2:** `topcv_jobs.csv` gồm 1,503 dòng, 14 cột.

**Cách merge cột:**
Chúng ta không thể nối đơn thuần vì cột `id` ở 2 nguồn có thể trùng nhau (ví dụ cả 2 file đều có dòng ID = 1). Giải pháp được lập trình trong `src/data/loader.py`:
1. **Chuẩn hóa về 16 cột chung:** `unique_job_id`, `source_id`, `source`, `job_title`, `company_name`, `salary`, `location`, `job_type`, `job_industry`, `experience_level`, `education_level`, `job_position`, `job_description`, `benefits`, `requirements`, `year`.
2. **Tạo khóa định danh toàn cục (`unique_job_id`):** Gắn thêm tiền tố nguồn:
   * TinixAI: `tinixai_1`, `tinixai_2`, ...
   * TopCV: `topcv_2309792`, `topcv_2309569`, ...
   * Lưu `source` tương ứng là `'tinixai'` và `'topcv'`.
3. **Khử trùng lặp (Deduplication):** 
   * Trên các trang tuyển dụng, một tin thường được công ty đăng lại nhiều lần ở các ngày khác nhau (Job repost).
   * Tạo khóa kiểm tra trùng lặp: `job_title` (viết thường) + `company_name` (viết thường) + `location` (viết thường) + 100 ký tự đầu của `job_description`.
   * Kết quả: Tổng số 15,467 dòng ban đầu sau khi khử trùng lặp còn lại **14,307 tin tuyển dụng duy nhất** (loại bỏ 1,160 tin trùng). File lưu tại: `data/processed/merged_jobs.csv`.

---

### 2.2 Làm sạch dữ liệu văn bản (Text Cleaning)
Mã nguồn đặt tại `src/data/cleaner.py`:
* **Xóa thẻ HTML:** Loại bỏ các thẻ `<br>`, `<div>`, `<p>`, `<li>`,...
* **Giải mã HTML entities:** Đổi `&amp;` thành `&`, `&nbsp;` thành khoảng trắng, `&quot;` thành `"`.
* **Chuẩn hóa Unicode:** Áp dụng chuẩn Unicode NFKC để thống nhất bộ gõ tiếng Việt (tránh lỗi tổ hợp vs dựng sẵn).
* **BẢO TOÀN KỸ THUẬT QUAN TRỌNG (Critical Rule):**
  * Không dùng hàm xóa dấu câu bừa bãi.
  * Các từ khóa công nghệ mang tính quyết định như: `C++`, `C#`, `.NET`, `CI/CD`, `Node.js`, `Vue.js`, `Python`, `SQL`, `AWS`, `PyTorch` được bảo tồn nguyên vẹn.

---

### 2.3 Bóc tách & Chuẩn hóa Lương (`src/data/salary_parser.py`)
Dữ liệu lương gốc ở dạng chuỗi tự do rất phức tạp: *"15 - 25 triệu"*, *"20.000.000 - 30.000.000 VND"*, *"1500 USD"*, *"Tới 40 triệu"*, *"15000000 - 25000000"*, *"Thỏa thuận"*, *"Đang cập nhật"*.

**Quy tắc xử lý:**
1. **Tin không có số cụ thể:** Các tin chứa *"Thỏa thuận"*, *"Thương lượng"*, *"Đang cập nhật"*, *"Negotiable"* (chiếm 2,507 tin ~ 17.52%) được gán trạng thái `unspecified` và **tuyệt đối KHÔNG điền khống bằng Median**.
2. **Quy đổi ngoại tệ:** Nếu phát hiện `USD` hoặc `$`, quy đổi theo tỷ giá chuẩn $1\text{ USD} = 25,000\text{ VND}$.
3. **Quy đổi đơn vị:** Nhận diện từ khóa *"triệu"*, *"tr"*, *"M"* để nhân với $10^6$.
4. **Tính lương đại diện:** 
   * Trích xuất `salary_min_vnd` và `salary_max_vnd`.
   * Mức lương mục tiêu cho bài toán hồi quy:
     $$\text{salary\_mean\_vnd} = \frac{\text{salary\_min\_vnd} + \text{salary\_max\_vnd}}{2}$$
5. **Lọc Outlier bất hợp lý:** Các mức lương ngoài khoảng $[3,000,000\text{ VND}, 150,000,000\text{ VND}]$ (như gõ nhầm 0 đồng hoặc vài tỷ) bị loại bỏ.
* **Kết quả:** Có **11,292 tin tuyển dụng (78.93%)** có mức lương số hợp lệ, lưu vào `data/processed/regression_data.csv`.

---

### 2.4 Chuẩn hóa Nhãn Nghề (`src/data/label_generator.py`)
Từ cột `job_title`, chúng ta ánh xạ theo biểu thức chính quy (Regex) có thứ tự ưu tiên từ hẹp tới rộng:
1. `AI / ML Engineer` (AI, Machine Learning, Deep Learning, NLP, Computer Vision) $\rightarrow$ 274 tin.
2. `Data Scientist` (Khoa học dữ liệu, Data Scientist) $\rightarrow$ 86 tin.
3. `Data Engineer` (Kỹ sư dữ liệu, Big Data, ETL, Data Pipeline) $\rightarrow$ 202 tin.
4. `Data Analyst / BI` (Data Analyst, Business Analyst, BI, Power BI, Tableau) $\rightarrow$ 533 tin.
5. `QA / Tester` (Tester, QA, QC, Kiểm thử, Automation Test) $\rightarrow$ 452 tin.
6. `DevOps / Cloud` (DevOps, SRE, Cloud, AWS, Azure, Sysadmin) $\rightarrow$ 469 tin.
7. `IT Support / Network / Security` (IT Helpdesk, Quản trị mạng, An ninh mạng) $\rightarrow$ 1,208 tin.
8. `Software Engineer` (Lập trình viên, Developer, Backend, Frontend, Fullstack, Mobile) $\rightarrow$ 3,960 tin.
* Các tin không thuộc chuyên môn kỹ thuật hoặc quá chung chung (7,123 tin) được xếp vào `Other / Unassigned` và loại khỏi bài toán Classification để tránh nhãn rác.
* **Tập dữ liệu phân loại:** **7,184 bản ghi** thuộc 8 nhóm nghề, lưu tại `data/processed/classification_data.csv`.

---

## 3. Biểu diễn Văn Bản bằng Sentence-BERT (SBERT)

Thay vì dùng TF-IDF truyền thống (chỉ đếm từ), đề tài sử dụng mô hình học sâu ngôn ngữ:
* **Pretrained Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
* **Lý do chọn:** Hỗ trợ đa ngôn ngữ (Multilingual), xử lý cực tốt các văn bản tuyển dụng tại Việt Nam (kết hợp cả câu tiếng Việt và thuật ngữ kỹ thuật tiếng Anh).
* **Kích thước vector:** **384 chiều** (dense vectors).
* **Chuẩn hóa (Normalization):** Vector được chuẩn hóa L2 (độ dài = 1.0), giúp tích vô hướng trực tiếp bằng độ tương đồng Cosine (Cosine Similarity).
* **Hai bộ vector độc lập:**
  1. `embeddings/sbert_embeddings.npy` $(14307 \times 384)$: Mã hóa riêng `job_description`.
  2. `embeddings/sbert_clustering_embeddings.npy` $(14307 \times 384)$: Mã hóa văn bản tổng hợp `job_title + job_description + requirements`.

---

## 4. Chi tiết 3 bài toán Machine Learning

### BÀI TOÁN 1: PHÂN LOẠI JOB ROLE (CLASSIFICATION)
* **Mục tiêu:** Nhận một bản mô tả công việc (JD) mới, hệ thống tự động đoán xem đây là nghề gì.
* **Input ($X$):** Vector SBERT 384 chiều của văn bản `job_description`.
* **Output ($y$):** 1 trong 8 nhãn nghề (*Software Engineer, Data Analyst, AI Engineer,...*).
* **LƯU Ý CỰC KỲ QUAN TRỌNG:**
  * **Tại sao KHÔNG dùng `job_title` làm input?**
  * Vì nhãn $y$ được tạo ra từ `job_title`. Nếu đưa `job_title` vào input, mô hình sẽ bị **Target Leakage (rò rỉ nhãn)**, mô hình chỉ nhìn tiêu đề là đoán được ngay mà không hề học nội dung mô tả JD. Để bài toán có ý nghĩa thực tế, input chỉ là nội dung JD!
* **Phân chia tập dữ liệu:**
  * Tổng: 7,184 mẫu.
  * Tỉ lệ: **80% Train ($N=5,747$) / 20% Test ($N=1,437$)**.
  * Phương pháp: **Stratified Split** (phân tầng theo nhãn) với `random_state = 42`. Cách chia này đảm bảo lớp hiếm như Data Scientist ($N=86$) vẫn xuất hiện cân đối ở cả tập train và test.
* **Thuật toán so sánh:** Logistic Regression, Linear SVM, K-Nearest Neighbors (KNN với cosine metric), Random Forest, XGBoost Classifier.
* **Kết quả:**
  * **K-Nearest Neighbors (KNN)** đạt kết quả tốt nhất: **Macro-F1 = 0.7759**, **Accuracy = 88.45%**.
  * XGBoost bám sát với Macro-F1 = 0.7625, Accuracy = 88.24%.
  * Linear SVM đạt Macro-F1 = 0.7525, Accuracy = 86.85%.
  * F1-score từng nghề: Software Engineer (0.93), IT Support (0.88), Data Analyst (0.85), QA/Tester (0.84), Data Engineer (0.78), DevOps (0.73), AI/ML Engineer (0.72), Data Scientist (0.47).

---

### BÀI TOÁN 2: DỰ ĐOÁN MỨC LƯƠNG (REGRESSION)
* **Mục tiêu:** Dự đoán mức lương trung bình tháng (tính theo Triệu VNĐ).
* **Input ($X$):** Kết hợp đa phương thức (Multi-modal):
  * Vector văn bản: SBERT 384 chiều từ `job_description`.
  * Đặc trưng cấu trúc: `job_role_label`, `norm_location` (Hà Nội, HCM, Đà Nẵng, Khác), `norm_position` (Nhân viên, Quản lý/Leader,...), `norm_job_type`, `exp_years` (số năm kinh nghiệm), và `year_clean` (năm tuyển dụng).
* **Output ($y$):** Mức lương trung bình tháng `salary_mean_vnd / 1,000,000` (Triệu VNĐ).
* **Tại sao BẮT BUỘC phải có trường `year`?**
  * Thị trường công nghệ biến động rất mạnh theo thời gian. Mức lương cho một vị trí năm 2026 sẽ có sự khác biệt so với các giai đoạn trước do trượt giá và tái cơ cấu ngân sách tuyển dụng. Trường `year` giúp mô hình nắm bắt xu hướng thời gian.
* **Phân chia tập dữ liệu:**
  * Tổng: 11,292 mẫu có lương hợp lệ.
  * Tỉ lệ: **80% Train ($N=9,033$) / 20% Test ($N=2,259$)**, `random_state = 42`.
* **Ablation Study (Nghiên cứu bóc tách 4 cấu hình để chứng minh giá trị từng nhóm feature):**
  * **E1 (Đặc trưng bảng KHÔNG có Year):** XGBoost đạt MAE = 5.80M, $R^2 = 0.2736$.
  * **E2 (Đặc trưng bảng CÓ Year):** XGBoost đạt MAE = 5.56M, $R^2 = 0.3217$.  
    $\rightarrow$ *Kết luận 1: Thêm Year giúp $R^2$ tăng vọt từ 0.27 lên 0.32 (tăng 4.8% khả năng giải thích biến sai lương).*
  * **E3 (SBERT Text + Bảng KHÔNG có Year):** XGBoost đạt MAE = 5.41M, $R^2 = 0.3354$.  
    $\rightarrow$ *Kết luận 2: Thêm SBERT giúp mô hình hiểu độ khó và công nghệ trong JD, giảm thêm sai số MAE.*
  * **E4 (SBERT Text + Bảng CÓ Year - Full Model):** **XGBoost Regressor** đạt **MAE = 5.24 Triệu VNĐ**, **$R^2 = 0.3683$**.  
    $\rightarrow$ *So với Median Baseline (MAE = 6.93 Triệu VNĐ), mô hình đã giảm sai số tới 24.4%!*

---

### BÀI TOÁN 3: PHÂN CỤM CÔNG VIỆC & KỸ NĂNG (CLUSTERING)
* **Mục tiêu:** Tự động khám phá các nhóm nghề và tổ hợp kỹ năng tự nhiên trong thị trường mà không cần nhãn người gán trước.
* **Input ($X$):** SBERT 384 chiều của văn bản tổng hợp: `job_title + ". " + job_description + " Yêu cầu: " + requirements`.
* **Tại sao ở bài này lại ghép cả tiêu đề + mô tả + yêu cầu?**
  * Vì đây là học không giám sát (Unsupervised), không có bài toán dự đoán nhãn $y$ nên không sợ rò rỉ nhãn. Việc ghép cả 3 trường giúp mô hình có đầy đủ nhất ngữ cảnh của một vị trí công việc.
* **Số lượng mẫu:** Toàn bộ 14,307 tin tuyển dụng.
* **Thuật toán so sánh:** K-Means ($k=3, 4, 5, 6, 8, 10$), Agglomerative Hierarchical, DBSCAN, Gaussian Mixture Model (GMM trên 25 chiều PCA).
* **Kết quả:**
  * K-Means với **$k=10$** đạt điểm Silhouette cao nhất (**0.1368**), chỉ số Davies-Bouldin là 3.16.
  * Tự động tách thành công các cụm chuyên biệt rõ ràng:
    * **Cụm 0 (QA / QC & Testing):** `test, kiểm thử, testing, tester, automation`
    * **Cụm 4 (AI & Machine Learning):** `ai, learning, mô hình, python, deep learning, trí tuệ`
    * **Cụm 5 (Mobile & Game Development):** `android, ios, game, mobile, ứng dụng, flutter`
    * **Cụm 9 (Data Analytics & BI):** `dữ liệu, data, phân tích, sql, báo cáo, power bi`
    * **Cụm 2 (Hạ tầng mạng & Phần cứng):** `máy, mạng, hệ thống, thiết bị, cài đặt`
    * **Cụm 1, 7, 8 (Phát triển phần mềm tổng thể):** `phát triển, phần mềm, java, backend, api`

---

## 5. Phân tích vấn đề Rò rỉ Dữ liệu (Data Leakage)

Khi giảng viên hỏi: *"Quy trình của các em có bị Data Leakage không?"*, nhóm hãy tự tin trả lời: **Dự án hoàn toàn KHÔNG bị Data Leakage nhờ 4 nguyên tắc sau:**

1. **Cách ly Train / Test trước khi Fit Scaler và Encoder:**
   * Trong `src/features/feature_builder.py`, `OneHotEncoder` và `StandardScaler` **chỉ được gọi hàm `.fit_transform()` trên tập Train**.
   * Với tập Test, mã nguồn **chỉ gọi hàm `.transform()`**.
   * Nghĩa là các tham số trung bình, phương sai, và từ điển danh mục hoàn toàn không biết trước thông tin của tập Test.
2. **Cách ly Target trong bài toán Classification:**
   * Nhãn mục tiêu $y$ được trích xuất từ `job_title`.
   * Input $X$ vào mô hình phân loại **chỉ chứa `job_description`**, tuyệt đối không chứa `job_title`.
3. **Không Impute Target trong bài toán Regression:**
   * 2,507 tin không rõ lương được gán trạng thái `unspecified` và loại riêng ra. Nhóm không điền bừa lương trung vị vào tập này để train hồi quy.
4. **Mô hình SBERT dùng Pretrained cố định (Frozen):**
   * Mô hình MiniLM-L12-v2 được dùng trực tiếp ở chế độ trích xuất đặc trưng (Feature Extraction), không can thiệp fine-tune làm rò rỉ nhãn.

---

## 6. Bảng tổng hợp số liệu cốt lõi (Cheat Sheet để vấn đáp)

| Nội dung | Con số / Kết quả cụ thể |
| :--- | :--- |
| **Tổng số tin sau làm sạch** | **14,307 tin duy nhất** (từ 15,467 ban đầu, bỏ 1,160 tin trùng) |
| **Mẫu dùng cho Classification**| **7,184 mẫu** (8 nhóm nghề kỹ thuật, bỏ nhóm Other) |
| **Tập Train / Test Classification**| 5,747 train (80%) / 1,437 test (20%) - Stratified |
| **Model Classification tốt nhất**| **K-Nearest Neighbors (Cosine)**: Macro-F1: **77.59%**, Accuracy: **88.45%** |
| **Mẫu dùng cho Salary Regression**| **11,292 mẫu** (có mức lương số trong khoảng 3M - 150M) |
| **Tập Train / Test Regression**| 9,033 train (80%) / 2,259 test (20%) |
| **Model Regression tốt nhất** | **XGBoost Regressor (E4)**: MAE = **5.24 triệu**, $R^2 = \mathbf{0.3683}$ |
| **Hiệu quả của trường Year** | Thêm Year làm $R^2$ tăng từ **0.2736 (E1)** lên **0.3217 (E2)** |
| **Hiệu quả của SBERT Text** | Thêm SBERT giúp giảm MAE từ 5.80M xuống 5.34M |
| **Model Phân cụm tốt nhất** | **K-Means ($k=10$)**: Silhouette Score = **0.1368** |

---

*Tài liệu này được tạo tự động dựa trên kết quả thực thi thực tế của mã nguồn JobLens Vietnam.*

