# JobLens Vietnam: Báo cáo IntroDS

## 1. Mục tiêu và dữ liệu

Phân tích title, kỹ năng, nhóm nghề và lương niêm yết của **14,034** tin đã lọc từ [TinixAI](https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions). Bộ lọc IT/AI/Data trong notebook dựa trên title; dữ liệu gốc, nội dung raw và năm được giữ riêng với các cột đã xử lý.

Dữ liệu không phải mẫu ngẫu nhiên của toàn thị trường. Mức phủ giữa các năm, việc đăng lại tin và thay đổi nguồn có thể ảnh hưởng số dòng; năm 2026 có thể chưa đủ cả năm. `year` là năm do nguồn cung cấp, không có ngày/tháng để xác minh thời điểm riêng từng tin.

License của nguồn: CC BY-NC 4.0. Ghi công TiniX AI; dataset card dẫn công trình CareerPathKG (Le và cộng sự, 2026).

## 2. Tiền xử lý

Từ điển có 305 tên kỹ năng chuẩn và các alias. FlashText khớp theo ranh giới từ. HTML, entity và khoảng trắng được làm sạch; `C++`, `C#`, `.NET` được giữ khi trích kỹ năng.

Kinh nghiệm thiếu: **1,149** tin. `Không` đơn lẻ được để thiếu; `không yêu cầu kinh nghiệm` được lấy là 0. Khoảng kinh nghiệm được biểu diễn bằng cận dưới, không phải số năm thực tế của ứng viên.

Lương đọc được để tạo midpoint: **11,859** tin. Khoảng đóng dùng trung bình hai đầu; lương thỏa thuận, giá trị không hợp lệ và cận mở không được điền một midpoint giả. USD quy đổi bằng giả định cố định **25,500 VND/USD**, không phải tỷ giá lịch sử. Lương chưa ghi đơn vị thời gian được giả định là tháng; gross/net và lạm phát chưa được hiệu chỉnh.

Có **1,254** dòng nằm trong nhóm nội dung trùng theo công ty, title, nội dung và năm. Dữ liệu EDA giữ các dòng này để phản ánh nguồn; mô hình chia theo công ty để giảm rò rỉ do đăng lại cùng JD.

## 3. EDA

### Nhóm nghề

| role | jobs |
| --- | --- |
| Other | 8682 |
| Software Engineer | 4271 |
| Data Analyst | 550 |
| AI/ML Engineer | 259 |
| Data Engineer | 175 |
| Data Scientist | 97 |

![Nhóm nghề](figures/roles.png)

### Phân bố năm

| year | jobs | median_salary_vnd | salary_rows |
| --- | --- | --- | --- |
| 2,022.000 | 2,219.000 | 12,500,000.000 | 2,213.000 |
| 2,023.000 | 2,428.000 | 12,500,000.000 | 2,395.000 |
| 2,024.000 | 2,453.000 | 13,500,000.000 | 2,404.000 |
| 2,025.000 | 4,178.000 | 15,000,000.000 | 3,294.000 |
| 2,026.000 | 2,756.000 | 19,000,000.000 | 1,553.000 |

![Phân bố năm](figures/years.png)

### Kỹ năng phổ biến

| skill | jobs | share |
| --- | --- | --- |
| JavaScript | 2062 | 0.147 |
| SQL | 2037 | 0.145 |
| Python | 1822 | 0.130 |
| HTML | 1748 | 0.125 |
| CSS | 1729 | 0.123 |
| Git | 1684 | 0.120 |
| MySQL | 1548 | 0.110 |
| Java | 1440 | 0.103 |
| Machine Learning | 1411 | 0.101 |
| REST API | 1378 | 0.098 |
| React | 1325 | 0.094 |
| Linux | 1153 | 0.082 |
| SQL Server | 1133 | 0.081 |
| Docker | 1128 | 0.080 |
| CI/CD | 1051 | 0.075 |

![Kỹ năng](figures/skills.png)

### Lương theo nhóm

| Nghề | Số tin | Trung vị (triệu VND/tháng) | Trung bình (triệu VND/tháng) |
| --- | --- | --- | --- |
| AI/ML Engineer | 150 | 30.000 | 29.614 |
| Data Analyst | 379 | 16.000 | 20.581 |
| Data Engineer | 95 | 24.000 | 24.307 |
| Data Scientist | 29 | 30.000 | 29.039 |
| Other | 7822 | 12.500 | 15.168 |
| Software Engineer | 3384 | 17.500 | 20.651 |

![Lương theo nhóm](figures/salary_roles.png)

## 4. Thiết kế đánh giá

Train/validation/test được chia theo `company_group` với seed cố định, khoảng 60/20/20 theo số công ty. Tỷ lệ theo số tin có thể khác. Các công ty đã nhận diện không xuất hiện ở hai tập; tên công ty viết khác nhau vẫn có thể thuộc cùng thực thể. Tin thiếu công ty được nhóm theo fingerprint nội dung.

Nhãn nghề tạo từ title bằng quy tắc, không phải nhãn chuyên gia. Tin không thuộc rõ một trong năm nhóm hoặc có nhiều nhóm được ghi `Other`, giữ cho EDA và loại khỏi bài toán phân loại năm nhóm. Mô hình phân loại chỉ dùng mô tả/yêu cầu; không đưa title, matched_groups hoặc role_label vào đầu vào. F1 đo mức khớp nhãn sơ bộ, không chứng minh độ chính xác với nhãn đã kiểm định.

TF-IDF, imputer, categorical encoder và vocabulary kỹ năng của mô hình chỉ fit trên train. Chọn mô hình theo validation; test dùng để báo cáo. Nhãn title yếu và các dấu hiệu chức danh trong JD vẫn có thể giúp mô hình khớp quy tắc.

## 5. Phân loại nghề

| Mô hình | F1 validation | F1 test | Precision macro test | Recall macro test |
| --- | --- | --- | --- | --- |
| tfidf_logistic | 0.742 | 0.844 | 0.919 | 0.818 |
| tfidf_svm | 0.751 | 0.839 | 0.936 | 0.793 |
| tfidf_mlp | 0.678 | 0.749 | 0.909 | 0.679 |

Chọn **tfidf_svm** theo macro-F1 validation. Baseline dự đoán lớp phổ biến trên train có macro-F1 test **0.179**.

![Confusion matrix](figures/classification_confusion.png)

### Precision/recall từng nghề trên test

| Nghề | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| AI/ML Engineer | 0.800 | 0.889 | 0.842 | 45.000 |
| Data Analyst | 0.898 | 0.942 | 0.919 | 103.000 |
| Data Engineer | 1.000 | 0.667 | 0.800 | 21.000 |
| Data Scientist | 1.000 | 0.476 | 0.645 | 21.000 |
| Software Engineer | 0.984 | 0.994 | 0.989 | 799.000 |

Số tin phân loại: train=3,139, validation=1,224, test=989. Các nhóm ít mẫu cần được đọc cùng support; F1 của chúng kém ổn định hơn.

## 6. Dự đoán lương

| Mô hình | MAE validation (triệu VND) | MAE test (triệu VND) | RMSE test (triệu VND) | R² test |
| --- | --- | --- | --- | --- |
| lightgbm | 7.414 | 6.119 | 9.526 | 0.387 |
| random_forest | 7.574 | 6.276 | 9.877 | 0.341 |
| ridge | 7.799 | 6.374 | 9.902 | 0.337 |

Chọn **lightgbm** theo MAE validation. Baseline trung vị train có MAE test **8.19 triệu VND**. Bài toán dùng tin có một nhóm nghề xác định và midpoint hợp lệ.

Ngưỡng cắt đuôi lương được học từ train: 2,000,000–65,000,000 VND. Validation/test giữ các midpoint hợp lệ, kể cả nằm ngoài ngưỡng này.

Mô hình học `log1p(midpoint)`, đánh giá MAE/RMSE sau biến đổi về VND. Dự đoán là midpoint lương niêm yết; không phải lương thực nhận hay lời hứa trả lương.

![Dự đoán lương](figures/salary_predictions.png)

### Giải thích bằng SHAP

Tree SHAP được tính cho **lightgbm** trên thang log1p(lương), với 150 tin test. Mean |SHAP| đo mức ảnh hưởng trong mô hình; dấu dương/âm không phải phần bù lương nhân quả. Kỹ năng tương quan có thể chia sẻ đóng góp. Không diễn giải kết quả này thành mức tăng lương khi học thêm một kỹ năng.

![SHAP](figures/salary_shap.png)

## 7. Mạng kỹ năng và cụm JD

Đồ thị có **171** kỹ năng và **3437** cạnh đồng xuất hiện sau ngưỡng lọc. Cạnh có count, Jaccard và lift; không suy ra quan hệ phụ thuộc kỹ thuật từ việc cùng được nhắc trong JD.

![Mạng kỹ năng](figures/skill_network.png)

### Các nút có tổng số lần đồng xuất hiện lớn nhất

| skill | job_count | degree | weighted_degree | degree_centrality | betweenness_centrality |
| --- | --- | --- | --- | --- | --- |
| JavaScript | 2062 | 99 | 19667 | 0.582 | 0.020 |
| Git | 1684 | 125 | 19237 | 0.735 | 0.051 |
| Python | 1822 | 136 | 17659 | 0.800 | 0.073 |
| MySQL | 1548 | 119 | 17483 | 0.700 | 0.031 |
| SQL | 2037 | 133 | 17423 | 0.782 | 0.066 |
| CSS | 1729 | 89 | 16930 | 0.524 | 0.012 |
| HTML | 1748 | 89 | 16688 | 0.524 | 0.012 |
| REST API | 1378 | 102 | 16182 | 0.600 | 0.021 |
| Docker | 1128 | 122 | 15655 | 0.718 | 0.040 |
| React | 1325 | 88 | 14563 | 0.518 | 0.009 |

Weighted degree là tổng count cạnh; betweenness được tính trên đồ thị không trọng số, vì count lớn không biểu thị khoảng cách lớn. Centrality phụ thuộc vào ngưỡng giữ nút/cạnh.

Sentence embeddings chuẩn hóa được giảm chiều bằng UMAP. KMeans và HDBSCAN mô tả cấu trúc nội dung; không được xem là nhóm nghề đúng đã xác minh. Text bị cắt ở max_seq_length của embedding model; JD dài có thể mất phần cuối.

| method | clusters | noise_rows | silhouette_reduced_space | rows |
| --- | --- | --- | --- | --- |
| kmeans | 8 | 0 | 0.482 | 14034 |
| hdbscan | 44 | 5060 | 0.528 | 14034 |

![Các cụm](figures/clusters.png)

## 8. Kết luận và hướng mở rộng

**tfidf_svm** đạt macro-F1 test **0.839** trên 989 tin có nhãn nghề đơn từ title. Đây là mức khớp nhãn sơ bộ; cần một tập nhãn độc lập do người kiểm định để đánh giá độ đúng về nghề.

**lightgbm** có MAE test **6.12 triệu VND/tháng**, R² **0.387**, giảm MAE 25.3% so với baseline trung vị train trên cùng tập test. Sai số vẫn lớn so với nhiều midpoint lương trong dữ liệu; dự đoán cần được đọc như ước lượng có sai số.

Bộ dữ liệu cho phép mô tả kỹ năng được đề cập, lương niêm yết và các nhóm nghề trong phạm vi title đã lọc. Kết quả phân loại cần được đánh giá thêm bằng mẫu gán nhãn thủ công; lương cần tách gross/net, thời gian và đơn vị chính xác hơn trước khi so sánh xu hướng năm.

Bước tiếp theo: rà soát title giữ/bỏ, gán nhãn nghề thủ công bằng hai người độc lập, chuẩn hóa công ty để giảm alias, đánh giá holdout theo năm, bootstrap khoảng tin cậy và nghiên cứu sai lệch do các tin không công khai lương.

## 9. Chạy lại và artifacts

`python main.py --stage all` chạy ingestion, preprocessing, features, EDA, phân loại, hồi quy, graph/clustering và tạo báo cáo. `python main.py --stage all --no-embeddings` chạy các phần không cần tải sentence model; kết quả sẽ ghi rõ phần semantic chưa chạy. Cấu hình ở `configs/pipeline.json`.

Mô hình lưu trong `artifacts/models/`, metrics và bảng nằm trong `reports/`, dữ liệu sạch trong `data/processed/jobs_cleaned.csv`/Parquet. Notebook EDA, modeling và clustering dùng lại các module này.

## 10. Tài liệu phương pháp

[GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html), [LightGBM](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html), [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html), [SentenceTransformer](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).

Phiên bản thư viện, Python, revision nguồn và fingerprint dữ liệu được ghi trong `run_manifest.json`. Các bảng và slide là snapshot của lượt chạy; cần tạo lại khi thay đổi dữ liệu/cấu hình.
