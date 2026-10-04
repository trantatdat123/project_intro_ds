# JobLens Vietnam

Dự án IntroDS phân tích tin tuyển dụng IT, AI và Data Science từ [TinixAI Vietnamese Job Descriptions](https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions), theo [plan.md](plan.md).

## Chạy project

Python 3.11 trở lên được khuyến nghị. Mở terminal tại thư mục project, dùng cùng môi trường Python cho terminal và notebook:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --stage all
```

`all` tái sử dụng dữ liệu TinixAI đã tải/lọc nếu có; sau đó làm sạch, chia tập theo công ty, tạo features, EDA, huấn luyện 6 mô hình, SHAP, mạng kỹ năng, UMAP + KMeans/HDBSCAN và báo cáo. Lượt đầu cần Internet để tải dataset và sentence model; embedding toàn bộ JD có thể mất 15–30 phút trên CPU. Những lượt sau dùng cache nếu nội dung và cấu hình embedding không đổi. Mô hình ML được huấn luyện lại khi chạy `all`.

Chạy các phần không dùng sentence embeddings:

```powershell
python main.py --stage all --no-embeddings
```

### Chạy riêng từng bước

| Stage | Chức năng |
| --- | --- |
| `acquire` | Tải raw nếu chưa có, lọc lại title theo `configs/title_filters.json` |
| `prepare` | Làm sạch, parse lương/kinh nghiệm, gán nghề, trích kỹ năng, chia tập |
| `features` | Multi-hot kỹ năng và sentence embeddings có cache |
| `eda` | Bảng thống kê và biểu đồ |
| `classify` | TF-IDF + Logistic Regression / LinearSVC / MLP |
| `salary` | LightGBM / Random Forest / Ridge, baseline và Tree SHAP |
| `cluster` | Đồng xuất hiện, centrality, UMAP + KMeans/HDBSCAN |
| `report` | Báo cáo Markdown và mẫu gán nhãn thủ công từ kết quả đã lưu |

Ví dụ: `python main.py --stage classify`. Nếu chưa có dữ liệu sạch, runner tự chạy prepare. Sau khi thay đổi dữ liệu, parser hoặc quy tắc, chạy lại `all` để cập nhật các kết quả liên quan.

Muốn tải lại phiên bản nguồn hiện tại: `python main.py --stage acquire --refresh`. Muốn chỉ thay đổi bộ lọc title: sửa `configs/title_filters.json`, chạy `acquire`, rồi `all`. Notebook download ban đầu có bộ lọc riêng trong cell; sửa bộ lọc CLI ở config hoặc sửa bộ lọc notebook trong notebook tương ứng.

`--config đường_dẫn.json` dùng cấu hình khác. `--limit N` chỉ áp dụng khi prepare/all, ghi đè dữ liệu sạch bằng N dòng đầu để chạy quy trình trên tập nhỏ; các lớp ít tin có thể không đủ để huấn luyện/chia tập. Chạy lại `all` không có limit để phục hồi kết quả toàn bộ dữ liệu.

## Notebook

Chạy các cell từ trên xuống. Ba notebook phân tích đọc dữ liệu sạch và artifacts; tùy chọn `RETRAIN` / `RECOMPUTE` mặc định tắt để mở nhanh.

1. [Tải và lọc TinixAI](notebooks/tinix_download_filter.ipynb) — notebook tự chứa đã có trước.
2. [EDA](notebooks/01_exploratory_data_analysis.ipynb) — chất lượng dữ liệu, năm, nghề, kỹ năng, lương và kinh nghiệm.
3. [Đánh giá mô hình](notebooks/02_modeling_evaluation.ipynb) — chia tập, metrics, confusion matrix, SHAP và dùng model đã lưu.
4. [Mạng kỹ năng và phân cụm](notebooks/03_skills_and_clustering.ipynb) — multi-hot, centrality, KMeans/HDBSCAN và đọc ví dụ cụm.

## Cấu trúc

```text
configs/               Bộ lọc title, từ điển kỹ năng, tham số pipeline
src/data/              Tải và lọc dataset theo batch
src/preprocessing/     Text, lương, kinh nghiệm, nghề và kỹ năng
src/features/          Skill multi-hot, embeddings và cache
src/models/            Chia tập, classifier, regressor, graph/clustering
src/reporting.py       Bảng EDA, figures và báo cáo từ số liệu thật
src/pipeline.py        Điều phối các stage
src/predict.py         Áp dụng model đã lưu cho dữ liệu sạch
main.py                CLI pipeline
notebooks/             Notebook download, EDA, modeling, clustering
scripts/               Nguồn tạo slide từ metrics và bảng kết quả
data/raw/              TinixAI raw
data/processed/        Dataset đã lọc và jobs_cleaned.csv/Parquet
data/features/         Skill matrix, text_embeddings.npy, UMAP coordinates
artifacts/models/      Model joblib và skill_network.graphml
artifacts/embedding_model/  Cache mô hình Hugging Face
artifacts/numba_cache/  Cache Numba trong project cho SHAP/UMAP
reports/               Metrics, dự đoán, bảng, report và slide
tests/                 Thư mục dự phòng theo plan, chưa có test suite
```

Dữ liệu, model và cache được bỏ khỏi Git. Giữ config, source, notebook, metrics và báo cáo để review. Figures được tái tạo bằng pipeline. Slide được lưu trong `reports/slides/`; cách tạo lại ở `scripts/README.md`.

## Thiết kế xử lý và đánh giá

- Giữ cột raw cùng cột đã làm sạch. Lương khoảng đóng dùng midpoint; lương thỏa thuận và cận mở giữ thông tin nhưng không tạo midpoint giả. USD dùng tỷ giá cố định trong config, chưa hiệu chỉnh gross/net/lạm phát.
- `experience_min` là cận dưới kinh nghiệm yêu cầu. Giá trị thiếu giữ null; chỉ impute bên trong model trên train.
- FlashText khớp canonical skill và alias theo ranh giới từ. Kỹ năng lấy từ mô tả/yêu cầu; việc được đề cập chưa đồng nghĩa kỹ năng bắt buộc.
- Năm nghề: Data Analyst, Data Scientist, Data Engineer, AI/ML Engineer, Software Engineer. Nhãn tạo bằng quy tắc title. Nghề ngoài năm nhóm/mơ hồ giữ `Other` cho EDA và loại khỏi phân loại năm nhóm.
- Train/validation/test chia khoảng 60/20/20 theo **nhóm công ty**, seed 42. Các công ty đã chuẩn hóa không giao nhau giữa tập; tên công ty viết khác nhau vẫn có thể là cùng thực thể.
- TF-IDF, vocabulary skill của regressor, imputer và one-hot encoder chỉ fit trên train. Chọn classifier bằng macro-F1 validation, regressor bằng MAE validation. Test dùng để báo cáo; baseline là lớp phổ biến/trung vị train.
- Hồi quy dùng tin có nhãn nghề đơn và midpoint hợp lệ. Cắt đuôi 1–99% học từ train, chỉ áp dụng train. Target `log1p(midpoint)`; MAE/RMSE đánh giá bằng VND/tháng, SHAP bằng thang log.
- Features toàn bộ dữ liệu dùng cho EDA/phân cụm, tách khỏi vocabulary mô hình lương. Đồ thị xuất count/Jaccard/lift và centrality. Silhouette đo trên UMAP, không chứng minh cụm là nghề đúng.

## Dùng model đã lưu

```powershell
python -m src.predict --input data/processed/jobs_cleaned.parquet --output reports/inference_predictions.csv
```

Input phải có schema dữ liệu đã prepare. Classifier chỉ trả một trong năm nhóm và không có lớp Other. CLI đánh dấu phạm vi lương bằng nhãn nghề đã chuẩn hóa; tin ngoài phạm vi không được xuất dự đoán lương.

## Kết quả và báo cáo

[Báo cáo dự án](reports/project_report.md) dùng số liệu chạy thực tế. Metrics gồm `classification_metrics.json`, `salary_metrics.json`, `clustering_metrics.json`, `data_quality.json`; `run_manifest.json` ghi cấu hình và lượt chạy. Báo cáo nhận diện giới hạn dữ liệu và nhãn yếu, không suy kết quả thành thống kê toàn thị trường hoặc mức tăng lương nhân quả của một kỹ năng.

[Slide trình bày](reports/slides/joblens_vietnam_v3.pptx) gồm 13 trang, có bảng và biểu đồ chỉnh sửa được.

[Từ điển dữ liệu](reports/data_dictionary.md) giải thích các cột raw, cột đã xử lý và schema kết quả. Runner ghi fingerprint để ngăn báo cáo ghép metrics của dữ liệu khác khi phát hiện được.

`reports/annotation_template.csv` có mẫu title/JD và cột `human_role` trống để gán nhãn thủ công trước bước đánh giá tiếp theo.

## Nguồn và ghi công

- TinixAI Vietnamese Job Descriptions, CC BY-NC 4.0. Dataset card có thông tin tác giả và CareerPathKG. Bộ dữ liệu đang dùng có 606.878 tin gốc, 14.034 tin được bộ lọc giữ lại; revision nguồn nằm trong metadata.
- Sentence model: [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
- scikit-learn, LightGBM, SHAP, FlashText, NetworkX, UMAP, HDBSCAN và Sentence-Transformers. Kết quả cần đánh giá thêm trên nhãn do người kiểm định và tập giữ lại theo năm.
