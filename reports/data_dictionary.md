# Các cột dữ liệu JobLens

Các cột gốc TinixAI được giữ nguyên trong `jobs_cleaned.csv` và Parquet. CSV ghi list dưới dạng JSON; dùng `src.utils.read_frame` để đọc lại thành list.

## Cột raw

| Cột | Ý nghĩa |
| --- | --- |
| id | ID từ nguồn; dùng loại trùng ID |
| job_title / company_name | Title và công ty từ TinixAI |
| salary / location | Chuỗi lương và địa điểm gốc |
| job_type / job_industry | Hình thức và ngành từ nguồn |
| experience_level / education_level / job_position | Chuỗi yêu cầu kinh nghiệm, học vấn, cấp/vị trí từ nguồn |
| job_description / requirements / benefits | Mô tả, yêu cầu, phúc lợi gốc |
| year | Năm do nguồn cung cấp; không có ngày/tháng để xác minh từng tin |
| title_normalized / matched_groups / matched_phrases | Audit bộ lọc title, trước bước chuẩn hóa nghề |

## Cột đã xử lý

| Cột | Kiểu / ý nghĩa |
| --- | --- |
| title_clean | Title đã loại HTML/entity/khoảng trắng thừa |
| description_clean / requirements_clean / benefits_clean | Text sạch từng phần |
| text_clean | Mô tả + yêu cầu, dùng cho classifier và embeddings; không thêm title/phúc lợi |
| experience_min | Số năm tối thiểu yêu cầu; null khi không đọc được/thiếu thông tin |
| experience_missing | Cờ thiếu kinh nghiệm |
| location_group | Thành phố nhận diện, nhiều địa điểm, remote hoặc nhóm chưa xác định |
| seniority | Management / Lead / Senior / Middle / Junior / Unspecified từ title |
| role_candidates | List nhóm nghề khớp các quy tắc |
| role_standard | Một trong 5 nghề nếu có đúng một candidate; còn lại Other |
| role_label_status | single / ambiguous / unmapped |
| role_label_source | title_rule: nhãn sơ bộ từ quy tắc title |
| salary_min / salary_max | Cận lương quy về VND/tháng; cận còn lại có thể null với khoảng mở |
| salary_mean | Midpoint hai cận đóng hoặc lương exact; null với khoảng mở/thỏa thuận/không hợp lệ |
| currency | VND / USD nhận diện từ chuỗi nguồn |
| is_negotiable | Cờ lương thỏa thuận |
| salary_bound | range / exact / upper / lower / missing |
| salary_period | year / month_assumed / hour/day / unknown; year đã chia 12 ở các cột số |
| salary_parse_status | Lý do parsed, negotiable, unknown_currency, invalid_range... |
| extracted_skills | List canonical skill, mỗi skill tối đa một lần/JD |
| skill_count | Số kỹ năng được đề cập theo từ điển |
| text_fingerprint | SHA256 của nội dung đã fold, hỗ trợ nhận diện tin trùng |
| company_group | Tên công ty đã fold; thiếu công ty dùng fingerprint nội dung |
| duplicate_content | Thuộc nhóm trùng công ty + title + nội dung + năm; các dòng vẫn được giữ |
| classification_eligible | Nhãn nghề đơn, text ít nhất 30 ký tự |
| salary_eligible | Có midpoint dương; mô hình lương còn yêu cầu classification_eligible |
| split | train / validation / test theo nhóm công ty, seed cố định |

## Tệp features và kết quả

- `skill_matrix.parquet`: ID + cột skill binary theo ngưỡng tần suất toàn bộ; dùng cho phân tích khám phá.
- `text_embeddings.npy`: hàng cùng thứ tự với jobs_cleaned; metadata kiểm tra fingerprint text, model/revision, số dòng và độ dài token.
- `umap_coordinates.npy`: tọa độ UMAP nhiều chiều, cùng thứ tự jobs_cleaned.
- `skill_cooccurrence_matrix.csv`: đường chéo là số JD đề cập skill, ô ngoài chéo là count cùng đề cập sau ngưỡng lọc cạnh. GraphML không có self-loop.
- `skill_cooccurrence.csv`: edge count, Jaccard và lift.
- `skill_centrality.csv`: degree, weighted degree, degree centrality và betweenness không trọng số.
- `classification_predictions.csv` / `salary_predictions.csv`: dự đoán trên test, không phải toàn bộ dữ liệu.
- `cluster_assignments.csv`: ID, title, nhãn nghề sơ bộ, KMeans/HDBSCAN label; -1 là HDBSCAN noise.

Features mô hình lương dùng imputer/one-hot/skill vocabulary fit trên train bên trong model joblib, không lấy vocabulary toàn bộ của file skill_matrix.
