# JobLens Vietnam

Dự án IntroDS phân tích việc làm IT, AI và Data Science từ dataset [TinixAI Vietnamese Job Descriptions](https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions).

## Dữ liệu và notebook

```text
notebooks/
  tinix_download_filter.ipynb
data/
  raw/tinixai_vietnamese_job_descriptions/
    data.parquet
  processed/
    tinixai_it_ai_data_jobs.csv
    tinixai_it_ai_data_jobs.parquet
    tinixai_it_ai_data_jobs.metadata.json
```

## Cách chạy

1. Cài thư viện: `pip install -r requirements.txt`.
2. Mở `notebooks/tinix_download_filter.ipynb` bằng Python kernel của dự án.
3. Chạy các cell từ trên xuống để tải dataset và lọc theo `job_title`.
4. Sửa `TITLE_PHRASES` và `TITLE_ACRONYMS` để thay đổi bộ lọc, rồi chạy lại cell tạo hàm và cell lọc.

Notebook đọc từng batch để giảm RAM, giữ các cột gốc và thêm nhóm/cụm từ đã khớp. Kết quả lưu thành CSV và Parquet; metadata ghi phiên bản nguồn, bộ lọc và số dòng. Một tin có thể khớp nhiều nhóm nhưng chỉ được ghi một lần.

Cột `year` được giữ nguyên từ nguồn. Dataset phát hành theo CC BY-NC 4.0; nguồn và thông tin ghi công nằm trên dataset card.
