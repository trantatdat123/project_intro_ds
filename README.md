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

## Crawl tin TopCV

Mở `notebooks/topcv_it_jobs.ipynb` và chạy các cell:

1. Nạp crawler và đặt `YEAR = 2026`, `WORKERS = 2`, `DELAY = 2.0`.
2. Quét sitemap, lọc IT/CNTT/Data Science và năm 2026, lưu TXT.
3. Đọc TXT, crawl chi tiết và lưu JSONL với đúng 14 cột gốc của TinixAI. Chạy lại cell crawl để tiếp tục; tin lỗi vẫn được thử lại.
4. Chạy cell xuất CSV/Parquet khi cần dùng bảng dữ liệu; thứ tự và tên cột giống TinixAI.

Năm dựa trên `lastmod` của sitemap, không phải ngày đăng; URL thiếu ngày bị bỏ qua. Bộ từ khóa ngành nằm trong `IT_TERMS` của crawler, được khớp với slug URL ngay khi lấy link. File Python chỉ cung cấp hàm để notebook import.

### File lưu

- Link: `data/raw/topcv/topcv_sitemap_it_data_urls_2026.txt` (tên thay đổi theo năm).
- Dữ liệu crawl: `data/raw/topcv/topcv_jobs_raw.jsonl` (mỗi dòng một tin, 14 cột gốc TinixAI, dùng chung giữa các năm).
- Bảng xuất: `data/processed/topcv_jobs.csv` và `data/processed/topcv_jobs.parquet`.

File link được thay thế sau khi quét thành công; JSONL được ghi nối tiếp. Các cột là `id`, `job_title`, `company_name`, `salary`, `location`, `job_type`, `job_industry`, `experience_level`, `education_level`, `job_position`, `job_description`, `benefits`, `requirements`, `year`. `year` lấy từ `datePosted` trên trang chi tiết, không phải `lastmod` của sitemap; nếu ngày đăng thiếu hoặc không hợp lệ thì để trống. ID của TopCV và TinixAI thuộc hai hệ riêng, cần thêm nguồn khi gộp hai tập dữ liệu.

Trong notebook chỉ cần `YEAR`, `WORKERS`, `DELAY`. Khi đã có file link đã lọc, bỏ qua cell sitemap và chạy cell crawl:

```python
crawler.get_job_urls_from_sitemap(2026)
urls = crawler.load_job_urls(2026)
crawler.crawl_from_links(urls)
```
