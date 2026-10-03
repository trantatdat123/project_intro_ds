# 🕷️ Job Crawlers (`src/crawler/`)

Module thu thập dữ liệu việc làm ngành Công nghệ Thông tin & Dữ liệu từ TopCV qua phương pháp **Sitemap Time-Travel** (kết hợp Wayback Machine lưu trữ lịch sử 2022–2025 và TopCV Live 2026).

**Giới hạn của dữ liệu TopCV:** Wayback hiện chỉ dùng để tìm URL trong sitemap; bước crawl chi tiết tải trang TopCV hiện tại. Các mốc 2022, 2023, 2025 và 2026 có độ phủ rất khác nhau, không dùng số dòng theo năm để suy ra nhu cầu tuyển dụng. `posted_date` của các lượt crawl mới chỉ lấy từ JSON-LD `datePosted`; sitemap `<lastmod>` được lưu riêng là `sitemap_lastmod` vì đó là ngày sửa trang. File CSV chi tiết đã crawl trước thay đổi này không có cột nguồn ngày, nên cần xem `notebooks/02_topcv_wayback_eda.ipynb` để kiểm tra và không thể xác minh từng ngày cũ nếu thiếu HTML gốc.

## CareerViet

`CareerVietCrawler` lấy URL tiếng Việt từ sitemap CareerViet, lọc slug theo các từ khóa IT/Data rõ nghĩa, rồi crawl trang chi tiết tuần tự bằng `curl_cffi` và kiểm tra tiêu đề bằng bộ lọc TopCV. Bộ lọc từ khóa có thể bỏ sót tin có tiêu đề quá chung. Kết quả dùng cùng 10 cột với TopCV nhưng lưu file riêng. Sitemap CareerViet là dữ liệu đang công bố, không phải nguồn lưu trữ lịch sử.

Chạy từ thư mục gốc dự án:

```python
from src.crawler import CareerVietCrawler

with CareerVietCrawler(min_delay=2.5, max_delay=5.0) as crawler:
    crawler.collect_urls_from_sitemaps()           # data/raw/careerviet_it_job_urls.csv
    crawler.crawl_jobs_from_url_file(limit=50)     # data/raw/careerviet_it_jobs.csv
```

Bỏ `limit=50` để crawl toàn bộ URL đã gom. Hàm crawl tạo `careerviet_it_jobs.state.json` để chạy tiếp sau khi dừng; khi gặp HTTP 403/429, nó dừng và giữ URL chưa xử lý. `posted_date` lấy từ `datePosted` trên trang chi tiết, không lấy ngày sửa sitemap.

Mỗi lần chạy, `limit` áp dụng cho số URL **chưa xử lý**. Chạy lại `crawl_jobs_from_url_file(limit=50)` để xử lý tiếp 50 URL; không cần gom lại sitemap. CSV được ghi sau mỗi tin thành công, còn `save_every=50` quy định số URL giữa các lần lưu state. Dùng `resume=False` để bắt đầu lại và ghi đè file kết quả. Có thể gọi `crawler.parse_job(url)` để lấy một tin dưới dạng dict; trường không có trên trang sẽ để trống.

---

## 📁 Cấu trúc thư mục

```text
src/crawler/
├── __init__.py           # Package export
├── topcv_crawler.py      # Core crawler class TopCVCrawler (curl_cffi requests.Session)
├── careerviet_crawler.py # CareerVietCrawler, sitemap + JD một luồng
├── vietnamworks_crawler.py # VietnamWorksCrawler, sitemap + Next.js/JSON-LD
├── run_crawl.py          # Script thực thi dòng lệnh (CLI runner)
└── README.md             # Tài liệu hướng dẫn sử dụng module
```

---

## 🚀 Hướng dẫn chạy (từ thư mục gốc dự án)

### 1. Chỉ gom URL việc làm IT từ Sitemaps (Bước 1):
```bash
python src/crawler/run_crawl.py --step urls
```
*Kết quả lưu tại:* `data/raw/topcv_it_job_urls.csv`

### 2. Cào chi tiết JD tuần tự (Bước 2):
```bash
# Cào tuần tự, mỗi lần chỉ gửi một request bằng curl_cffi:
python src/crawler/run_crawl.py --step crawl
```
*Kết quả lưu tại:* `data/raw/topcv_it_jobs_wayback.csv` kèm checkpoint `topcv_it_jobs_wayback.state.json`

### 3. Cào thử nghiệm N bài đăng:
```bash
python src/crawler/run_crawl.py --step crawl --limit 50
```

### 4. Chạy toàn bộ (Tự động gom link nếu chưa có rồi cào):
```bash
python src/crawler/run_crawl.py --step all
```


## CareerViet / CareerBuilder lịch sử

Chọn `MODE = "history"` hoặc `"both"`, rồi chạy các cell gom snapshot và crawl lịch sử trong `notebooks/01_careerviet_crawler.ipynb`. Mặc định 2022–2025; sửa `HISTORY_START_YEAR`, `HISTORY_END_YEAR` để chọn khoảng năm chụp.

```python
from src.crawler import CareerVietCrawler

with CareerVietCrawler(min_delay=5, max_delay=10, timeout=60) as crawler:
    crawler.collect_historical_urls(start_year=2022, end_year=2025)
    crawler.crawl_historical_jobs(limit=50)
```

- Tìm CDX của cả `careerbuilder.vn` và `careerviet.vn`, chọn một snapshot mỗi URL/năm. CDX phân trang và lưu tiến trình từng trang trong `careerviet_history_urls.cdx.json`; chạy lại để tiếp tục sau lỗi.
- URL lưu ở `data/raw/careerviet_history_urls.csv`; kết quả và checkpoint crawl lưu ở `careerviet_history_jobs.csv` và `careerviet_history_jobs.state.json`.
- Tải HTML của snapshot, không chuyển sang website hiện tại khi Wayback chuyển hướng. Snapshot lỗi hoặc không nhận diện được JD được giữ để thử lại.
- Kết quả có 10 cột thông thường và thêm `archive_timestamp`, `archive_url`. Ngày chụp không thay thế ngày đăng; ngày đăng không xác định thì để trống. Khi thống kê job, cần khử trùng theo URL/ID vì một tin có thể xuất hiện ở nhiều snapshot/năm.
- `resume=True` giữ và bổ sung danh sách đã gom. Muốn một bộ dữ liệu riêng cho khoảng năm khác, chọn `save_path` khác hoặc dùng `resume=False` để ghi đè danh sách URL.
- Chỉ lấy được những trang Wayback đã lưu và cho phép truy cập; không bảo đảm đủ các năm. Chưa xác minh HTML lịch sử trực tiếp do Wayback timeout trong môi trường phát triển. Đã kiểm tra phân trang, checkpoint, provenance và ngày thiếu bằng dữ liệu giả lập.

Tham khảo: [CDX API chính thức](https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server), [thông báo CareerBuilder đổi tên thành CareerViet](https://careerviet.vn/vi/talentcommunity/thong-cao-bao-chi-careerbuilder-vn-chuyen-sang-ten-mien-moi-careerviet-vn.35A524FB.html).

## VietnamWorks

Chạy `notebooks/03_vietnamworks_crawler.ipynb` từ trên xuống. Mặc định `MODE = "live"`, `CRAWL_LIMIT = 50`, request cách nhau 3–6 giây. Chạy lại cell crawl để xử lý tiếp các URL chưa hoàn tất; đặt `CRAWL_LIMIT = None` để crawl hết.

```python
from src.crawler import VietnamWorksCrawler

with VietnamWorksCrawler(min_delay=3, max_delay=6) as crawler:
    crawler.collect_urls_from_sitemaps()
    jobs = crawler.crawl_jobs_from_url_file(limit=50)
```

- Lấy URL từ [sitemap chính thức](https://www.vietnamworks.com/sitemap/sitemap.xml), lọc IT/Data theo slug và kiểm tra lại tiêu đề. Bộ lọc có thể bỏ sót tin có tiêu đề chung chung.
- Đọc dữ liệu Next.js nhúng trong HTML bằng `curl_cffi`, không cần trình duyệt; giữ cùng 10 cột với TopCV/CareerViet. Dùng chung phần request tuần tự và checkpoint của CareerViet để giảm code lặp.
- Trang mới lấy `approvedOn` làm ngày duyệt đăng, `expiredOn` làm hạn nộp. `onlineOn` có thể là ngày đẩy lại tin; `lastUpdatedOn`, `lastSyncedOn`, sitemap `lastmod` và ngày chụp đều không thay thế ngày đăng. Thiếu ngày thì để trống.
- File hiện tại: `data/raw/vietnamworks_it_job_urls.csv`, `vietnamworks_it_jobs.csv`, `vietnamworks_it_jobs.state.json`. Khi 403/429, dừng và giữ URL chưa hoàn tất. HTML thiếu dữ liệu tin được giữ để thử lại.
- Có thể chọn `MODE = "history"`/`"both"`, hoặc gọi `collect_historical_urls(start_year=2022, end_year=2025)` và `crawl_historical_jobs(limit=50)`. Lưu riêng `vietnamworks_history_urls.csv`, `vietnamworks_history_jobs.csv` và checkpoint; thêm `archive_timestamp`, `archive_url`.
- Lịch sử tải HTML snapshot, không chuyển sang trang live. Chỉ nhận trang có dữ liệu nhúng phù hợp hoặc JobPosting JSON-LD; trang cũ khác cấu trúc sẽ được giữ lại. CDX chọn một snapshot/URL/năm chụp, không bảo đảm đủ tin hay đủ năm. Chưa chạy crawl lịch sử thực tế trong lần bổ sung này.
