"""
Script thu thập việc làm IT lịch sử từ các snapshot SITEMAP (Wayback Machine + TopCV Live).
Quy trình được tách rời thành 2 BƯỚC ĐỘC LẬP:
  - BƯỚC 1: Quét sitemaps (Wayback + Live), lọc từ khóa IT (TECH_JOB_KEYWORDS) và gom danh sách link
            lưu riêng vào: data/raw/topcv_it_job_urls.csv (chỉ mất ~2-3 phút cho toàn bộ các năm).
  - BƯỚC 2: Đọc file link trên và tiến hành cào chi tiết JD lưu vào: data/raw/topcv_it_jobs_wayback.csv
            (hỗ trợ resume, checkpoint kép, bắt Ctrl+C an toàn).

Cách dùng (từ thư mục gốc dự án):
  1. Chỉ gom URL từ sitemaps:
     python src/crawler/run_crawl.py --step urls

  2. Cào chi tiết JD từ danh sách URLs đã gom:
     python src/crawler/run_crawl.py --step crawl

  3. Cào thử nghiệm 50 jobs đầu tiên:
     python src/crawler/run_crawl.py --step crawl --limit 50

  4. Chạy toàn bộ (tự động gom link nếu chưa có rồi cào):
     python src/crawler/run_crawl.py --step all
"""

import argparse
import sys
from pathlib import Path

# Cấu hình UTF-8 cho Windows Terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo thư mục gốc dự án luôn có trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.crawler.topcv_crawler import TopCVCrawler
except ModuleNotFoundError:
    from topcv_crawler import TopCVCrawler


URLS_FILE = str(PROJECT_ROOT / "data" / "raw" / "topcv_it_job_urls.csv")
JOBS_FILE = str(PROJECT_ROOT / "data" / "raw" / "topcv_it_jobs_wayback.csv")


def main():
    parser = argparse.ArgumentParser(
        description="TopCV IT Jobs Crawler - Sitemaps Time-Travel (2022 - 2026)"
    )
    parser.add_argument(
        "--step",
        choices=["urls", "crawl", "all", "1", "2"],
        default="all",
        help=(
            "Chọn bước chạy: 'urls' (hoặc 1) = chỉ gom link, "
            "'crawl' (hoặc 2) = cào chi tiết JD, 'all' = cả hai"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số bài đăng chi tiết cần cào (ví dụ: --limit 100 để test nhanh)",
    )
    parser.add_argument(
        "--urls-file",
        type=str,
        default=URLS_FILE,
        help="Đường dẫn file lưu danh sách URL",
    )
    parser.add_argument(
        "--jobs-file",
        type=str,
        default=JOBS_FILE,
        help="Đường dẫn file lưu dữ liệu chi tiết JD",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Lưu checkpoint sau mỗi N bài đăng (mặc định: 50)",
    )

    args = parser.parse_args()

    crawler = TopCVCrawler()

    # Chạy BƯỚC 1: Gom URL
    if args.step in ["urls", "1"]:
        print("\n" + "=" * 70)
        print("🎯 BƯỚC 1: GOM DANH SÁCH URL VIỆC LÀM IT TỪ SITEMAPS")
        print("=" * 70)
        crawler.collect_urls_from_sitemaps(save_path=args.urls_file)

    # Chạy BƯỚC 2: Cào chi tiết JD
    elif args.step in ["crawl", "2"]:
        print("\n" + "=" * 70)
        print("🎯 BƯỚC 2: CÀO CHI TIẾT JD TUẦN TỰ VỚI CURL_CFFI")
        print("=" * 70)
        crawler.crawl_jobs_from_url_file(
            url_file_path=args.urls_file,
            save_path=args.jobs_file,
            limit=args.limit,
            save_every=args.save_every,
            resume=True,
        )

    # Chạy TOÀN BỘ: Gom nếu chưa có, rồi cào
    else:
        print("\n" + "=" * 70)
        print("🎯 TIẾN TRÌNH TOÀN DIỆN (BƯỚC 1 + BƯỚC 2 - CRAWL TUẦN TỰ)")
        print("=" * 70)
        crawler.crawl_from_wayback_sitemaps(
            url_file_path=args.urls_file,
            save_path=args.jobs_file,
            limit=args.limit,
            save_every=args.save_every,
            resume=True,
        )


if __name__ == "__main__":
    main()
