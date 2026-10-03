"""Crawl tin tuyển dụng IT/Data CareerViet từ sitemap công khai, tuần tự một luồng."""

import csv
import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse, urlencode, unquote_plus

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests
from tqdm import tqdm

from .topcv_crawler import DEFAULT_COLUMNS, TopCVCrawler


class RateLimitedError(RuntimeError):
    """Server yêu cầu dừng gửi request."""


class CareerVietCrawler:
    """Thu thập URL IT/Data và JD CareerViet bằng curl_cffi, không dùng async."""

    SITEMAP_INDEX = "https://careerviet.vn/sitemap/sitemap.xml"
    DEFAULT_URL_FILE = "data/raw/careerviet_it_job_urls.csv"
    DEFAULT_JOB_FILE = "data/raw/careerviet_it_jobs.csv"
    ARCHIVE_URL_FILE = "data/raw/careerviet_history_urls.csv"
    ARCHIVE_JOB_FILE = "data/raw/careerviet_history_jobs.csv"
    ARCHIVE_COLUMNS = DEFAULT_COLUMNS + ["archive_timestamp", "archive_url"]
    SOURCE_NAME = "CareerViet"
    ARCHIVE_DOMAINS = ("careerbuilder.vn", "careerviet.vn")
    ARCHIVE_PATHS = ("vi/tim-viec-lam/*", "vi/search-job/*")
    ARCHIVE_FILTERS = ()
    _JOB_PATH = re.compile(r"^/vi/(?:tim-viec-lam|search-job)/[^/]+\.[0-9a-fA-F]{8}(?:\.html)?$")

    def __init__(self, min_delay: float = 2.5, max_delay: float = 5.0, timeout: int = 30):
        if min_delay < 0 or max_delay < min_delay:
            raise ValueError("Cần 0 <= min_delay <= max_delay")
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.session = requests.Session(impersonate="chrome120")
        self.session.headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.7",
        })

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @classmethod
    def _is_job_url(cls, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme == "https" and parsed.hostname == "careerviet.vn" and bool(cls._JOB_PATH.match(parsed.path))

    @staticmethod
    def _slug(url: str) -> str:
        return urlparse(url).path.rsplit("/", 1)[-1].rsplit(".", 2)[0]

    @staticmethod
    def _is_it_slug(slug: str) -> bool:
        """Dùng dấu gạch ngang làm ranh giới từ để tránh khớp nhầm từ ngắn."""
        tokens = set(slug.lower().split("-"))
        strong_words = {
            "it", "cntt", "data", "software", "developer", "devops", "helpdesk",
            "frontend", "backend", "fullstack", "tester", "python", "java",
            "javascript", "typescript", "php", "react", "angular", "nodejs",
            "database", "sql", "mlops", "cloud", "cybersecurity", "blockchain",
        }
        strong_phrases = (
            "lap-trinh", "phan-mem", "du-lieu", "cong-nghe-thong-tin",
            "tri-tue-nhan-tao", "machine-learning", "deep-learning",
            "kiem-thu-phan-mem", "an-ninh-mang", "bao-mat-thong-tin",
            "quan-tri-mang", "he-thong-thong-tin", "business-analyst",
        )
        padded = f"-{slug.lower()}-"
        return bool(tokens & strong_words) or any(f"-{phrase}-" in padded for phrase in strong_phrases)

    @staticmethod
    def _xml_locs(xml: str) -> list[str]:
        root = ET.fromstring(xml)
        return [(el.text or "").strip() for el in root.findall(".//{*}loc")]

    @staticmethod
    def _retry_after(value: str | None) -> str:
        if not value:
            return ""
        if value.isdigit():
            return f"; thử lại sau ít nhất {value} giây"
        try:
            seconds = max(0, int((parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()))
            return f"; thử lại sau ít nhất {seconds} giây"
        except (TypeError, ValueError, OverflowError):
            return ""

    def _get(self, url: str, allow_redirects: bool = True):
        """Một request tại một thời điểm; dừng khi gặp 403/429."""
        for attempt in range(3):
            time.sleep(random.uniform(self.min_delay, self.max_delay))
            try:
                response = self.session.get(url, timeout=self.timeout, allow_redirects=allow_redirects)
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"Không tải được {url}: {exc}") from exc
                time.sleep(3 * (attempt + 1))
                continue
            if response.status_code in (403, 429):
                hint = self._retry_after(response.headers.get("Retry-After")) if response.status_code == 429 else ""
                raise RateLimitedError(f"HTTP {response.status_code} tại {url}{hint}. Đã dừng để giữ checkpoint.")
            if response.status_code == 404:
                return None
            if not allow_redirects and 300 <= response.status_code < 400:
                raise RuntimeError("Snapshot chuyển hướng; giữ lại để kiểm tra, không tải trang live")
            if response.status_code in (421, 500, 502, 503, 504) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        return None

    def collect_urls_from_sitemaps(
        self,
        save_path: str = DEFAULT_URL_FILE,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Lấy URL tiếng Việt từ các sitemap job_vi_*.xml và lọc IT/Data theo slug."""
        if limit is not None and limit <= 0:
            raise ValueError("limit phải là số nguyên dương hoặc None")
        response = self._get(self.SITEMAP_INDEX)
        if response is None:
            raise RuntimeError("Không tìm thấy sitemap CareerViet")
        sitemaps = [
            url for url in self._xml_locs(response.text)
            if re.fullmatch(r"https://careerviet\.vn/sitemap/job_vi_\d+\.xml", url)
        ]
        if not sitemaps:
            raise RuntimeError("Sitemap index không có job_vi_*.xml")

        urls: dict[str, None] = {}
        for sitemap in sitemaps:
            try:
                response = self._get(sitemap)
            except RateLimitedError:
                raise
            except Exception as exc:
                print(f"Bỏ qua sitemap lỗi {sitemap}: {exc}")
                continue
            if response is None:
                continue
            for url in self._xml_locs(response.text):
                if self._is_job_url(url) and self._is_it_slug(self._slug(url)):
                    urls[url] = None
                    if limit is not None and len(urls) >= limit:
                        break
            print(f"{sitemap}: đã gom {len(urls):,} URL IT/Data")
            if limit is not None and len(urls) >= limit:
                break

        result = pd.DataFrame({"url": list(urls)})
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            result.to_csv(save_path, index=False, encoding="utf-8-sig")
        return result

    @staticmethod
    def _job_json_ld(soup: BeautifulSoup) -> dict:
        def candidates(value):
            if isinstance(value, list):
                for item in value:
                    yield from candidates(item)
            elif isinstance(value, dict):
                if value.get("@type") == "JobPosting" or "JobPosting" in (value.get("@type") or []):
                    yield value
                if "@graph" in value:
                    yield from candidates(value["@graph"])

        for script in soup.select('script[type="application/ld+json"]'):
            try:
                return next(candidates(json.loads(script.string or script.get_text())))
            except (ValueError, StopIteration, TypeError):
                continue
        return {}

    @classmethod
    def _archive_record(cls, url: str, timestamp: str) -> dict:
        parsed = urlparse(url)
        if (parsed.scheme not in {"http", "https"}
                or (parsed.hostname or "").removeprefix("www.") not in cls.ARCHIVE_DOMAINS
                or not cls._JOB_PATH.fullmatch(parsed.path)
                or not re.fullmatch(r"\d{14}", timestamp)):
            raise ValueError(f"Snapshot không hợp lệ: {timestamp} {url}")
        return {"url": url, "archive_timestamp": timestamp,
                "archive_url": f"https://web.archive.org/web/{timestamp}id_/{url}"}

    def collect_historical_urls(
        self, start_year: int = 2022, end_year: int = 2025,
        save_path: str = ARCHIVE_URL_FILE, page_size: int = 1000,
        resume: bool = True,
    ) -> pd.DataFrame:
        """CDX theo năm chụp: một snapshot/URL/năm, phân trang và checkpoint."""
        if not 1996 <= start_year <= end_year <= datetime.now().year or page_size <= 0:
            raise ValueError("Khoảng năm hoặc page_size không hợp lệ")
        columns = ["url", "archive_timestamp", "archive_url"]
        state_path = f"{os.path.splitext(save_path)[0]}.cdx.json"
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        records = {}
        cursors = {}
        if resume and os.path.exists(save_path):
            for row in pd.read_csv(save_path, dtype=str, keep_default_na=False).to_dict("records"):
                record = self._archive_record(row["url"], row["archive_timestamp"])
                records[record["archive_url"]] = record
            if os.path.exists(state_path):
                with open(state_path, encoding="utf-8") as file:
                    cursors = json.load(file)

        def checkpoint():
            temp = f"{save_path}.tmp"
            pd.DataFrame(records.values(), columns=columns).to_csv(temp, index=False, encoding="utf-8-sig")
            os.replace(temp, save_path)
            with open(f"{state_path}.tmp", "w", encoding="utf-8") as file:
                json.dump(cursors, file)
            os.replace(f"{state_path}.tmp", state_path)

        try:
            for year in range(start_year, end_year + 1):
                for domain in self.ARCHIVE_DOMAINS:
                    for route in self.ARCHIVE_PATHS:
                        query = f"{domain}/{route}"
                        key = f"{year}:{query}"
                        if key in cursors and cursors[key] is None:
                            continue
                        cursor = cursors.get(key, "")
                        while True:
                            params = {"url": query, "from": str(year), "to": str(year),
                                      "output": "json", "fl": "timestamp,original",
                                      "filter": ["statuscode:200", "mimetype:text/html", *self.ARCHIVE_FILTERS],
                                      "collapse": "urlkey", "limit": page_size, "showResumeKey": "true"}
                            if cursor:
                                params["resumeKey"] = unquote_plus(cursor)
                            response = self._get("https://web.archive.org/cdx/search/cdx?" + urlencode(params, doseq=True))
                            if response is None:
                                raise RuntimeError("CDX trả 404; chưa đánh dấu truy vấn hoàn tất")
                            payload = response.json()
                            if not isinstance(payload, list):
                                raise ValueError("CDX không trả danh sách JSON")
                            if payload and payload[0] != ["timestamp", "original"]:
                                raise ValueError("CDX trả schema không mong đợi")
                            next_cursor = ""
                            for row in payload[1:]:
                                if len(row) == 1:
                                    next_cursor = row[0]
                                elif len(row) == 2:
                                    timestamp, url = row
                                    try:
                                        record = self._archive_record(url, timestamp)
                                    except ValueError:
                                        continue
                                    if self._is_it_slug(self._slug(url)):
                                        records[record["archive_url"]] = record
                            if next_cursor and next_cursor == cursor:
                                raise RuntimeError("CDX lặp cursor; dừng để tránh gửi request vô hạn")
                            cursors[key] = next_cursor or None
                            checkpoint()
                            print(f"{year} {domain}/{route}: tổng {len(records):,} snapshot IT/Data")
                            if not next_cursor:
                                break
                            cursor = next_cursor
        finally:
            checkpoint()
        result = pd.DataFrame(records.values(), columns=columns)
        if result.empty:
            print("Không tìm thấy snapshot phù hợp; không đồng nghĩa website không có tin trong giai đoạn này.")
        return result

    def crawl_historical_jobs(
        self, url_file_path: str = ARCHIVE_URL_FILE, save_path: str = ARCHIVE_JOB_FILE,
        limit: int | None = None, save_every: int = 50, resume: bool = True,
    ) -> pd.DataFrame:
        """Tải HTML snapshot thực, giữ ngày chụp riêng với ngày đăng."""
        return self.crawl_jobs_from_url_file(
            url_file_path, save_path, limit, save_every, resume, archived=True,
        )

    @staticmethod
    def _text(element) -> str:
        return element.get_text("\n", strip=True) if element is not None else ""

    @classmethod
    def _section(cls, soup: BeautifulSoup, names: tuple[str, ...]) -> str:
        for row in soup.select("div.detail-row"):
            heading = row.find(["h2", "h3"], class_="detail-title")
            if heading and any(name in heading.get_text(" ", strip=True).lower() for name in names):
                body = heading.find_next_sibling("div")
                return cls._text(body)
        return ""

    @classmethod
    def _field(cls, soup: BeautifulSoup, label: str) -> str:
        for strong in soup.select("li > strong, div.map > strong"):
            if strong.get_text(" ", strip=True).casefold() == label.casefold():
                return cls._text(strong.parent.find("p", recursive=False))
        return ""

    @staticmethod
    def _date(value) -> str:
        if not isinstance(value, str):
            return ""
        for pattern, fmt in ((r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"), (r"\d{2}/\d{2}/\d{4}", "%d/%m/%Y")):
            match = re.match(pattern, value.strip())
            if match:
                try:
                    return datetime.strptime(match.group(), fmt).date().isoformat()
                except ValueError:
                    pass
        return ""

    @classmethod
    def parse_job_html(cls, html: str, url: str) -> dict | None:
        """Trả về cùng 10 cột với TopCV; ngày đăng lấy từ datePosted, không lấy lastmod."""
        soup = BeautifulSoup(html, "html.parser")
        data = cls._job_json_ld(soup)
        if not data and not soup.select_one("div.detail-row .detail-title"):
            return None
        title = data.get("title") or cls._text(soup.find("h1"))
        if not title:
            return None
        organization = data.get("hiringOrganization") or {}
        company = organization.get("name", "") if isinstance(organization, dict) else ""
        if not company:
            company = cls._text(soup.select_one('[itemprop="hiringOrganization"] [itemprop="name"]'))
        salary = cls._field(soup, "Lương")
        if not salary:
            base = data.get("baseSalary") or {}
            value = base.get("value", {}) if isinstance(base, dict) else {}
            salary = str(value.get("value", "")) if isinstance(value, dict) else str(value or "")
        location = cls._field(soup, "Địa điểm")
        if not location:
            places = data.get("jobLocation") or []
            if isinstance(places, dict):
                places = [places]
            location = ", ".join(dict.fromkeys(
                str(place.get("address", {}).get("addressLocality", ""))
                for place in places if isinstance(place, dict) and isinstance(place.get("address"), dict)
            )).strip(", ")
        description = cls._section(soup, ("mô tả công việc",))
        if not description and isinstance(data.get("description"), str):
            description = cls._text(BeautifulSoup(data["description"], "html.parser"))
        return {
            "job_title": title,
            "company_name": company,
            "salary_raw": salary,
            "experience_raw": cls._field(soup, "Kinh nghiệm"),
            "location_raw": location,
            "posted_date": cls._date(data.get("datePosted")) or cls._date(cls._field(soup, "Ngày đăng")),
            "deadline_date": cls._date(data.get("validThrough")) or cls._date(cls._field(soup, "Hết hạn nộp")),
            "job_description_raw": description,
            "job_requirements_raw": cls._section(soup, ("yêu cầu công việc", "yêu cầu ứng viên")),
            "url": url,
        }

    def parse_job(self, url: str) -> dict | None:
        """Tải một tin; trả None cho URL hết hạn hoặc không còn trang JD."""
        if not self._is_job_url(url):
            raise ValueError(f"Không phải URL tin {self.SOURCE_NAME}: {url}")
        response = self._get(url)
        return self.parse_job_html(response.text, url) if response is not None else None

    @staticmethod
    def _save_state(path: str, processed: set[str]):
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump({"processed_urls": sorted(processed)}, file, ensure_ascii=False)
        os.replace(temp_path, path)

    def crawl_jobs_from_url_file(
        self,
        url_file_path: str = DEFAULT_URL_FILE,
        save_path: str = DEFAULT_JOB_FILE,
        limit: int | None = None,
        save_every: int = 50,
        resume: bool = True,
        *, archived: bool = False,
    ) -> pd.DataFrame:
        """Crawl tuần tự; lưu CSV + trạng thái để chạy tiếp sau khi dừng."""
        if limit is not None and limit <= 0:
            raise ValueError("limit phải là số nguyên dương hoặc None")
        if save_every < 0:
            raise ValueError("save_every phải >= 0")
        df_urls = pd.read_csv(url_file_path, dtype=str, keep_default_na=False)
        if "url" not in df_urls.columns:
            raise ValueError(f"{url_file_path} thiếu cột url")
        columns = self.ARCHIVE_COLUMNS if archived else DEFAULT_COLUMNS
        identity = "archive_url" if archived else "url"
        archive_records = {}
        if archived:
            for row in df_urls.to_dict("records"):
                record = self._archive_record(row["url"], row["archive_timestamp"])
                archive_records[record["archive_url"]] = record
            urls = list(archive_records)
        else:
            urls = list(dict.fromkeys(url for url in df_urls["url"] if self._is_job_url(url)))
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        state_path = f"{os.path.splitext(save_path)[0]}.state.json"
        processed: set[str] = set()
        if resume and os.path.exists(save_path) and os.path.exists(state_path):
            try:
                with open(state_path, encoding="utf-8") as file:
                    processed = set(json.load(file).get("processed_urls", []))
            except (ValueError, TypeError, AttributeError) as exc:
                print(f"State lỗi, khôi phục từ CSV: {exc}")
        if not resume:
            pd.DataFrame(columns=columns).to_csv(save_path, index=False, encoding="utf-8-sig")
        elif os.path.exists(save_path):
            existing = pd.read_csv(save_path)
            if list(existing.columns) != columns:
                raise ValueError(f"{save_path} không đúng schema {self.SOURCE_NAME}; hãy chọn file đầu ra khác")
            processed.update(existing[identity].dropna().astype(str))
        else:
            processed.clear()
            pd.DataFrame(columns=columns).to_csv(save_path, index=False, encoding="utf-8-sig")

        pending = [url for url in urls if url not in processed]
        if limit is not None:
            pending = pending[:limit]
        since_save = 0
        try:
            with tqdm(pending, desc=f"{self.SOURCE_NAME}, 1 luồng") as progress:
                for url in progress:
                    try:
                        response = self._get(url, allow_redirects=False) if archived else self._get(url)
                        if response is None and archived:
                            print(f"Snapshot chưa tải được (404), giữ lại: {url}")
                            continue
                        if response is not None:
                            original = archive_records[url]["url"] if archived else url
                            job = self.parse_job_html(response.text, original)
                            if job is None:
                                print(f"Không thấy JobPosting/h1: {url}; giữ URL để thử lại")
                                continue
                            if TopCVCrawler.is_it_job(job["job_title"]):
                                if archived:
                                    job.update(archive_records[url])
                                with open(save_path, "a", newline="", encoding="utf-8") as file:
                                    csv.DictWriter(file, fieldnames=columns).writerow(job)
                        processed.add(url)
                        since_save += 1
                        if save_every > 0 and since_save >= save_every:
                            self._save_state(state_path, processed)
                            since_save = 0
                    except RateLimitedError as exc:
                        print(exc)
                        break
                    except Exception as exc:
                        print(f"Lỗi {url}: {exc}; giữ URL để thử lại")
        except KeyboardInterrupt:
            print("Đã dừng bằng Ctrl+C, đang lưu checkpoint")
        finally:
            self._save_state(state_path, processed)
        return pd.read_csv(save_path)
