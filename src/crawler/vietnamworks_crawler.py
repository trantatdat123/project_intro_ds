"""VietnamWorks IT/Data: sitemap, HTML Next.js và snapshot Wayback, một luồng."""

import json
import os
import re
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup

from .careerviet_crawler import CareerVietCrawler


class VietnamWorksCrawler(CareerVietCrawler):
    """Dùng chung request/checkpoint với CareerViet; đọc dữ liệu nhúng, không chạy JS."""

    SOURCE_NAME = "VietnamWorks"
    SITEMAP_INDEX = "https://www.vietnamworks.com/sitemap/sitemap.xml"
    DEFAULT_URL_FILE = "data/raw/vietnamworks_it_job_urls.csv"
    DEFAULT_JOB_FILE = "data/raw/vietnamworks_it_jobs.csv"
    ARCHIVE_URL_FILE = "data/raw/vietnamworks_history_urls.csv"
    ARCHIVE_JOB_FILE = "data/raw/vietnamworks_history_jobs.csv"
    ARCHIVE_DOMAINS = ("vietnamworks.com",)
    ARCHIVE_PATHS = ("*",)
    ARCHIVE_FILTERS = (r"original:https?://(?:www\.)?vietnamworks\.com/[^/]+-\d+-jv/?(?:\?.*)?",)
    _JOB_PATH = re.compile(r"^/[^/]+-\d+-jv/?$")

    @classmethod
    def _is_job_url(cls, url: str) -> bool:
        parsed = urlparse(url)
        return (parsed.scheme == "https"
                and (parsed.hostname or "").removeprefix("www.") in cls.ARCHIVE_DOMAINS
                and bool(cls._JOB_PATH.fullmatch(parsed.path)))

    @staticmethod
    def _slug(url: str) -> str:
        return re.sub(r"-\d+-jv/?$", "", urlparse(url).path.lstrip("/")).strip("-")

    def collect_urls_from_sitemaps(
        self, save_path: str = DEFAULT_URL_FILE, limit: int | None = None,
    ) -> pd.DataFrame:
        """Sitemap hiện tại; lọc IT/Data theo slug, khử trùng theo ID tin."""
        if limit is not None and limit <= 0:
            raise ValueError("limit phải là số nguyên dương hoặc None")
        index = self._get(self.SITEMAP_INDEX)
        if index is None:
            raise RuntimeError("Không tìm thấy sitemap VietnamWorks")
        sitemaps = [url for url in self._xml_locs(index.text)
                    if re.fullmatch(r"https://www\.vietnamworks\.com/sitemap/jobs(?:[._-]\d+)?\.xml", url)]
        if not sitemaps:
            raise RuntimeError("Sitemap index không có sitemap jobs")
        urls = {}
        for sitemap in sitemaps:
            response = self._get(sitemap)
            if response is None:
                continue
            for url in self._xml_locs(response.text):
                if self._is_job_url(url) and self._is_it_slug(self._slug(url)):
                    job_id = re.search(r"-(\d+)-jv", urlparse(url).path).group(1)
                    urls.setdefault(job_id, url)
                    if limit is not None and len(urls) >= limit:
                        break
            if limit is not None and len(urls) >= limit:
                break
        result = pd.DataFrame({"url": list(urls.values())})
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            result.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"Đã gom {len(result):,} URL VietnamWorks IT/Data")
        return result

    @staticmethod
    def _flight_records(soup: BeautifulSoup) -> dict:
        """Ghép chunk Flight; bản ghi T dùng độ dài byte UTF-8, có thể chứa newline."""
        decoder = json.JSONDecoder()
        chunks = []
        for script in soup.select("script:not([src])"):
            content = script.get_text()
            for marker in re.finditer(r"self\.__next_f\.push\(", content):
                try:
                    item, _ = decoder.raw_decode(content[marker.end():].lstrip())
                except ValueError:
                    continue
                if isinstance(item, list) and len(item) > 1 and item[0] == 1 and isinstance(item[1], str):
                    chunks.append(item[1])
        stream = "".join(chunks).encode("utf-8")
        records, position = {}, 0
        while position < len(stream):
            if stream[position:position + 1] == b"\n":
                position += 1
                continue
            header = re.match(rb"([0-9a-f]+):", stream[position:])
            if header is None:
                break
            key = header[1].decode()
            position += header.end()
            text_header = re.match(rb"T([0-9a-f]+),", stream[position:])
            if text_header:
                position += text_header.end()
                end = position + int(text_header[1], 16)
                if end > len(stream):
                    raise ValueError("HTML thiếu chunk dữ liệu VietnamWorks")
                records[key] = stream[position:end].decode("utf-8")
                position = end
            else:
                end = stream.find(b"\n", position)
                end = len(stream) if end < 0 else end
                try:
                    records[key] = json.loads(stream[position:end])
                except ValueError:
                    pass  # Bản ghi module (I), hint (HL), v.v. không phải dữ liệu tin.
                position = end
        return records

    @classmethod
    def _page_job(cls, soup: BeautifulSoup, job_id: str) -> dict:
        records = cls._flight_records(soup)
        roots = list(records.values())
        next_data = soup.select_one("script#__NEXT_DATA__")
        if next_data:
            try:
                roots.append(json.loads(next_data.get_text()))
            except ValueError:
                pass

        def find(value):
            if isinstance(value, dict):
                if str(value.get("jobId")) == job_id and value.get("jobTitle"):
                    yield value
                else:
                    for child in value.values():
                        yield from find(child)
            elif isinstance(value, list):
                for child in value:
                    yield from find(child)

        def resolve(value, seen=frozenset()):
            if isinstance(value, str) and re.fullmatch(r"\$[0-9a-f]+", value):
                key = value[1:]
                return resolve(records[key], seen | {key}) if key in records and key not in seen else ""
            if isinstance(value, dict):
                return {key: resolve(child, seen) for key, child in value.items()}
            if isinstance(value, list):
                return [resolve(child, seen) for child in value]
            return value

        return resolve(next((job for root in roots for job in find(root)), {}))

    @classmethod
    def parse_job_html(cls, html: str, url: str) -> dict | None:
        """Ngày đăng = approvedOn (ngày duyệt công khai), không dùng onlineOn/lastmod."""
        soup = BeautifulSoup(html, "html.parser")
        match = re.search(r"-(\d+)-jv/?$", urlparse(url).path)
        if match is None:
            return None
        data = cls._page_job(soup, match[1])
        if not data:
            # Snapshot giao diện cũ có JSON-LD JobPosting: giữ nguyên nguồn ngày.
            structured = cls._job_json_ld(soup)
            job = super().parse_job_html(html, url) if structured else None
            if job is None:
                return None
            for heading in soup.select("h2, h3"):
                if heading.get_text(" ", strip=True).casefold() in {"yêu cầu công việc", "job requirements"}:
                    job["job_requirements_raw"] = cls._text(heading.find_next_sibling())
                    break
            if not job["job_description_raw"] and not job["job_requirements_raw"]:
                return None
            return job
        if not data.get("jobDescription") and not data.get("jobRequirement"):
            return None
        places = data.get("workingLocations") or []
        location = ", ".join(dict.fromkeys(
            place.get("cityNameVI") or place.get("cityName") or place.get("address") or ""
            for place in places if isinstance(place, dict)
        )).strip(", ")
        years = data.get("yearsOfExperience")
        salary = data.get("prettySalaryVI") or data.get("prettySalary") or ""
        if data.get("isSalaryVisible") is False:
            salary = "Thương lượng"
        return {
            "job_title": data["jobTitle"],
            "company_name": data.get("companyName") or "",
            "salary_raw": salary,
            "experience_raw": f"{years} năm" if years is not None else "",
            "location_raw": location,
            "posted_date": cls._date(data.get("approvedOn")),
            "deadline_date": cls._date(data.get("expiredOn")),
            "job_description_raw": cls._text(BeautifulSoup(data.get("jobDescription") or "", "html.parser")),
            "job_requirements_raw": cls._text(BeautifulSoup(data.get("jobRequirement") or "", "html.parser")),
            "url": url,
        }

    def crawl_jobs_from_url_file(
        self, url_file_path: str = DEFAULT_URL_FILE, save_path: str = DEFAULT_JOB_FILE,
        limit: int | None = None, save_every: int = 50, resume: bool = True,
        *, archived: bool = False,
    ) -> pd.DataFrame:
        return super().crawl_jobs_from_url_file(url_file_path, save_path, limit, save_every, resume, archived=archived)

    def collect_historical_urls(
        self, start_year: int = 2022, end_year: int = 2025,
        save_path: str = ARCHIVE_URL_FILE, page_size: int = 1000, resume: bool = True,
    ) -> pd.DataFrame:
        return super().collect_historical_urls(start_year, end_year, save_path, page_size, resume)

    def crawl_historical_jobs(
        self, url_file_path: str = ARCHIVE_URL_FILE, save_path: str = ARCHIVE_JOB_FILE,
        limit: int | None = None, save_every: int = 50, resume: bool = True,
    ) -> pd.DataFrame:
        return self.crawl_jobs_from_url_file(url_file_path, save_path, limit, save_every, resume, archived=True)
