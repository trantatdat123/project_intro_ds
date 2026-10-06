"""TopCV: sitemap -> lọc IT/Data -> file link -> JSONL theo schema TinixAI.

Chỉ lấy link IT/CNTT/Data Science có lastmod năm 2026 (mặc định).
Import và gọi các hàm từ notebooks/topcv_it_jobs.ipynb.

Chạy lại crawl sẽ bỏ qua tin đã lưu thành công.
"""

from __future__ import annotations

import html
import json
import random
import re
import sys
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup
from curl_cffi import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data/raw/topcv"
RAW_JSONL = RAW_DIR / "topcv_jobs_raw.jsonl"
JOBS_SITEMAP_URL = "https://www.topcv.vn/sitemap/jobs.xml"
TINIXAI_COLUMNS = (
    "id", "job_title", "company_name", "salary", "location", "job_type",
    "job_industry", "experience_level", "education_level", "job_position",
    "job_description", "benefits", "requirements", "year",
)

# Khớp nguyên từ/cụm từ; không dùng "bi" đơn lẻ vì trùng "thiết bị".
IT_TERMS = [
    'information technology', 'software', 'developer', 'programmer',
    'frontend', 'front end', 'backend', 'back end',
    'fullstack', 'full stack', 'devops', 'devsecops',
    'platform engineer', 'cloud engineer', 'cloud architect', 'cloud computing',
    'system engineer', 'systems engineer', 'system administrator', 'systems administrator',
    'network engineer', 'network administrator', 'network architect', 'database administrator',
    'database engineer', 'database developer', 'cybersecurity', 'cyber security',
    'information security', 'security engineer', 'security analyst', 'pentester',
    'penetration tester', 'software tester', 'manual tester', 'automation tester',
    'qa engineer', 'qa automation', 'test automation', 'helpdesk',
    'help desk', 'it support', 'it business analyst', 'business systems analyst',
    'embedded', 'firmware', 'webmaster', 'bridge system engineer',
    'solution architect', 'solutions architect', 'technical architect', 'sap consultant',
    'erp consultant', 'python', 'java', 'javascript',
    'typescript', 'php', 'golang', 'c++',
    'c#', '.net developer', '.net engineer', 'dotnet',
    'react', 'reactjs', 'nodejs', 'node.js',
    'vuejs', 'angular', 'flutter', 'kotlin',
    'android developer', 'ios developer', 'công nghệ thông tin', 'lập trình',
    'phần mềm', 'quản trị mạng', 'quản trị hệ thống', 'quản trị cơ sở dữ liệu',
    'an ninh mạng', 'an toàn thông tin', 'bảo mật thông tin', 'kiểm thử phần mềm',
    'hỗ trợ kỹ thuật it', 'kỹ sư cầu nối', 'data science', 'data scientist',
    'data analyst', 'data analytics', 'data analysis', 'data engineer',
    'data engineering', 'data architect', 'data architecture', 'data warehouse',
    'data warehousing', 'data platform', 'data modeling', 'data modelling',
    'data governance', 'data quality', 'analytics engineer', 'analytics engineering',
    'business intelligence', 'power bi', 'powerbi', 'bi developer',
    'bi analyst', 'bi engineer', 'bi specialist', 'reporting analyst',
    'big data', 'bigdata', 'etl', 'elt',
    'khoa học dữ liệu', 'phân tích dữ liệu', 'phân tích số liệu', 'kỹ sư dữ liệu',
    'kiến trúc dữ liệu', 'kho dữ liệu', 'khai phá dữ liệu', 'it',
    'cntt', 'sre', 'dba', 'cto',
    'ciso', 'brse', 'tester', 'kiểm thử',
    'business analyst', 'phân tích nghiệp vụ', 'system admin', 'network admin',
    'kỹ sư mạng', 'kỹ sư hệ thống', 'ai engineer', 'ml engineer',
    'machine learning engineer', 'kỹ sư ai',
]


# --------------------------------------------------------------------------- HTTP

_thread_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session(impersonate="chrome")
        _thread_local.session.headers.update({"Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8"})
    return _thread_local.session


_cooldown_lock = threading.Lock()
_cooldown_until = 0.0


class RateLimitError(RuntimeError):
    """TopCV tiếp tục chặn sau khi đã chờ và thử lại."""


def _wait_cooldown() -> None:
    while (remaining := _cooldown_until - time.monotonic()) > 0:
        time.sleep(min(remaining, 5))


def _start_cooldown(seconds: float) -> None:
    """Khi 1 luồng bị 429, tất cả luồng cùng tạm dừng để TopCV 'nguội' lại."""
    global _cooldown_until
    with _cooldown_lock:
        _cooldown_until = max(_cooldown_until, time.monotonic() + seconds)


def fetch(url: str, *, delay: float, retries: int = 6) -> str | None:
    """GET có retry + backoff. Trả về None nếu tin không còn (404/410)."""
    for attempt in range(retries + 1):
        _wait_cooldown()
        time.sleep(delay * random.uniform(0.7, 1.3))
        resp = None
        try:
            resp = get_session().get(url, timeout=40)
        except Exception as exc:  # lỗi mạng
            err = repr(exc)
        else:
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (404, 410):
                return None
            err = f"HTTP {resp.status_code}"
            if resp.status_code in (403, 429):
                # Bị chặn/giới hạn: bỏ session cũ (cookie) để lần sau tạo mới
                _thread_local.__dict__.pop("session", None)
        if attempt == retries:
            break
        retry_after = resp.headers.get("retry-after") if resp is not None else None
        retry_delay = int(retry_after) if retry_after and retry_after.isdigit() else min(180, 15 * 2 ** attempt)
        if err in ("HTTP 429", "HTTP 403"):
            _start_cooldown(retry_delay)
        tqdm.write(f"[retry {attempt + 1}/{retries}] {url} -> {err}; chờ {retry_delay}s", file=sys.stderr)
        time.sleep(retry_delay)
    if err in ("HTTP 403", "HTTP 429"):
        raise RateLimitError(f"Không tải được {url}: {err} sau {retries + 1} lần thử")
    raise RuntimeError(f"Không tải được {url}: {err}")


# ------------------------------------------------------------------- text helpers

def clean_text(value) -> str:
    """Chuyển HTML/text thành văn bản thuần."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return clean_text(" ".join(clean_text(item) for item in value))
    if isinstance(value, dict):
        return clean_text(value.get("name") or value.get("value") or "")
    if hasattr(value, "get_text"):
        value = value.get_text(" ")
    elif not isinstance(value, str):
        value = str(value)
    if "<" in value:
        value = BeautifulSoup(value, "lxml").get_text(" ")
    value = unicodedata.normalize("NFC", html.unescape(value)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def job_id_from_url(url: str) -> int | None:
    match = re.search(r"/(?:j)?(\d+)\.html", url)
    return int(match.group(1)) if match else None


# ----------------------------------------------------------- links from sitemap

def normalize_topcv_job_url(url: str) -> str:
    """Chuẩn hóa URL việc làm lấy từ sitemap hoặc file link."""
    parts = urlsplit(url.strip())
    if (parts.scheme not in ("http", "https")
            or parts.hostname not in ("topcv.vn", "www.topcv.vn")
            or not parts.path.startswith("/viec-lam/")
            or job_id_from_url(parts.path) is None):
        raise ValueError(f"Không phải URL việc làm TopCV: {url}")
    return urlunsplit(("https", "www.topcv.vn", parts.path.rstrip("/"), "", ""))


def normalize_title(text: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    text = "".join(char for char in unicodedata.normalize("NFD", text)
                   if unicodedata.category(char) != "Mn").replace("đ", "d")
    return re.sub(r"[^a-z0-9+#]+", " ", text).strip()


def sitemap_title_pattern() -> re.Pattern:
    terms = {normalize_title(term) for term in IT_TERMS}
    choices = "|".join(re.escape(term) for term in sorted(terms, key=lambda term: (-len(term), term)))
    return re.compile(rf"(?<![a-z0-9+#])(?:{choices})(?![a-z0-9+#])")


def job_url_matches_scope(url: str, pattern: re.Pattern) -> bool:
    """Khớp tên công việc trong slug, không khớp domain/query/ID."""
    slug = unquote(urlsplit(url).path.rstrip("/").rsplit("/", 2)[-2])
    return pattern.search(normalize_title(slug)) is not None


def filter_job_urls(urls: Iterable[str]) -> list[str]:
    """Lọc IT/CNTT/Data Science từ danh sách link sẵn có, không gửi request."""
    pattern = sitemap_title_pattern()
    kept = {}
    for url in urls:
        try:
            normalized = normalize_topcv_job_url(url)
        except ValueError:
            continue
        if job_url_matches_scope(normalized, pattern):
            kept[normalized] = None
    return list(kept)


def job_urls_file(year: int | None = 2026) -> Path:
    """Đường dẫn file link theo năm."""
    suffix = f"_{year}" if year is not None else ""
    return RAW_DIR / f"topcv_sitemap_it_data_urls{suffix}.txt"


def load_job_urls(year: int = 2026) -> list[str]:
    """Đọc file link đã lưu, không truy cập mạng."""
    path = job_urls_file(year)
    urls = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not urls:
        raise ValueError(f"File link rỗng: {path}")
    return urls


def save_job_urls(urls: list[str], destination: Path | str) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")
    temporary.replace(destination)


def iter_job_urls_from_sitemap(*, delay: float = 2.0, year: int | None = 2026) -> Iterable[str]:
    """Duyệt sitemap/jobs.xml và mọi sitemap con, yield URL việc làm duy nhất.

    Lọc theo năm lastmod trong sitemap (ngày cập nhật, không phải ngày đăng).
    year=None lấy mọi năm. Khi lọc năm, bỏ URL thiếu lastmod hoặc ngày không hợp lệ.
    Lọc IT/CNTT/Data Science theo slug ngay khi đọc URL, trước khi tải chi tiết.
    """
    if delay < 0:
        raise ValueError("delay phải >= 0")
    if year is not None and not 1 <= year <= 9999:
        raise ValueError("year phải trong khoảng 1..9999 hoặc None")
    title_pattern = sitemap_title_pattern()
    queue = deque([JOBS_SITEMAP_URL])
    enqueued = {JOBS_SITEMAP_URL}
    seen_urls = set()
    skipped_undated = 0
    with tqdm(total=1, desc="Sitemap TopCV", unit="file") as bar:
        while queue:
            sitemap_url = queue.popleft()
            xml_text = fetch(sitemap_url, delay=delay)
            if xml_text is None:
                raise RuntimeError(f"Sitemap không tồn tại: {sitemap_url}")
            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError as exc:
                raise ValueError(f"Sitemap XML không hợp lệ: {sitemap_url}") from exc

            kind = root.tag.rsplit("}", 1)[-1]
            if kind == "sitemapindex":
                children = [loc.text.strip() for loc in root.findall("{*}sitemap/{*}loc") if loc.text]
                if not children:
                    raise ValueError(f"Sitemap index rỗng: {sitemap_url}")
                for child in children:
                    child_url = urljoin(sitemap_url, child)
                    child_parts = urlsplit(child_url)
                    if (child_parts.scheme != "https"
                            or child_parts.hostname not in ("topcv.vn", "www.topcv.vn")):
                        raise ValueError(f"Sitemap con ngoài TopCV: {child_url}")
                    if child_url not in enqueued:
                        queue.append(child_url)
                        enqueued.add(child_url)
                        bar.total += 1
                bar.refresh()
            elif kind == "urlset":
                for item in root:
                    if item.tag.rsplit("}", 1)[-1] != "url":
                        continue
                    fields = {child.tag.rsplit("}", 1)[-1]: (child.text or "").strip()
                              for child in item}
                    location = fields.get("loc", "")
                    try:
                        normalized = normalize_topcv_job_url(location)
                    except ValueError:
                        continue
                    if year is not None:
                        try:
                            lastmod_year = datetime.fromisoformat(fields.get("lastmod", "").replace("Z", "+00:00")).year
                        except ValueError:
                            skipped_undated += 1
                            continue
                        if lastmod_year != year:
                            continue
                    if not job_url_matches_scope(normalized, title_pattern):
                        continue
                    if normalized not in seen_urls:
                        seen_urls.add(normalized)
                        yield normalized
            else:
                raise ValueError(f"Không nhận diện được sitemap XML: {sitemap_url}")
            bar.update(1)
    if skipped_undated:
        print(f"Bỏ {skipped_undated:,} URL thiếu lastmod/ngày không hợp lệ khi lọc năm {year}.")


def get_job_urls_from_sitemap(year: int | None = 2026, *, delay: float = 2.0,
                              output_file: Path | str | None = None) -> list[str]:
    """Lấy link IT/CNTT/Data Science có lastmod thuộc year, lưu TXT để crawl sau."""
    urls = list(iter_job_urls_from_sitemap(delay=delay, year=year))
    if not urls:
        raise ValueError(f"Sitemap TopCV không có URL IT/CNTT/Data Science với bộ lọc năm {year}")
    destination = Path(output_file) if output_file is not None else job_urls_file(year)
    save_job_urls(urls, destination)
    print(f"Đã lưu {len(urls):,} link: {destination}")
    return urls


# -------------------------------------------------------------------- detail page

def find_job_posting(soup: BeautifulSoup) -> dict:
    def walk(value):
        if isinstance(value, dict):
            kind = value.get("@type")
            if kind == "JobPosting" or isinstance(kind, list) and "JobPosting" in kind:
                yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text()
        if "JobPosting" not in raw:
            continue
        try:
            data = json.loads(raw, strict=False)
        except json.JSONDecodeError:
            continue
        for obj in walk(data):
            return obj
    return {}


def find_label_value(soup: BeautifulSoup, label: str) -> str:
    """Tìm ô kiểu <div class="...title|label">Học vấn</div><div>Đại học</div>.

    Hoạt động với cả layout thường (``box-job-information-*``) và layout
    trang brand/premium (``general-information-data__*``).
    """
    pattern = re.compile(rf"^\s*{re.escape(label)}\s*:?\s*$", re.IGNORECASE)
    for node in soup.find_all(string=pattern):
        parent = node.parent
        classes = " ".join(parent.get("class") or [])
        if parent.name not in ("div", "span", "p") or not re.search(r"title|label", classes):
            continue
        sibling = parent.find_next_sibling()
        value = clean_text(sibling) if sibling else ""
        if value and "{{" not in value:
            return value
    return ""


def split_description(description_html: str) -> dict[str, str]:
    """JSON-LD description gồm các khối <h2>Mô tả công việc</h2>, <h2>Yêu cầu ứng viên</h2>, <h2>Quyền lợi...</h2>."""
    soup = BeautifulSoup(description_html or "", "lxml")
    sections: dict[str, list[str]] = {}
    current = "description"
    body = soup.body or soup
    for element in body.find_all(recursive=False):
        if element.name in ("h2", "h3"):
            heading = normalize_title(element.get_text(" "))
            if "mo ta" in heading:
                current = "description"
            elif "yeu cau" in heading:
                current = "requirements"
            elif "quyen loi" in heading or "phuc loi" in heading:
                current = "benefits"
            else:
                current = "other"
            continue
        sections.setdefault(current, []).append(element.get_text(" "))
    return {key: clean_text(" ".join(parts)) for key, parts in sections.items()}


def parse_detail(text: str, listing: dict) -> dict:
    soup = BeautifulSoup(text, "lxml")
    posting = find_job_posting(soup)
    if not posting or not (posting.get("title") or posting.get("name")):
        raise ValueError(f"Trang chi tiết không chứa JobPosting hợp lệ: {listing['url']}")
    sections = split_description(posting.get("description", ""))
    if not sections.get("description"):
        raise ValueError(f"Trang chi tiết không có mô tả công việc: {listing['url']}")

    base_salary = posting.get("baseSalary") or {}
    salary_value = base_salary.get("value") or {} if isinstance(base_salary, dict) else base_salary
    salary_jsonld = salary_value.get("value") if isinstance(salary_value, dict) else salary_value
    if salary_jsonld is None and isinstance(salary_value, dict) and salary_value.get("minValue"):
        salary_jsonld = f"{salary_value.get('minValue')} - {salary_value.get('maxValue')}"

    job_location = posting.get("jobLocation") or {}
    if isinstance(job_location, list):
        job_location = job_location[0] if job_location else {}
    address = job_location.get("address") or {} if isinstance(job_location, dict) else {}
    if not isinstance(address, dict):
        address = {}
    organization = posting.get("hiringOrganization") or {}
    if isinstance(organization, list):
        organization = organization[0] if organization else {}
    employment = posting.get("employmentType")
    employment = employment[0] if isinstance(employment, list) and employment else employment
    job_type = find_label_value(soup, "Loại hình làm việc") or {
        "FULL_TIME": "Toàn thời gian",
        "PART_TIME": "Bán thời gian",
        "INTERN": "Thực tập",
    }.get(employment, clean_text(employment))
    posted = str(posting.get("datePosted") or "")
    try:
        year = datetime.fromisoformat(posted.replace("Z", "+00:00")).year
    except ValueError:
        year = None

    return {
        "id": int(listing["job_id"]),
        "job_title": clean_text(posting.get("title") or posting.get("name")),
        "company_name": clean_text(organization.get("name") if isinstance(organization, dict) else organization),
        "salary": clean_text(str(salary_jsonld)) if salary_jsonld is not None else "",
        "location": find_label_value(soup, "Địa điểm") or clean_text(address.get("addressRegion")),
        "job_type": job_type,
        "job_industry": clean_text(posting.get("industry")),
        "experience_level": find_label_value(soup, "Kinh nghiệm"),
        "education_level": find_label_value(soup, "Học vấn"),
        "job_position": clean_text(posting.get("occupationalCategory")) or find_label_value(soup, "Cấp bậc"),
        "job_description": sections.get("description", ""),
        "benefits": sections.get("benefits", "") or clean_text(posting.get("jobBenefits")),
        "requirements": sections.get("requirements", ""),
        "year": year,
    }


# ----------------------------------------------------------- crawl and save

def load_done_ids() -> set[int]:
    if not RAW_JSONL.exists():
        return set()
    done = set()
    with RAW_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                if record.get("job_description"):
                    done.add(int(record["id"]))
            except (json.JSONDecodeError, AttributeError, KeyError, ValueError):
                continue
    return done


def ensure_jsonl_line_boundary() -> None:
    """Không nối bản ghi mới vào một dòng cuối bị dở sau khi chương trình bị ngắt."""
    if not RAW_JSONL.exists() or RAW_JSONL.stat().st_size == 0:
        return
    with RAW_JSONL.open("rb+") as handle:
        handle.seek(-1, 2)
        if handle.read(1) != b"\n":
            handle.write(b"\n")


def crawl_from_links(urls: Iterable[str], *, workers: int = 2, delay: float = 2.0,
                     refresh: bool = False) -> bool:
    """Tải chi tiết trực tiếp từ link đã lấy; không quét trang danh sách/sitemap."""
    listings = {}
    if isinstance(urls, str):
        urls = [urls]
    for url in urls:
        normalized = normalize_topcv_job_url(url)
        job_id = job_id_from_url(normalized)
        listings[job_id] = {"job_id": job_id, "url": normalized}
    if workers < 1 or delay < 0:
        raise ValueError("workers phải >= 1 và delay phải >= 0")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ensure_jsonl_line_boundary()
    done = set() if refresh else load_done_ids()
    todo = [job for job in listings.values() if job["job_id"] not in done]
    print(f"Tìm thấy {len(listings):,} tin; đã có {len(listings) - len(todo):,}; cần tải {len(todo):,}")

    failed = []
    rate_limited = False

    def work(listing: dict):
        text = fetch(listing["url"], delay=delay)
        return None if text is None else parse_detail(text, listing)

    with RAW_JSONL.open("a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=workers) as pool, \
            tqdm(total=len(todo), desc="Chi tiết", unit="tin") as bar:
        remaining = iter(todo)
        futures = {}

        while True:
            if not rate_limited:
                while len(futures) < workers:
                    job = next(remaining, None)
                    if job is None:
                        break
                    futures[pool.submit(work, job)] = job
            if not futures:
                break
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                job = futures.pop(future)
                try:
                    record = future.result()
                except Exception as exc:
                    failed.append(job["url"])
                    if isinstance(exc, RateLimitError):
                        rate_limited = True
                        tqdm.write(f"TopCV đang giới hạn truy cập: {exc}")
                    else:
                        tqdm.write(f"Lỗi {job['url']}: {exc}")
                else:
                    if record:
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out.flush()
                bar.update(1)
            if rate_limited:
                for future in list(futures):
                    if future.cancel():
                        futures.pop(future)
    if rate_limited:
        print("Dừng tải chi tiết do TopCV giới hạn truy cập; chạy lại để tiếp tục từ JSONL.")
    if failed:
        print(f"{len(failed)} tin lỗi (chạy lại cell crawl để thử lại):")
        for url in failed[:20]:
            print("  ", url)
    return not failed

