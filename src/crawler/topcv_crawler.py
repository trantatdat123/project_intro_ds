import json
import os
import random
import re
import sys
import time
import unicodedata

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests
from tqdm import tqdm

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TECH_JOB_KEYWORDS = [
    'data', 'analyst', 'analytics', 'software', 'cntt', 'ai', 'ml', 'dl', 'nlp', 'cv', 'da', 'ds',
    'de', 'bi', 'dwh', 'etl', 'elt', 'ba', 'qa', 'qc', 'pqa', 'tester', 'dev', 'engineer',
    'developer', 'pm', 'po', 'sm', 'scrum', 'agile', 'brse', 'comtor', 'it', 'tech', 'architect',
    'admin', 'helpdesk', 'support', 'specialist', 'consultant', 'officer', 'auditor', 'intern',
    'trainee', 'fresher', 'junior', 'mid', 'middle', 'senior', 'sr', 'lead', 'teamlead', 'techlead',
    'staff', 'principal', 'manager', 'director', 'head', 'vp', 'cto', 'expert', 'python', 'py',
    'java', 'c', 'c++', 'c#', '.net', 'dotnet', 'js', 'ts', 'javascript', 'typescript', 'php', 'go',
    'golang', 'ruby', 'rust', 'r', 'sql', 'nosql', 'pl/sql', 't-sql', 'swift', 'kotlin', 'dart',
    'scala', 'bash', 'shell', 'html', 'css', 'genai', 'llm', 'prompt', 'rag', 'agent', 'vision',
    'chatbot', 'bot', 'spark', 'hadoop', 'kafka', 'flink', 'airflow', 'dbt', 'pytorch', 'torch',
    'tensorflow', 'tf', 'keras', 'opencv', 'yolo', 'bert', 'scikit', 'pandas', 'numpy', 'powerbi',
    'pbi', 'tableau', 'looker', 'dataops', 'mlops', 'deep learning', 'machine learning', 'react',
    'reactjs', 'native', 'vue', 'vuejs', 'angular', 'next', 'nextjs', 'nest', 'nestjs', 'node',
    'nodejs', 'express', 'django', 'flask', 'fastapi', 'spring', 'springboot', 'laravel', 'rails',
    'flutter', 'ios', 'android', 'unity', 'unreal', 'godot', 'cloud', 'aws', 'azure', 'gcp',
    'docker', 'k8s', 'kubernetes', 'ci/cd', 'cicd', 'git', 'linux', 'ubuntu', 'centos', 'unix',
    'terraform', 'ansible', 'nginx', 'security', 'secops', 'devsecops', 'pentest', 'soc', 'siem',
    'network', 'system', 'infra', 'mysql', 'postgres', 'postgresql', 'oracle', 'mssql', 'mongo',
    'mongodb', 'redis', 'elastic', 'elasticsearch', 'cassandra', 'neo4j', 'snowflake', 'bigquery',
    'redshift', 'fintech', 'edtech', 'healthtech', 'ecom', 'ecommerce', 'game', 'gaming', 'iot',
    'hardware', 'embedded', 'firmware', 'blockchain', 'crypto', 'web3', 'fullstack', 'frontend',
    'backend', 'remote', 'hybrid', 'onsite', 'global', 'lập trình viên', 'kỹ sư', 'chuyên viên',
    'thực tập', 'thực tập sinh', 'nhân viên', 'trưởng nhóm', 'quản lý', 'phần mềm', 'dữ liệu',
    'trí tuệ nhân tạo', 'hệ thống', 'mạng', 'bảo mật', 'kiểm thử', 'cầu nối', 'an toàn thông tin',
    'an ninh mạng',
]

# Các từ mang tính cấp bậc hoặc vai trò chung, không đủ để tự xác định là việc làm IT
GENERIC_JOB_MODIFIERS = {
    'admin', 'agent', 'auditor', 'ba', 'bot', 'c', 'chuyên viên', 'consultant', 'cv', 'da', 'de',
    'director', 'expert', 'fresher', 'global', 'head', 'hybrid', 'intern', 'junior', 'kỹ sư', 'lead',
    'manager', 'mid', 'middle', 'nhân viên', 'officer', 'onsite', 'pm', 'po', 'principal', 'prompt',
    'quản lý', 'r', 'remote', 'senior', 'sm', 'soc', 'specialist', 'sr', 'staff', 'support',
    'teamlead', 'techlead', 'thực tập', 'thực tập sinh', 'trainee', 'trưởng nhóm', 'vp',
}

# Các từ khóa công nghệ nòng cốt được trích xuất từ TECH_JOB_KEYWORDS
CORE_TECH_KEYWORDS = [
    keyword.lower()
    for keyword in TECH_JOB_KEYWORDS
    if keyword.lower() not in GENERIC_JOB_MODIFIERS
]


def _norm_slug(kw: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", kw) if unicodedata.category(c) != "Mn")
    for old, new in (("đ", "d"), ("Đ", "D"), (" ", "-"), ("++", "-plus-plus"), ("#", "-sharp"), (".", "")):
        s = s.replace(old, new)
    return s.strip("-")


_SLUG_KWS = [_norm_slug(k) for k in CORE_TECH_KEYWORDS if _norm_slug(k)]
CORE_TECH_SLUG_WORDS = {k for k in _SLUG_KWS if '-' not in k}
CORE_TECH_SLUG_PHRASES = [k for k in _SLUG_KWS if '-' in k]

# Các từ khóa ngành nghề phi IT cần loại trừ khi không đi kèm kỹ năng IT đặc trưng
NON_TECH_DOMAINS = [
    'bán hàng', 'ban-hang', 'kinh doanh', 'kinh-doanh', 'bất động sản', 'bat-dong-san', 'kế toán',
    'ke-toan', 'nhân sự', 'nhan-su', 'thời trang', 'thoi-trang', 'may mặc', 'may-mac', 'xây dựng',
    'xay-dung', 'cơ khí', 'co-khi', 'bác sĩ', 'bac-si', 'nha khoa', 'nha-khoa', 'y tế', 'y-te',
    'dược', 'duoc', 'mầm non', 'mam-non', 'tiểu học', 'tieu-hoc', 'giáo viên', 'giao-vien', 'lễ tân',
    'le-tan', 'tạp vụ', 'tap-vu', 'phục vụ', 'phuc-vu', 'lái xe', 'lai-xe', 'giao hàng', 'giao-hang',
    'bếp', 'bep', 'thu ngân', 'thu-ngan', 'thi công', 'thi-cong', 'xử lý nước', 'xu-ly-nuoc',
    'cơ điện', 'co-dien', 'nội thất', 'noi-that',
]

DEFAULT_COLUMNS = [
    'job_title', 'company_name', 'salary_raw', 'experience_raw', 'location_raw', 'posted_date',
    'deadline_date', 'job_description_raw', 'job_requirements_raw', 'url',
]

# Danh sách các mốc snapshot sitemap đại diện trên Wayback Machine và TopCV Live
DEFAULT_WAYBACK_SNAPSHOTS = [
    {"year": 2022, "period": '07/2022', "sub_ts": '20220704052951',
     "sitemap_count": 16, "label": 'Wayback Snapshot T07/2022'},
    {"year": 2022, "period": '12/2022', "sub_ts": '20221201051958',
     "sitemap_count": 16, "label": 'Wayback Snapshot T12/2022'},
    {"year": 2023, "period": '03/2023', "sub_ts": '20230319132533',
     "sitemap_count": 19, "label": 'Wayback Snapshot T03/2023'},
    {"year": 2023, "period": '06/2023', "sub_ts": '20230612034015',
     "sitemap_count": 19, "label": 'Wayback Snapshot T06/2023'},
    {"year": 2025, "period": '09/2025', "sub_ts": '20250918143210',
     "sitemap_count": 100, "label": 'Wayback Snapshot T09/2025'},
    {"year": 2026, "period": 'Live/2026', "sub_ts": 'live',
     "sitemap_count": None, "label": 'Live TopCV Sitemaps (Toàn bộ ~223 sitemaps)'},
]


def _first_text(soup, selectors, default=""):
    """Tìm theo thứ tự selector, giữ thứ tự ưu tiên của các layout TopCV."""
    for selector in selectors:
        element = soup.find("h1") if selector == "h1" else soup.select_one(selector)
        if element:
            return element.get_text(strip=True)
    return default


def _job_dates(soup):
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            if not script.string:
                continue
            data = json.loads(script.string)
            for candidate in data if isinstance(data, list) else [data]:
                if not isinstance(candidate, dict):
                    continue
                entries = candidate.get("@graph", [candidate])
                entries = [entries] if isinstance(entries, dict) else entries
                entry = next((item for item in entries if isinstance(item, dict)
                              and (item.get("@type") == "JobPosting" or
                                   isinstance(item.get("@type"), list) and "JobPosting" in item["@type"])), None)
                if entry is not None:
                    dates = tuple(str(entry.get(key) or "").split("T")[0] for key in ("datePosted", "validThrough"))
                    if any(dates):
                        return dates
        except Exception:
            pass
    return "", ""


class TopCVCrawler:
    """TopCV: sitemap Wayback/live, crawl JD một luồng và checkpoint CSV + JSON."""

    def __init__(self, **kwargs):
        # Keep accepting legacy options passed by older notebooks.
        self._session: requests.Session | None = None

    def get_session(self) -> requests.Session:
        """Khởi tạo hoặc trả về session TopCV (Chrome 120 impersonation)."""
        if self._session is None:
            s = requests.Session(impersonate="chrome120")
            s.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
                "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
            })
            try:
                s.get("https://www.topcv.vn", timeout=10)
                time.sleep(0.5)
            except Exception:
                pass
            self._session = s
        return self._session

    def reset_session(self) -> requests.Session:
        """Đóng session cũ và tạo mới một session sạch."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
        self._session = None
        return self.get_session()

    @property
    def session(self) -> requests.Session:
        """Thuộc tính tương thích cho các phương thức truy xuất self.session."""
        return self.get_session()

    def warmup(self):
        """Khởi tạo session từ trang chủ để lấy cookie và token của Cloudflare."""
        self.get_session()

    @staticmethod
    def get_slug_from_url(url: str) -> str:
        """Trích xuất slug tiêu đề bài đăng từ TopCV URL."""
        parts = url.rstrip("/").split("/")
        if not parts:
            return ""
        last_part = parts[-1].split("?")[0]
        if last_part.endswith(".html"):
            name = last_part[:-5]
            if name.isdigit() and len(parts) >= 2:
                return parts[-2]
            return name
        return last_part

    @classmethod
    def is_it_job(cls, title: str) -> bool:
        """Lọc tiêu đề theo từ khóa IT/Data và loại nhiễu ngành nghề khác."""
        if not title:
            return False
        t_lower = f" {title.lower()} "

        # Nếu có từ khóa phi IT rõ ràng và không có từ khóa IT cực mạnh -> Loại bỏ
        strong_it = ["it", "cntt", "lập trình", "developer", "phần mềm", "software", "data", "tester", "devops"]
        has_strong_it = any(re.search(rf"\b{re.escape(k)}\b", t_lower) for k in strong_it)
        if not has_strong_it and any(ex in t_lower for ex in NON_TECH_DOMAINS):
            return False

        for kw in CORE_TECH_KEYWORDS:
            if len(kw) <= 3:
                pattern = rf"(?<![a-zA-Z0-9]){re.escape(kw)}(?![a-zA-Z0-9])"
                if re.search(pattern, t_lower):
                    return True
            elif kw in t_lower:
                return True
        return False

    @classmethod
    def is_it_job_slug(cls, slug: str) -> bool:
        """Kiểm tra URL slug (không dấu, cách nhau bởi gạch nối) có thuộc ngành IT hay không."""
        if not slug:
            return False
        slug_clean = re.sub(r"\.html$", "", slug.lower()).strip()
        words = set(re.split(r"[-_.]", slug_clean))

        strong_it = {"it", "cntt", "lap-trinh", "developer", "phan-mem", "software", "data", "tester", "devops"}
        non_it = ["kinh-doanh", "ban-hang", "ke-toan", "nhan-su", "thoi-trang", "xay-dung",
                  "co-khi", "bac-si", "lai-xe", "giao-hang", "phuc-vu"]
        if not (words & strong_it) and any(ex in slug_clean for ex in non_it):
            return False

        if words & CORE_TECH_SLUG_WORDS:
            return True
        return any(p in slug_clean for p in CORE_TECH_SLUG_PHRASES)

    @staticmethod
    def _parse_job_html(html: str, final_url: str, orig_url: str, slug: str) -> dict | None:
        """Bóc tách JD từ các layout TopCV, giữ schema 10 cột."""
        if not final_url.endswith(".html"):
            return None
        soup = BeautifulSoup(html, "html.parser")
        title = _first_text(soup, (
            ".box-header-job__title", ".premium-job-basic-information__content--title",
            ".job-detail__info--title", "h1",
        ))
        if not title or "tìm việc làm nhanh" in title.lower():
            return None
        posted_date, deadline_date = _job_dates(soup)

        experience, location = "", ""
        for item in soup.select(".box-header-job-list-info__item, .basic-information-item"):
            text = item.get_text(" ", strip=True)
            if "kinh nghiệm" in text.lower():
                experience = re.sub(r"^(?:kinh nghiệm)\s*", "", text, flags=re.I).strip()
            elif "địa điểm" in text.lower():
                location = re.sub(r"^(?:địa điểm)\s*", "", text, flags=re.I).strip()

        description, requirements = "", ""
        for section in soup.select(".box-job-information-detail-item, .premium-job-description__box, .job-description__item"):
            heading = section.find(["h2", "h3", "strong"])
            label = heading.get_text(strip=True).lower() if heading else ""
            content = section.get_text("\n", strip=True)
            if "mô tả" in label:
                description = content
            elif "yêu cầu" in label:
                requirements = content

        return {
            "job_title": title,
            "company_name": _first_text(soup, (
                ".company-name-label", ".box-company-info-detail__header",
                ".company-name", ".company-title", "h1.title",
            )),
            "salary_raw": _first_text(soup, (".box-header-job__salary, .salary-value",), "Thỏa thuận"),
            "experience_raw": experience, "location_raw": location,
            "posted_date": posted_date, "deadline_date": deadline_date,
            "job_description_raw": description, "job_requirements_raw": requirements,
            "url": final_url,
        }

    def parse_job(self, url: str) -> dict | None:
        """Bóc tách thông tin từ 1 link tuyển dụng đồng bộ (dùng cho test đơn lẻ)."""
        session = self.get_session()
        slug = self.get_slug_from_url(url) or url.split("/")[-1]
        try:
            resp = session.get(url, headers={"Referer": "https://www.topcv.vn/viec-lam-it"}, timeout=12)
            if resp.status_code == 200:
                final_url = str(resp.url).split("?")[0]
                return self._parse_job_html(resp.text, final_url, url, slug)
        except Exception as error:
            tqdm.write(f"⚠️ [Lỗi tải URL] {slug}: {error}")
        return None

    def _extract_sitemap_urls(self, xml_text: str, it_urls: list[str], url_date_map: dict[str, str]) -> None:
        """Trích xuất URL việc làm IT và ngày sửa trang (lastmod), không phải ngày đăng."""
        for block in re.findall(r"<url>(.*?)</url>", xml_text, re.DOTALL):
            m_loc = re.search(r"<loc>(.*?)</loc>", block)
            if not m_loc:
                continue
            loc = m_loc.group(1).strip()
            slug = self.get_slug_from_url(loc)
            if self.is_it_job_slug(slug):
                if loc not in url_date_map:
                    it_urls.append(loc)
                m_date = re.search(r"<lastmod>(.*?)</lastmod>", block)
                if m_date:
                    url_date_map[loc] = m_date.group(1).strip().split("T")[0]

    def get_it_urls_from_wayback_snapshot(
        self, sub_ts: str = "20220704052951", sitemap_count: int | None = 16,
    ) -> tuple[list[str], dict[str, str]]:
        """Trả (URL IT, lastmod); live + count=None đọc toàn bộ sitemap index."""
        live = sub_ts == "live"
        urls, lastmods, sitemaps = [], {}, []
        if live:
            self.warmup()
            try:
                response = self.session.get("https://www.topcv.vn/sitemap/jobs.xml")
                if response.status_code == 200:
                    matches = re.findall(r"<loc>(.*?)</loc>", response.text)
                    sitemaps = matches if sitemap_count is None else matches[:sitemap_count]
            except Exception:
                pass
        if not sitemaps:
            count = sitemap_count if sitemap_count is not None else (223 if live else 20)
            prefix = "" if live else f"http://web.archive.org/web/{sub_ts}id_/"
            sitemaps = [f"{prefix}https://www.topcv.vn/sitemap/jobs_{i}.xml" for i in range(count)]

        consecutive_404 = 0
        label = "Live TopCV (2026)" if live else f"Wayback ({sub_ts[:4]})"
        for sitemap in tqdm(sitemaps, desc=label):
            text = ""
            for attempt in range(1 if live else 3):
                try:
                    response = self.session.get(sitemap, timeout=10 if live else 20)
                    code = response.status_code
                    if live:
                        if code == 200:
                            self._extract_sitemap_urls(response.text, urls, lastmods)
                        break
                    if code == 200:
                        text, consecutive_404 = response.text, 0
                        break
                    if code == 404:
                        consecutive_404 += 1
                    elif code == 429:
                        time.sleep(3 + attempt * 2)
                        continue
                    break
                except Exception:
                    if not live:
                        time.sleep(1.5)
            if live:
                time.sleep(random.uniform(1.5, 3.0))
            else:
                if consecutive_404 >= 3:
                    break
                self._extract_sitemap_urls(text, urls, lastmods)
                time.sleep(0.2)
        return urls, lastmods

    def collect_urls_from_sitemaps(
        self, snapshots: list[dict] | None = None,
        save_path: str = "data/raw/topcv_it_job_urls.csv",
    ) -> pd.DataFrame:
        """Gom URL IT, giữ metadata sitemap của lần xuất hiện đầu tiên."""
        snapshots = DEFAULT_WAYBACK_SNAPSHOTS if snapshots is None else snapshots
        records = {}
        for snapshot in snapshots:
            year = snapshot["year"]
            period = snapshot.get("period", str(year))
            label = snapshot.get("label", f"Năm {year}")
            urls, lastmods = self.get_it_urls_from_wayback_snapshot(
                sub_ts=snapshot["sub_ts"], sitemap_count=snapshot.get("sitemap_count"),
            )
            before = len(records)
            for url in urls:
                if url not in records:
                    records[url] = {
                        "url": url, "sitemap_lastmod": lastmods.get(url, ""),
                        "snapshot_period": period, "year": year, "snapshot_label": label,
                    }
            print(f"{label}: {len(urls):,} URL, thêm {len(records) - before:,}; tổng {len(records):,}")
        result = pd.DataFrame(records.values())
        if save_path and not result.empty:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            result.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"Đã gom {len(result):,} URL tại {save_path}")
            print(result.year.value_counts().sort_index().to_string())
        return result

    def _request_detail(self, session, url: str, slug: str):
        """Trả (response, stop); giữ nguyên nhịp request và ba lượt retry."""
        for attempt in range(3):
            time.sleep(random.uniform(3, 6))
            try:
                response = session.get(
                    url, headers={"Referer": "https://www.topcv.vn/viec-lam-it"}, timeout=20,
                )
            except Exception as exc:
                tqdm.write(f"Request lỗi {slug}: {exc}")
                time.sleep(3.0 * (attempt + 1))
                continue
            code = response.status_code
            if code == 403:
                tqdm.write(f"HTTP 403: dừng, giữ URL {slug} để thử sau")
                return None, True
            if code == 429:
                tqdm.write(f"HTTP 429 {slug}: chờ 60s trước khi retry")
                time.sleep(60.0)
            elif code == 421 or code >= 500:
                tqdm.write(f"HTTP {code} {slug}: retry {attempt + 1}/3")
                if attempt < 2:
                    time.sleep(3.0 * (attempt + 1))
            else:
                return response, False
            if attempt == 2:
                return None, True
        return None, False

    def crawl_jobs_from_url_file(
        self, url_file_path: str = "data/raw/topcv_it_job_urls.csv",
        save_path: str = "data/raw/topcv_it_jobs_wayback.csv", limit: int | None = None,
        min_date: str | None = "2020-01-01", max_date: str | None = None,
        save_every: int = 50, resume: bool = True, **kwargs,
    ) -> pd.DataFrame:
        """Crawl JD một luồng; CSV và state tương thích các lượt crawl trước."""
        if not os.path.exists(url_file_path):
            raise FileNotFoundError(f"Không tìm thấy {url_file_path}; chạy collect_urls_from_sitemaps trước")
        df_urls = pd.read_csv(url_file_path)
        if df_urls.empty or "url" not in df_urls:
            print("File URL rỗng hoặc thiếu cột url")
            return pd.DataFrame(columns=DEFAULT_COLUMNS)

        state_path = f"{os.path.splitext(save_path)[0]}.state.json" if save_path else None
        results, evaluated_urls = [], set()
        if resume and save_path and os.path.exists(save_path):
            try:
                existing = pd.read_csv(save_path)
                results = existing.to_dict("records")
                if "url" in existing:
                    evaluated_urls.update(existing.url.dropna())
            except Exception as exc:
                print(f"Không đọc được CSV checkpoint: {exc}")
        if resume and state_path and os.path.exists(state_path):
            try:
                with open(state_path, encoding="utf-8") as file:
                    evaluated_urls.update(json.load(file).get("evaluated_urls", []))
            except Exception:
                pass

        pending = [url for url in df_urls.url if url not in evaluated_urls]
        if limit:
            pending = pending[:limit]
        print(f"TopCV: {len(df_urls):,} URL, {len(evaluated_urls):,} đã xử lý, {len(pending):,} lượt này")
        print("Một luồng, nghỉ 3–6 giây giữa các request")
        if not pending:
            return pd.DataFrame(results, columns=DEFAULT_COLUMNS)

        def checkpoint():
            if save_path and results:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                pd.DataFrame(results, columns=DEFAULT_COLUMNS).to_csv(save_path, index=False, encoding="utf-8-sig")
            if state_path:
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
                    with open(state_path, "w", encoding="utf-8") as file:
                        json.dump({
                            "total_crawled": len(results), "total_evaluated": len(evaluated_urls),
                            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "evaluated_urls": list(evaluated_urls),
                        }, file, indent=2)
                except Exception as exc:
                    print(f"Không lưu được state: {exc}")

        progress = tqdm(total=len(pending), desc="Cào tuần tự")
        session = self.get_session()
        stop = False
        try:
            for url in pending:
                slug = self.get_slug_from_url(url) or url.split("/")[-1]
                completed = False
                try:
                    response, stop = self._request_detail(session, url, slug)
                    if response is not None:
                        if response.status_code == 404:
                            completed = True
                        elif response.status_code != 200:
                            tqdm.write(f"HTTP {response.status_code} {slug}: giữ URL để thử sau")
                        else:
                            final_url = str(response.url).split("?")[0]
                            job = self._parse_job_html(response.text, final_url, url, slug)
                            accepted = job and job.get("job_title") and self.is_it_job(job["job_title"])
                            if accepted:
                                date = job.get("posted_date", "")
                                outside_dates = date and ((min_date and date < min_date) or (max_date and date > max_date))
                                if not outside_dates:
                                    results.append(job)
                                    completed = True
                                    tqdm.write(f"Đã lưu: {job['job_title']}")
                                    if save_every > 0 and len(results) % save_every == 0:
                                        checkpoint()
                            completed = True
                except KeyboardInterrupt:
                    stop = True
                except Exception as exc:
                    tqdm.write(f"Lỗi xử lý {slug}: {exc}")
                finally:
                    if completed:
                        evaluated_urls.add(url)
                    progress.update(1)
                    progress.set_postfix(saved_it=len(results))
                if stop:
                    tqdm.write("Dừng crawl; giữ URL chưa xử lý cho lần sau")
                    break
        except KeyboardInterrupt:
            print("Nhận Ctrl+C; đang lưu checkpoint")
        finally:
            progress.close()
            checkpoint()

        result = pd.DataFrame(results, columns=DEFAULT_COLUMNS)
        if save_path and not result.empty:
            print(f"Đã lưu {len(result):,} tin tại {save_path}")
            dates = result.loc[result.posted_date.str.len() >= 4, "posted_date"]
            if not dates.empty:
                print(dates.str[:4].value_counts().sort_index().to_string())
        return result

    def crawl_from_wayback_sitemaps(
        self, snapshots: list[dict] | None = None,
        url_file_path: str = "data/raw/topcv_it_job_urls.csv",
        save_path: str = "data/raw/topcv_it_jobs_wayback.csv", limit: int | None = None,
        min_date: str | None = "2020-01-01", max_date: str | None = None,
        save_every: int = 50, resume: bool = True, **kwargs,
    ) -> pd.DataFrame:
        """Gom URL nếu chưa có file, sau đó crawl JD tuần tự."""
        if not os.path.exists(url_file_path):
            print(f"-> File danh sách URL chưa tồn tại ({url_file_path}). Đang tiến hành gom URL trước...")
            self.collect_urls_from_sitemaps(snapshots=snapshots, save_path=url_file_path)

        return self.crawl_jobs_from_url_file(
            url_file_path=url_file_path, save_path=save_path, limit=limit,
            min_date=min_date, max_date=max_date, save_every=save_every,
            resume=resume, **kwargs,
        )


if __name__ == "__main__":
    crawler = TopCVCrawler()
    print("TopCVCrawler đã sẵn sàng!")
