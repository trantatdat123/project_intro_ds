"""Offline checks for TopCV parsing and interrupted-crawl recovery."""

import json

import pytest

from src.crawlers import topcv_it_jobs as crawler


def test_nested_job_posting_is_found_and_parsed():
    listing = {"job_id": 123, "url": "https://www.topcv.vn/viec-lam/example/123.html"}
    posting = {
        "@context": "https://schema.org",
        "@graph": [{"@type": ["Thing", "JobPosting"], "title": "Data Engineer",
                    "description": "<h2>Mô tả công việc</h2><p>Xây dựng pipeline</p>",
                    "hiringOrganization": [{"name": "Công ty mẫu"}],
                    "jobLocation": [{"address": {"addressRegion": "Hà Nội"}}],
                    "industry": ["Công nghệ thông tin", "Dữ liệu"]}],
    }
    page = f'<script type="application/ld+json">{json.dumps(posting)}</script>'

    record = crawler.parse_detail(page, listing)

    assert record["title"] == "Data Engineer"
    assert record["job_description"] == "Xây dựng pipeline"
    assert record["company"] == "Công ty mẫu"
    assert record["address_region"] == "Hà Nội"
    assert record["industry"] == "Công nghệ thông tin Dữ liệu"


@pytest.mark.parametrize("page", ["<html>blocked</html>",
                                      '<script type="application/ld+json">'
                                      '{"@type":"JobPosting","title":"Empty"}'
                                      '</script>'])
def test_invalid_detail_is_not_saved_as_completed(page):
    with pytest.raises(ValueError):
        crawler.parse_detail(page, {"url": "https://www.topcv.vn/example.html"})


def test_resume_skips_corrupt_and_incomplete_records(tmp_path, monkeypatch):
    raw = tmp_path / "jobs.jsonl"
    raw.write_text(
        json.dumps({"job_id": 1, "job_description": "Đủ dữ liệu"}, ensure_ascii=False) + "\n"
        + json.dumps({"job_id": 2}) + "\n"
        + '{"job_id": 3', encoding="utf-8"
    )
    monkeypatch.setattr(crawler, "RAW_JSONL", raw)

    assert crawler.load_done_ids() == {1}
    crawler.ensure_jsonl_line_boundary()
    with raw.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"job_id": 3, "job_description": "Đủ dữ liệu"}, ensure_ascii=False) + "\n")
    assert crawler.load_done_ids() == {1, 3}


def test_rate_limit_stops_submitting_more_details(tmp_path, monkeypatch):
    monkeypatch.setattr(crawler, "RAW_DIR", tmp_path)
    monkeypatch.setattr(crawler, "RAW_JSONL", tmp_path / "jobs.jsonl")
    listings = [{"job_id": job_id, "url": f"https://www.topcv.vn/viec-lam/data-engineer/{job_id}.html"}
                for job_id in range(1, 4)]
    fetched = []

    def blocked(url, *, delay):
        fetched.append(url)
        raise crawler.RateLimitError("HTTP 429")

    monkeypatch.setattr(crawler, "fetch", blocked)
    assert crawler.crawl_from_links([job["url"] for job in listings], workers=1, delay=0) is False
    assert fetched == [listings[0]["url"]]


@pytest.mark.parametrize("namespace", ['', 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'])
def test_sitemap_extracts_all_links_deduplicates_and_saves(tmp_path, monkeypatch, namespace):
    root_url = crawler.JOBS_SITEMAP_URL
    child0 = "https://www.topcv.vn/sitemap/jobs_0.xml"
    child1 = "https://www.topcv.vn/sitemap/jobs_1.xml"
    job0 = "https://www.topcv.vn/viec-lam/data-engineer/123.html"
    job1 = "https://www.topcv.vn/viec-lam/ml-engineer/456.html"
    pages = {
        root_url: f"<sitemapindex {namespace}><sitemap><loc>{child0}</loc></sitemap>"
                  f"<sitemap><loc>{child1}</loc></sitemap></sitemapindex>",
        child0: f"<urlset {namespace}><url><loc>{job0}</loc></url></urlset>",
        child1: f"<urlset {namespace}><url><loc>{job1}</loc></url>"
                f"<url><loc>{job0}?source=test</loc></url>"
                "<url><loc>https://www.topcv.vn/viec-lam/ke-toan/789.html</loc></url>"
                "<url><loc>https://other.example/viec-lam/example/999.html</loc></url></urlset>",
    }
    fetched = []

    def fake_fetch(url, *, delay):
        fetched.append(url)
        return pages[url]

    monkeypatch.setattr(crawler, "fetch", fake_fetch)
    output = tmp_path / "links.txt"
    assert crawler.get_job_urls_from_sitemap(delay=0, year=None, output_file=output) == [job0, job1]
    assert output.read_text(encoding="utf-8").splitlines() == [job0, job1]
    assert fetched == [root_url, child0, child1]


def test_sitemap_defaults_to_2026_and_excludes_unknown_dates(tmp_path, monkeypatch):
    def entry(job_id, lastmod=None):
        date = f"<lastmod>{lastmod}</lastmod>" if lastmod is not None else ""
        return f"<url><loc>https://www.topcv.vn/viec-lam/data-engineer/{job_id}.html</loc>{date}</url>"

    xml = "<urlset>" + "".join([
        entry(1, "2025-12-31T23:59:59Z"),
        entry(2, "2026-01-01T00:00:00Z"),
        entry(3, "2026-12-31"),
        entry(4, "2027-01-01"),
        entry(5),
        entry(6, "2026-invalid"),
        entry(1, "2026-10-05"),
    ]) + "</urlset>"
    monkeypatch.setattr(crawler, "fetch", lambda url, *, delay: xml)
    monkeypatch.setattr(crawler, "RAW_DIR", tmp_path)
    urls = crawler.get_job_urls_from_sitemap(delay=0)
    assert [crawler.job_id_from_url(url) for url in urls] == [2, 3, 1]
    assert (tmp_path / "topcv_sitemap_it_data_urls_2026.txt").read_text(encoding="utf-8").splitlines() == urls


@pytest.mark.parametrize("slug", [
    "nhan-vien-cong-nghe-thong-tin", "ky-su-cntt", "it-support", "software-developer",
    "ky-su-lap-trinh-phan-mem", "data-scientist", "data-analyst", "data-engineer",
    "chuyen-vien-phan-tich-du-lieu", "business-intelligence", "manual-tester",
    "ml-engineer", "ai-engineer", "ky-su-an-ninh-mang",
])
def test_slug_filter_keeps_it_and_data_jobs(slug):
    url = f"https://www.topcv.vn/viec-lam/{slug}/123.html"
    assert crawler.filter_job_urls([url]) == [url]


@pytest.mark.parametrize("slug", [
    "giao-vien-tieng-anh", "ke-toan", "nhan-vien-logistics", "sale-admin",
    "internal-auditor", "nhan-vien-data-entry", "nhan-vien-nhap-lieu",
    "nhan-vien-kinh-doanh-thiet-bi-y-te", "nhan-vien-hanh-chinh",
])
def test_slug_filter_rejects_unrelated_jobs(slug):
    url = f"https://www.topcv.vn/viec-lam/{slug}/123.html"
    assert crawler.filter_job_urls([url]) == []


def test_cached_link_filter_deduplicates_without_http(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("cached link filtering must not send HTTP requests")

    monkeypatch.setattr(crawler, "fetch", unexpected)
    url = "https://www.topcv.vn/viec-lam/data-engineer/123.html"
    assert crawler.filter_job_urls([url, url + "?source=test", "https://www.topcv.vn/viec-lam/ke-toan/456.html"]) == [url]


def test_sitemap_error_preserves_saved_links(tmp_path, monkeypatch):
    output = tmp_path / "links.txt"
    output.write_text("previous snapshot\n", encoding="utf-8")
    monkeypatch.setattr(crawler, "fetch", lambda url, *, delay: "<html>blocked</html>")
    with pytest.raises(ValueError, match="Không nhận diện"):
        crawler.get_job_urls_from_sitemap(delay=0, output_file=output)
    assert output.read_text(encoding="utf-8") == "previous snapshot\n"


def test_crawl_from_saved_links_resumes_without_scanning_sitemap(tmp_path, monkeypatch):
    raw = tmp_path / "jobs.jsonl"
    raw.write_text(json.dumps({"job_id": 123, "job_description": "Existing"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(crawler, "RAW_DIR", tmp_path)
    monkeypatch.setattr(crawler, "RAW_JSONL", raw)
    fetched = []
    job0 = "https://www.topcv.vn/viec-lam/data-engineer/123.html"
    job1 = "https://www.topcv.vn/viec-lam/ml-engineer/456.html"

    def fake_fetch(url, *, delay):
        fetched.append(url)
        return "detail"

    monkeypatch.setattr(crawler, "fetch", fake_fetch)
    monkeypatch.setattr(crawler, "parse_detail", lambda text, listing: {**listing, "job_description": "New"})
    assert crawler.crawl_from_links([job0, job1, job1 + "?source=test"], workers=1, delay=0)
    assert fetched == [job1]
    assert crawler.load_done_ids() == {123, 456}


def test_load_job_urls_rejects_empty_file(tmp_path, monkeypatch):
    monkeypatch.setattr(crawler, "RAW_DIR", tmp_path)
    crawler.job_urls_file().write_text("\ufeff\n  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="File link rỗng"):
        crawler.load_job_urls()


@pytest.mark.parametrize("failed_id", [None, 2])
def test_detail_workers_drain_queue_and_leave_failures_retryable(tmp_path, monkeypatch, failed_id):
    monkeypatch.setattr(crawler, "RAW_DIR", tmp_path)
    monkeypatch.setattr(crawler, "RAW_JSONL", tmp_path / "jobs.jsonl")
    urls = [f"https://www.topcv.vn/viec-lam/data-engineer/{job_id}.html" for job_id in range(1, 8)]
    fetched = []

    def fake_fetch(url, *, delay):
        job_id = crawler.job_id_from_url(url)
        fetched.append(job_id)
        if job_id == failed_id:
            raise RuntimeError("temporary failure")
        return "detail"

    monkeypatch.setattr(crawler, "fetch", fake_fetch)
    monkeypatch.setattr(crawler, "parse_detail", lambda text, listing: {**listing, "job_description": "Valid"})
    assert crawler.crawl_from_links(urls, workers=3, delay=0) is (failed_id is None)
    assert sorted(fetched) == list(range(1, 8))
    assert crawler.load_done_ids() == set(range(1, 8)) - {failed_id}
    records = crawler.RAW_JSONL.read_text(encoding="utf-8").splitlines()
    assert len(records) == 7 - (failed_id is not None)
