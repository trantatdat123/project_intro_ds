"""Làm sạch HTML/văn bản và lấy cận dưới của kinh nghiệm yêu cầu."""

import html
import re
import unicodedata

from bs4 import BeautifulSoup

from src.utils import fold_text


def normalize_text(raw_text) -> str:
    if not isinstance(raw_text, str):
        return ""
    text = unicodedata.normalize("NFC", html.unescape(raw_text))
    text = re.sub(r"&#?0*39;?", "'", text)
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text).replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def clean_html(raw_html) -> str:
    text = normalize_text(raw_html)
    if not re.search(r"</?[a-zA-Z][^>]*>", text):
        return text
    soup = BeautifulSoup(text, "html.parser")
    for element in soup.select("script, style, noscript"):
        element.decompose()
    return normalize_text(soup.get_text("\n", strip=True))


def parse_experience(raw_exp) -> float | None:
    """Số năm tối thiểu; 'Không' đơn lẻ là thiếu dữ liệu, không suy thành 0."""
    text = fold_text(raw_exp)
    if not text or text in {"khong", "khong ro", "dang cap nhat", "n a", "null"}:
        return None
    if any(term in text for term in ("khong yeu cau", "khong can kinh nghiem", "no experience", "fresher", "chua co kinh nghiem")):
        return 0.0
    original = normalize_text(raw_exp).casefold()
    numbers = re.findall(r"\d+(?:[.,]\d+)?", original)
    if not numbers:
        return None
    values = [float(number.replace(",", ".")) for number in numbers]
    multiplier = 1 / 12 if re.search(r"\b(thang|month|months)\b", text) else 1
    if not re.search(r"\b(nam|year|years|thang|month|months)\b", text):
        return None
    if any(term in text for term in ("duoi", "less than", "under")):
        return 0.0
    return min(values) * multiplier


def normalize_location(raw_location) -> str:
    text = fold_text(raw_location)
    cities = {
        "Hà Nội": ("ha noi", "hanoi"),
        "Hồ Chí Minh": ("ho chi minh", "hcm", "saigon", "sai gon"),
        "Đà Nẵng": ("da nang", "danang"),
        "Bình Dương": ("binh duong",), "Đồng Nai": ("dong nai",),
        "Hải Phòng": ("hai phong",), "Cần Thơ": ("can tho",),
        "Bắc Ninh": ("bac ninh",), "Huế": ("hue",),
    }
    found = [city for city, aliases in cities.items()
             if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text) for alias in aliases)]
    if len(found) > 1:
        return "Nhiều địa điểm"
    if found:
        return found[0]
    if re.search(r"\b(remote|tu xa|work from home)\b", text):
        return "Remote"
    return "Khác/Chưa xác định"


def seniority_from_title(title) -> str:
    text = fold_text(title)
    for label, pattern in (
        ("Management", r"\b(manager|director|head|cto|truong phong|giam doc)\b"),
        ("Lead", r"\b(lead|principal|architect|truong nhom)\b"),
        ("Senior", r"\b(senior|sr|cao cap)\b"),
        ("Junior", r"\b(junior|jr|fresher|intern|trainee|thuc tap)\b"),
        ("Middle", r"\b(middle|mid)\b"),
    ):
        if re.search(pattern, text):
            return label
    return "Unspecified"
