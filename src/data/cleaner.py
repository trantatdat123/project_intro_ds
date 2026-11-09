"""Text and structured feature cleaning module."""
import re
import html
import unicodedata
from typing import Any
import pandas as pd
import numpy as np
from src.config import PROCESSED_DATA_DIR

HTML_TAG_REGEX = re.compile(r'<[^>]+>')
WHITESPACE_REGEX = re.compile(r'\s+')

def clean_text(text: Any) -> str:
    """Clean text by stripping HTML, normalizing unicode and whitespace, preserving tech tokens."""
    if text is None or pd.isna(text):
        return ""
    s = str(text)
    # Unescape HTML entities (&amp;, &nbsp;, etc.)
    s = html.unescape(s)
    # Remove HTML tags
    s = HTML_TAG_REGEX.sub(' ', s)
    # Normalize unicode to NFKC
    s = unicodedata.normalize('NFKC', s)
    # Replace non-printable characters
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', s)
    # Normalize excessive punctuation (leave tech tokens like C++, C#, .NET intact)
    s = re.sub(r'([!?,:;]){2,}', r'\1', s)
    # Normalize whitespace
    s = WHITESPACE_REGEX.sub(' ', s).strip()
    return s

def normalize_location(loc: Any) -> str:
    """Normalize location to primary regions: Ha Noi, Ho Chi Minh, Da Nang, Remote/Other."""
    if loc is None or pd.isna(loc):
        return "Khác"
    s = str(loc).lower()
    if any(k in s for k in ['hồ chí minh', 'ho chi minh', 'hcm', 'tp.hcm', 'quận', 'thủ đức', 'bình thạnh', 'tân bình', 'quận 1', 'quận 7']):
        return "Hồ Chí Minh"
    elif any(k in s for k in ['hà nội', 'ha noi', 'hn', 'cầu giấy', 'nam từ liêm', 'thanh xuân', 'hoàn kiếm', 'ba đình', 'đống đa']):
        return "Hà Nội"
    elif any(k in s for k in ['đà nẵng', 'da nang']):
        return "Đà Nẵng"
    elif any(k in s for k in ['remote', 'từ xa', 'online']):
        return "Remote"
    return "Khác"

def parse_experience_years(val: Any) -> float:
    """Extract approximate numerical years of experience required."""
    if val is None or pd.isna(val):
        return 0.0
    s = str(val).lower()
    if any(k in s for k in ['không', 'chưa', 'fresher', 'intern', 'thực tập', 'dưới 1 năm']):
        return 0.0
    m = re.search(r'(\d+)(?:\s*(?:-|đến|to)\s*(\d+))?\s*(?:năm|year)', s)
    if m:
        if m.group(2):
            return (float(m.group(1)) + float(m.group(2))) / 2.0
        return float(m.group(1))
    # Fallback digit search
    digits = re.findall(r'\d+', s)
    if digits:
        v = float(digits[0])
        if v <= 20: # reasonable years
            return v
    return 0.0

def normalize_experience_category(years: float) -> str:
    """Group numerical experience into standard levels."""
    if years <= 0.5:
        return "Fresher / Intern (<1 yr)"
    elif years <= 2.0:
        return "Junior (1-2 yrs)"
    elif years <= 4.0:
        return "Middle (3-4 yrs)"
    else:
        return "Senior / Lead (5+ yrs)"

def normalize_job_position(pos: Any) -> str:
    """Normalize job position hierarchy."""
    if pos is None or pd.isna(pos):
        return "Nhân viên"
    s = str(pos).lower()
    if any(k in s for k in ['trưởng', 'lead', 'manager', 'giám đốc', 'quản lý', 'director']):
        return "Quản lý / Leader"
    elif any(k in s for k in ['thực tập', 'intern', 'fresher']):
        return "Thực tập sinh / Fresher"
    elif any(k in s for k in ['chuyên gia', 'expert', 'architect']):
        return "Chuyên gia / Architect"
    return "Nhân viên"

def normalize_job_type(jtype: Any) -> str:
    """Normalize employment type."""
    if jtype is None or pd.isna(jtype):
        return "Toàn thời gian"
    s = str(jtype).lower()
    if any(k in s for k in ['bán thời gian', 'part-time', 'part time']):
        return "Bán thời gian"
    elif any(k in s for k in ['thực tập', 'internship']):
        return "Thực tập"
    return "Toàn thời gian"

def normalize_year(yr: Any) -> int:
    """Clean and validate the year field."""
    if yr is None or pd.isna(yr):
        return 2026 # Default current collection year
    try:
        y = int(float(yr))
        if 2020 <= y <= 2027:
            return y
        return 2026
    except (ValueError, TypeError):
        return 2026

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply comprehensive cleaning to all text and structured columns."""
    cleaned = df.copy()
    
    # Text fields
    for col in ['job_title', 'company_name', 'job_description', 'requirements', 'benefits']:
        cleaned[col] = cleaned[col].apply(clean_text)
        
    # Structured fields
    cleaned['norm_location'] = cleaned['location'].apply(normalize_location)
    cleaned['exp_years'] = cleaned['experience_level'].apply(parse_experience_years)
    cleaned['norm_experience'] = cleaned['exp_years'].apply(normalize_experience_category)
    cleaned['norm_position'] = cleaned['job_position'].apply(normalize_job_position)
    cleaned['norm_job_type'] = cleaned['job_type'].apply(normalize_job_type)
    cleaned['year_clean'] = cleaned['year'].apply(normalize_year)
    
    return cleaned
