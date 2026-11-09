"""Text and structured feature cleaning module."""
import re
import html
import unicodedata
from typing import Any, Tuple, Dict
from pathlib import Path
import pandas as pd
import numpy as np
from src.config import PROCESSED_DATA_DIR, REPORTS_DIR

HTML_TAG_REGEX = re.compile(r'<[^>]+>')
WHITESPACE_REGEX = re.compile(r'\s+')

# Explicit city patterns
HANOI_PATTERNS = [
    r'\bhà nội\b', r'\bha noi\b', r'\bhanoi\b', r'\bthành phố hà nội\b',
    r'\btp\.?\s*hà nội\b', r'\btp\.?\s*ha noi\b', r'\bhn\b'
]
HCM_PATTERNS = [
    r'\bhồ chí minh\b', r'\bho chi minh\b', r'\bhcm\b', r'\btp\.?\s*hcm\b',
    r'\btp\.?\s*hồ chí minh\b', r'\btp\.?\s*ho chi minh\b', r'\bsài gòn\b',
    r'\bsaigon\b', r'\bthành phố hồ chí minh\b'
]
DANANG_PATTERNS = [
    r'\bđà nẵng\b', r'\bda nang\b', r'\bdanang\b', r'\btp\.?\s*đà nẵng\b',
    r'\btp\.?\s*da nang\b', r'\bthành phố đà nẵng\b'
]

# Specific district names when no city is explicitly specified
HANOI_DISTRICTS = [
    'cầu giấy', 'cau giay', 'nam từ liêm', 'nam tu liem', 'bắc từ liêm', 'bac tu liem',
    'thanh xuân', 'thanh xuan', 'hoàn kiếm', 'hoan kiem', 'ba đình', 'ba dinh',
    'đống đa', 'dong da', 'hai bà trưng', 'hai ba trung', 'hoàng mai', 'hoang mai',
    'long biên', 'long bien', 'tây hồ', 'tay ho', 'hà đông', 'ha dong',
    'gia lâm', 'gia lam', 'đông anh', 'dong anh', 'sóc sơn', 'thanh trì', 'hoài đức',
    'thạch thất', 'quốc oai', 'mê linh', 'đan phượng'
]

HCM_DISTRICTS = [
    'quận 1', 'quận 2', 'quận 3', 'quận 4', 'quận 5', 'quận 6', 'quận 7', 'quận 8',
    'quận 9', 'quận 10', 'quận 11', 'quận 12', 'district 1', 'district 2', 'district 3', 'district 7',
    'thủ đức', 'thu duc', 'bình thạnh', 'binh thanh', 'tân bình', 'tan binh',
    'tân phú', 'tan phu', 'phú nhuận', 'phu nhuan', 'gò vấp', 'go vap',
    'bình tân', 'binh tan', 'nhà bè', 'nha be', 'hóc môn', 'hoc mon', 'củ chi', 'cu chi', 'bình chánh', 'binh chanh'
]

DANANG_DISTRICTS = [
    'hải châu', 'hai chau', 'thanh khê', 'thanh khe', 'sơn trà', 'son tra',
    'ngũ hành sơn', 'ngu hanh son', 'liên chiểu', 'lien chieu', 'cẩm lệ', 'cam le', 'hòa vang'
]

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

def legacy_normalize_location(loc: Any) -> str:
    """Legacy buggy location normalization for audit and backward comparison."""
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

def normalize_location(loc: Any) -> str:
    """
    Robust location normalization:
    1. Detects explicit city names first (Hanoi, HCMC, Da Nang).
    2. Identifies multi-location postings ('Multiple').
    3. Uses district-level fallbacks only when city is unspecified.
    4. Never uses generic words ('quận', 'huyện', 'thành phố') as city indicators.
    5. Detects Remote / Work From Home.
    """
    if loc is None or pd.isna(loc):
        return "Khác"
    s = str(loc).lower().strip()
    if not s:
        return "Khác"
        
    # Check explicit cities
    is_hn = any(re.search(p, s) for p in HANOI_PATTERNS)
    is_hcm = any(re.search(p, s) for p in HCM_PATTERNS)
    is_dn = any(re.search(p, s) for p in DANANG_PATTERNS)
    
    # If multiple distinct major cities are mentioned
    if sum([is_hn, is_hcm, is_dn]) > 1:
        return "Multiple"
        
    if is_hn:
        return "Hà Nội"
    if is_hcm:
        return "Hồ Chí Minh"
    if is_dn:
        return "Đà Nẵng"
        
    # District-level fallbacks when no city is explicitly specified
    is_hn_dist = any(d in s for d in HANOI_DISTRICTS)
    is_hcm_dist = any(d in s for d in HCM_DISTRICTS)
    is_dn_dist = any(d in s for d in DANANG_DISTRICTS)
    
    if sum([is_hn_dist, is_hcm_dist, is_dn_dist]) > 1:
        return "Multiple"
    if is_hn_dist:
        return "Hà Nội"
    if is_hcm_dist:
        return "Hồ Chí Minh"
    if is_dn_dist:
        return "Đà Nẵng"
        
    # Check remote
    if re.search(r'\b(remote|từ xa|online|work from home|wfh)\b', s):
        return "Remote"
        
    return "Khác"

def generate_location_audit(df: pd.DataFrame, save_path: Path = None) -> pd.DataFrame:
    """Compare old vs new location mappings, generate audit table and statistics."""
    if save_path is None:
        save_path = REPORTS_DIR / "tables" / "location_fix_audit.csv"
        
    raw_locs = df['location'].dropna()
    audit_data = []
    
    loc_counts = raw_locs.value_counts()
    for raw_loc, count in loc_counts.items():
        old_val = legacy_normalize_location(raw_loc)
        new_val = normalize_location(raw_loc)
        if old_val != new_val:
            audit_data.append({
                "raw_location": raw_loc,
                "old_mapping": old_val,
                "new_mapping": new_val,
                "affected_records": count,
                "reason": "Hanoi/Da Nang/Multiple incorrectly matched by generic 'quận' keyword" if (old_val == "Hồ Chí Minh" and new_val != "Hồ Chí Minh") else "Refined boundary match"
            })
            
    audit_df = pd.DataFrame(audit_data)
    if not audit_df.empty:
        audit_df = audit_df.sort_values(by="affected_records", ascending=False).reset_index(drop=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"[LocationAudit] Found {len(audit_df)} unique location strings ({audit_df['affected_records'].sum()} records) affected by location bug. Saved to {save_path}")
    return audit_df

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
    digits = re.findall(r'\d+', s)
    if digits:
        v = float(digits[0])
        if v <= 20:
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
        return 2026
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
