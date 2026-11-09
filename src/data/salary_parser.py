"""Salary parsing and validation module for JobLens Vietnam."""
import re
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np

from src.config import (
    USD_TO_VND_RATE,
    MIN_PLAUSIBLE_MONTHLY_SALARY,
    MAX_PLAUSIBLE_MONTHLY_SALARY,
    PROCESSED_DATA_DIR,
    REPORTS_DIR
)

UNSPECIFIED_KEYWORDS = [
    'thỏa thuận', 'thoả thuận', 'cập nhật', 'thương lượng', 
    'cạnh tranh', 'negotiable', 'deal', 'hấp dẫn', 'trao đổi', 'up to'
]

def parse_salary_string(salary_str: Any) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    Parse raw salary string into (salary_min_vnd, salary_max_vnd, salary_mean_vnd, status).
    Returns monthly salary values in VND.
    """
    if salary_str is None or pd.isna(salary_str):
        return None, None, None, "missing"
        
    s = str(salary_str).strip().lower()
    
    # Check for unspecified keywords
    if any(k in s for k in ['thỏa thuận', 'thoả thuận', 'cập nhật', 'thương lượng', 'cạnh tranh', 'negotiable']):
        return None, None, None, "unspecified"
        
    # Check currency
    is_usd = 'usd' in s or '$' in s
    is_annual = 'năm' in s or 'annual' in s or '/year' in s
    
    # Remove thousand separators (. or , between digits)
    cleaned = re.sub(r'(?<=\d)[.,](?=\d{3})', '', s)
    # Replace remaining commas with dots for decimal
    cleaned = cleaned.replace(',', '.')
    
    # Unit multiplier
    unit = 1.0
    if is_usd:
        unit = float(USD_TO_VND_RATE)
    elif any(k in cleaned for k in ['triệu', 'tr', ' triệu vnd', 'm']):
        unit = 1_000_000.0
        
    # Find all numeric values
    num_matches = re.findall(r'\d+(?:\.\d+)?', cleaned)
    if not num_matches:
        return None, None, None, "no_numbers_found"
        
    numbers = [float(x) for x in num_matches]
    
    # Scale numbers if needed (e.g. if unit was not explicitly stated, but numbers are small like 15 - 25)
    scaled_numbers = []
    for n in numbers:
        if n == 0:
            continue
        if n < 1000 and not is_usd:
            # Assumed to be in millions VND (e.g. 15, 25)
            scaled_numbers.append(n * 1_000_000.0)
        elif is_usd and n < 50000:
            # USD monthly salary
            scaled_numbers.append(n * unit)
        else:
            # Already in VND (e.g. 15000000)
            scaled_numbers.append(n)
            
    if is_annual and scaled_numbers:
        scaled_numbers = [x / 12.0 for x in scaled_numbers]
        
    if len(scaled_numbers) == 0:
        return None, None, None, "invalid_numbers"
    elif len(scaled_numbers) == 1:
        val = scaled_numbers[0]
        if any(k in s for k in ['tới', 'đến', 'up to', 'max', 'dưới']):
            sal_min, sal_max = None, val
            sal_mean = val
        elif any(k in s for k in ['từ', 'from', 'min', 'trở lên', 'trên']):
            sal_min, sal_max = val, None
            sal_mean = val
        else:
            sal_min, sal_max = val, val
            sal_mean = val
    else:
        sal_min = min(scaled_numbers[0], scaled_numbers[1])
        sal_max = max(scaled_numbers[0], scaled_numbers[1])
        sal_mean = (sal_min + sal_max) / 2.0
        
    # Plausibility filtering
    if sal_mean < MIN_PLAUSIBLE_MONTHLY_SALARY:
        return sal_min, sal_max, sal_mean, "implausible_low"
    if sal_mean > MAX_PLAUSIBLE_MONTHLY_SALARY:
        return sal_min, sal_max, sal_mean, "implausible_high"
        
    return sal_min, sal_max, sal_mean, "valid"

def process_salaries(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse salary column for dataframe, add min/max/mean columns,
    and generate audit diagnostics.
    """
    results = df['salary'].apply(parse_salary_string)
    
    parsed_df = df.copy()
    parsed_df['salary_min_vnd'] = [r[0] for r in results]
    parsed_df['salary_max_vnd'] = [r[1] for r in results]
    parsed_df['salary_mean_vnd'] = [r[2] for r in results]
    parsed_df['salary_parse_status'] = [r[3] for r in results]
    
    # Audit report
    status_counts = parsed_df['salary_parse_status'].value_counts().reset_index()
    status_counts.columns = ['parse_status', 'count']
    status_counts['percentage'] = (status_counts['count'] / len(parsed_df) * 100).round(2)
    
    report_path = REPORTS_DIR / 'tables' / 'salary_parsing_report.csv'
    status_counts.to_csv(report_path, index=False)
    
    return parsed_df, status_counts

