"""Salary parsing and validation module for JobLens Vietnam."""
import re
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
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
    'cạnh tranh', 'negotiable', 'deal', 'hấp dẫn', 'trao đổi'
]

def parse_salary_string(salary_str: Any) -> Tuple[Optional[float], Optional[float], Optional[float], str, str]:
    """
    Parse raw salary string into:
    (salary_min_vnd, salary_max_vnd, salary_mean_vnd, salary_category, salary_parse_status)
    
    salary_category in:
      - 'range': explicit min and max
      - 'exact': single fixed salary point
      - 'upper_bound_only': 'up to', 'lên đến', 'tới'
      - 'lower_bound_only': 'từ', 'trên', 'from'
      - 'unspecified': 'thỏa thuận', 'negotiable', etc.
      - 'implausible_low': < MIN_PLAUSIBLE_MONTHLY_SALARY
      - 'implausible_high': > MAX_PLAUSIBLE_MONTHLY_SALARY
      - 'invalid': unparseable or contradictory
      - 'missing': null / empty
      
    salary_parse_status in:
      - 'valid': suitable for supervised regression training
      - 'unspecified' / 'invalid' / 'implausible'
    """
    if salary_str is None or pd.isna(salary_str):
        return None, None, None, "missing", "missing"
        
    s = str(salary_str).strip().lower()
    if not s:
        return None, None, None, "missing", "missing"
    
    # Check for unspecified keywords
    if any(k in s for k in UNSPECIFIED_KEYWORDS):
        return None, None, None, "unspecified", "unspecified"
        
    # Check currency
    is_usd = 'usd' in s or '$' in s
    is_annual = 'năm' in s or 'annual' in s or '/year' in s
    
    # Remove thousand separators (. or , between 3 digits)
    cleaned = re.sub(r'(?<=\d)[.,](?=\d{3})', '', s)
    # Replace remaining commas with dots for decimal points
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
        return None, None, None, "invalid", "invalid"
        
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
        return None, None, None, "invalid", "invalid"
        
    sal_min = None
    sal_max = None
    sal_mean = None
    category = "invalid"
    
    if len(scaled_numbers) == 1:
        val = scaled_numbers[0]
        if any(k in s for k in ['tới', 'đến', 'up to', 'max', 'dưới', 'lên tới', 'lên đến']):
            category = "upper_bound_only"
            sal_min = None
            sal_max = val
            sal_mean = val
        elif any(k in s for k in ['từ', 'from', 'min', 'trở lên', 'trên']):
            category = "lower_bound_only"
            sal_min = val
            sal_max = None
            sal_mean = val
        else:
            category = "exact"
            sal_min = val
            sal_max = val
            sal_mean = val
    else:
        # Two or more numbers
        n1 = scaled_numbers[0]
        n2 = scaled_numbers[1]
        sal_min = min(n1, n2)
        sal_max = max(n1, n2)
        sal_mean = (sal_min + sal_max) / 2.0
        category = "range"
        
    # Independent validation of min, max, and mean
    if sal_min is not None and sal_max is not None and sal_min > sal_max:
        return None, None, None, "invalid", "invalid"
        
    if sal_mean < MIN_PLAUSIBLE_MONTHLY_SALARY:
        return sal_min, sal_max, sal_mean, "implausible_low", "implausible"
        
    if sal_mean > MAX_PLAUSIBLE_MONTHLY_SALARY:
        return sal_min, sal_max, sal_mean, "implausible_high", "implausible"
        
    # Check bounds independently
    if sal_min is not None and sal_min < (MIN_PLAUSIBLE_MONTHLY_SALARY * 0.5):
        return sal_min, sal_max, sal_mean, "implausible_low", "implausible"
        
    if sal_max is not None and sal_max > (MAX_PLAUSIBLE_MONTHLY_SALARY * 1.5):
        return sal_min, sal_max, sal_mean, "implausible_high", "implausible"
        
    # Valid candidate for regression
    return sal_min, sal_max, sal_mean, category, "valid"

def process_salaries(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse salary column for dataframe, add min/max/mean/category/status columns,
    and generate audit diagnostics.
    Returns (parsed_df, category_report_df, audit_summary_df).
    """
    results = df['salary'].apply(parse_salary_string)
    
    parsed_df = df.copy()
    parsed_df['salary_min_vnd'] = [r[0] for r in results]
    parsed_df['salary_max_vnd'] = [r[1] for r in results]
    parsed_df['salary_mean_vnd'] = [r[2] for r in results]
    parsed_df['salary_category'] = [r[3] for r in results]
    parsed_df['salary_parse_status'] = [r[4] for r in results]
    
    # 1. Salary Category breakdown report
    cat_counts = parsed_df['salary_category'].value_counts().reset_index()
    cat_counts.columns = ['salary_category', 'count']
    cat_counts['percentage'] = (cat_counts['count'] / len(parsed_df) * 100).round(2)
    
    report_path = REPORTS_DIR / 'tables' / 'salary_parsing_report.csv'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cat_counts.to_csv(report_path, index=False)
    
    # 2. Detailed Audit Table
    valid_mask = parsed_df['salary_parse_status'] == 'valid'
    valid_salaries = parsed_df.loc[valid_mask, 'salary_mean_vnd'] / 1_000_000.0
    
    audit_rows = [
        {"Metric": "Total Job Postings Analyzed", "Value": str(len(parsed_df))},
        {"Metric": "Unspecified / Negotiable (Thỏa thuận)", "Value": f"{(parsed_df['salary_category'] == 'unspecified').sum()} ({((parsed_df['salary_category'] == 'unspecified').sum()/len(parsed_df)*100):.1f}%)"},
        {"Metric": "Disclosed Salary Range (Min - Max)", "Value": f"{(parsed_df['salary_category'] == 'range').sum()} ({((parsed_df['salary_category'] == 'range').sum()/len(parsed_df)*100):.1f}%)"},
        {"Metric": "Exact Fixed Salary Disclosed", "Value": f"{(parsed_df['salary_category'] == 'exact').sum()} ({((parsed_df['salary_category'] == 'exact').sum()/len(parsed_df)*100):.1f}%)"},
        {"Metric": "Upper Bound Only Disclosed ('Up to')", "Value": f"{(parsed_df['salary_category'] == 'upper_bound_only').sum()} ({((parsed_df['salary_category'] == 'upper_bound_only').sum()/len(parsed_df)*100):.1f}%)"},
        {"Metric": "Lower Bound Only Disclosed ('From')", "Value": f"{(parsed_df['salary_category'] == 'lower_bound_only').sum()} ({((parsed_df['salary_category'] == 'lower_bound_only').sum()/len(parsed_df)*100):.1f}%)"},
        {"Metric": "Implausible Outliers Excluded (<3M or >150M VND)", "Value": f"{(parsed_df['salary_parse_status'] == 'implausible').sum()}"},
        {"Metric": "Total Valid Samples for Regression", "Value": f"{valid_mask.sum()} ({valid_mask.mean()*100:.1f}%)"},
        {"Metric": "Plausible Monthly Salary Mean (Million VND)", "Value": f"{valid_salaries.mean():.2f}" if not valid_salaries.empty else "N/A"},
        {"Metric": "Plausible Monthly Salary Median (Million VND)", "Value": f"{valid_salaries.median():.2f}" if not valid_salaries.empty else "N/A"},
        {"Metric": "Plausible Monthly Salary Std (Million VND)", "Value": f"{valid_salaries.std():.2f}" if not valid_salaries.empty else "N/A"},
        {"Metric": "Plausible Monthly Salary Min (Million VND)", "Value": f"{valid_salaries.min():.2f}" if not valid_salaries.empty else "N/A"},
        {"Metric": "Plausible Monthly Salary Max (Million VND)", "Value": f"{valid_salaries.max():.2f}" if not valid_salaries.empty else "N/A"}
    ]
    audit_summary_df = pd.DataFrame(audit_rows)
    audit_summary_path = REPORTS_DIR / 'tables' / 'salary_parsing_audit.csv'
    audit_summary_df.to_csv(audit_summary_path, index=False)
    print(f"[SalaryParser] Parsed {len(parsed_df)} records: {valid_mask.sum()} valid for regression. Audit saved to {audit_summary_path}")
    
    return parsed_df, cat_counts, audit_summary_df
