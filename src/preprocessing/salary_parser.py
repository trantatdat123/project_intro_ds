"""Lương niêm yết: nhận diện tiền tệ, cận mở và quy đổi về VND/tháng."""

import re

import numpy as np
import pandas as pd

from src.utils import fold_text


def _number(token: str) -> float:
    parts = re.split(r"[.,]", token)
    if len(parts) > 1 and all(len(part) == 3 for part in parts[1:]):
        return float("".join(parts))
    if "." in token and "," in token:
        position = max(token.rfind("."), token.rfind(","))
        return float(re.sub(r"[.,]", "", token[:position]) + "." + token[position + 1:])
    return float(token.replace(",", "."))


def parse_salary_range(raw_salary, usd_to_vnd_rate: float = 25500.0) -> dict:
    if usd_to_vnd_rate <= 0:
        raise ValueError("usd_to_vnd_rate phải > 0")
    result = {"salary_min": None, "salary_max": None, "salary_mean": None,
              "currency": None, "is_negotiable": False, "salary_bound": "missing",
              "salary_period": "unknown", "salary_parse_status": "missing"}
    if not isinstance(raw_salary, str) or not raw_salary.strip():
        return result
    text = raw_salary.casefold().replace("\xa0", " ")
    folded = fold_text(re.sub(r"(?<=\d)(?=[a-zA-ZÀ-ỹ])", " ", text))
    if re.search(r"\b(thoa thuan|thuong luong|negotiable|competitive)\b", folded):
        result.update(is_negotiable=True, salary_parse_status="negotiable")
        return result
    currency = "USD" if re.search(r"\busd\b", folded) or "$" in text else "VND" if re.search(r"\b(vnd|vndong|dong|trieu|tr|mil|million|d)\b", folded) else None
    if currency is None:
        result["salary_parse_status"] = "unknown_currency"
        return result
    result["currency"] = currency
    if re.search(r"\b(hour|hourly|gio|day|daily|ngay)\b", folded):
        result.update(salary_period="hour/day", salary_parse_status="non_monthly")
        return result
    annual = bool(re.search(r"\b(year|yearly|annual|annually|nam)\b", folded))
    result["salary_period"] = "year" if annual else "month_assumed"
    numbers = re.findall(r"\d+(?:[.,]\d+)*", text)
    if not 1 <= len(numbers) <= 2:
        result["salary_parse_status"] = "ambiguous_numbers" if numbers else "no_amount"
        return result
    try:
        amounts = [_number(number) for number in numbers]
    except ValueError:
        result["salary_parse_status"] = "invalid_number"
        return result
    scale = 1.0
    if re.search(r"\b(trieu|tr|million|mil)\b", folded):
        scale = 1e6
    elif re.search(r"\b(ty|billion)\b", folded):
        scale = 1e9
    elif re.search(r"\d\s*k\b|\b(nghin|ngan|thousand)\b", folded):
        scale = 1e3
    # Đơn vị ở cuối áp dụng cho cả hai đầu: '10 - 20 triệu'.
    rate = usd_to_vnd_rate if currency == "USD" else 1.0
    amounts = [value * scale * rate / (12 if annual else 1) for value in amounts]
    if any(value <= 0 or not np.isfinite(value) for value in amounts) or (len(amounts) == 2 and amounts[0] > amounts[1]):
        result["salary_parse_status"] = "invalid_range"
        return result
    if len(amounts) == 2:
        result.update(salary_min=amounts[0], salary_max=amounts[1], salary_mean=sum(amounts) / 2, salary_bound="range")
    elif re.search(r"\b(up to|upto|toi|den|toi da|maximum|max)\b", folded):
        result.update(salary_max=amounts[0], salary_bound="upper")
    elif re.search(r"\b(tu|from|at least|minimum|min|tren|over)\b", folded) or "+" in text:
        result.update(salary_min=amounts[0], salary_bound="lower")
    else:
        result.update(salary_min=amounts[0], salary_max=amounts[0], salary_mean=amounts[0], salary_bound="exact")
    result["salary_parse_status"] = "parsed"
    return result


def remove_salary_outliers(df, salary_col="salary_mean", lower_quantile=0.01, upper_quantile=0.99):
    """Dùng cho tập train/EDA đã chọn; không dùng để học ngưỡng từ test."""
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Cần 0 <= lower_quantile < upper_quantile <= 1")
    values = pd.to_numeric(df[salary_col], errors="coerce")
    low, high = values.quantile([lower_quantile, upper_quantile])
    return df.loc[values.between(low, high)].copy()
