"""Chuẩn bị TinixAI, giữ raw và tạo các cột phục vụ EDA/mô hình."""

import hashlib

import pandas as pd

from src.preprocessing.cleaner import clean_html, normalize_location, parse_experience, seniority_from_title
from src.preprocessing.role_normalizer import role_candidates
from src.preprocessing.salary_parser import parse_salary_range
from src.preprocessing.skill_extractor import SkillExtractor
from src.utils import ROOT, fold_text, read_frame, save_frame, write_json


def prepare_jobs(input_path, output_path=ROOT / "data/processed/jobs_cleaned.parquet",
                 skill_dict_path=ROOT / "configs/skill_dictionary.json", usd_to_vnd_rate=25500, limit=None):
    frame = read_frame(input_path)
    required = {"job_title", "company_name", "salary", "experience_level", "job_description", "requirements", "year"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Thiếu cột TinixAI: {sorted(required - set(frame.columns))}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit phải > 0")
        frame = frame.head(limit).copy()
    input_count = len(frame)
    if "id" not in frame:
        frame["id"] = range(1, len(frame) + 1)
    frame = frame.drop_duplicates("id").reset_index(drop=True)
    for raw, clean in (("job_title", "title_clean"), ("job_description", "description_clean"),
                       ("requirements", "requirements_clean"), ("benefits", "benefits_clean")):
        frame[clean] = frame[raw].map(clean_html) if raw in frame else ""
    frame["text_clean"] = (frame.description_clean + "\n" + frame.requirements_clean).str.strip()
    frame["experience_min"] = frame.experience_level.map(parse_experience)
    frame["experience_missing"] = frame.experience_min.isna()
    frame["location_group"] = frame.location.map(normalize_location) if "location" in frame else "Khác/Chưa xác định"
    frame["seniority"] = frame.title_clean.map(seniority_from_title)
    frame["role_candidates"] = frame.title_clean.map(role_candidates)
    frame["role_standard"] = frame.role_candidates.map(lambda roles: roles[0] if len(roles) == 1 else "Other")
    frame["role_label_status"] = frame.role_candidates.map(lambda roles: "single" if len(roles) == 1 else "ambiguous" if roles else "unmapped")
    frame["role_label_source"] = "title_rule"
    salary = pd.DataFrame(frame.salary.map(lambda text: parse_salary_range(text, usd_to_vnd_rate)).tolist())
    frame = pd.concat([frame, salary], axis=1)
    extractor = SkillExtractor(skill_dict_path)
    # Kỹ năng lấy từ nội dung mô tả/yêu cầu, không dùng title để dự đoán nhãn nghề.
    frame["extracted_skills"] = frame.text_clean.map(extractor.extract_skills)
    frame["skill_count"] = frame.extracted_skills.map(len)
    frame["text_fingerprint"] = frame.text_clean.map(lambda text: hashlib.sha256(fold_text(text).encode()).hexdigest())
    frame["company_group"] = frame.company_name.map(fold_text)
    missing_company = frame.company_group.eq("")
    frame.loc[missing_company, "company_group"] = "unknown_" + frame.loc[missing_company, "text_fingerprint"]
    frame["duplicate_content"] = frame.duplicated(["company_group", "title_clean", "text_fingerprint", "year"], keep=False)
    frame["classification_eligible"] = frame.role_label_status.eq("single") & frame.text_clean.str.len().ge(30)
    frame["salary_eligible"] = frame.salary_mean.notna() & frame.salary_mean.gt(0)
    summary = {
        "input_rows": input_count, "cleaned_rows": len(frame), "duplicate_ids_removed": input_count - len(frame),
        "duplicate_content_rows": int(frame.duplicate_content.sum()),
        "role_counts": frame.role_standard.value_counts().to_dict(),
        "label_status_counts": frame.role_label_status.value_counts().to_dict(),
        "salary_status_counts": frame.salary_parse_status.value_counts().to_dict(),
        "salary_eligible_rows": int(frame.salary_eligible.sum()),
        "classification_eligible_rows": int(frame.classification_eligible.sum()),
        "experience_missing_rows": int(frame.experience_missing.sum()),
        "jobs_with_skills": int(frame.skill_count.gt(0).sum()),
        "usd_to_vnd_rate": usd_to_vnd_rate, "limit": limit,
    }
    save_frame(frame, output_path)
    write_json(ROOT / "reports/data_quality.json", summary)
    return frame, summary
