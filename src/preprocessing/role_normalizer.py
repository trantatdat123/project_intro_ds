"""Nhãn nghề từ title: quy tắc tạo nhãn sơ bộ, không phải nhãn chuyên gia."""

import re

from src.utils import fold_text


CORE_ROLES = ["Data Analyst", "Data Scientist", "Data Engineer", "AI/ML Engineer", "Software Engineer"]
PATTERNS = {
    "Data Analyst": r"\b(data analyst|data analytics|business intelligence|bi analyst|bi developer|power bi|powerbi|reporting analyst|phan tich du lieu|phan tich so lieu)\b",
    "Data Scientist": r"\b(data scientist|data science|khoa hoc du lieu)\b",
    "Data Engineer": r"\b(data engineer|data engineering|data architect|analytics engineer|etl developer|ky su du lieu|kien truc du lieu)\b",
    "AI/ML Engineer": r"\b(ai engineer|ai developer|ai researcher|ml engineer|machine learning|deep learning|computer vision|nlp engineer|llm engineer|mlops|llmops|ky su ai|ky su tri tue nhan tao|chuyen vien ai|tri tue nhan tao|hoc may)\b",
    "Software Engineer": r"\b(software engineer|software developer|developer|programmer|frontend|front end|backend|back end|fullstack|full stack|lap trinh|ky su phan mem|embedded|firmware)\b",
}


def role_candidates(title: str) -> list[str]:
    text = fold_text(title)
    if re.search(r"\b(cnc|plc|khuon|co khi)\b", text):
        return []
    roles = [role for role, pattern in PATTERNS.items() if re.search(pattern, text)]
    # 'AI Developer'/'Data Engineer + Python Developer' có thể là nhiều vị trí.
    # Từ developer ngay sau AI/ML chỉ là cách viết khác của cùng nghề AI.
    if "AI/ML Engineer" in roles and "Software Engineer" in roles:
        without_ai = re.sub(r"\b(ai developer|machine learning developer|ml developer)\b", "", text)
        if not re.search(PATTERNS["Software Engineer"], without_ai):
            roles.remove("Software Engineer")
    return roles


def map_standard_role(job_title: str) -> str:
    roles = role_candidates(job_title)
    return roles[0] if len(roles) == 1 else "Other"
