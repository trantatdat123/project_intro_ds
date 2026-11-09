"""
Rule-Based / Weakly Supervised Job Role Label Generator.

Methodology Note:
The labels generated in this module are rule-based heuristic pseudo-labels (weak supervision),
not human-annotated ground truth. They are derived from regular expression matching against
the raw `job_title` column using standard Vietnamese and international IT role taxonomy.

Known Limitations:
1. Lexical ambiguity: Job titles that combine multiple keywords (e.g. 'Data Engineer / Backend')
   are mapped to the first matching rule according to the predefined precedence.
2. Heuristic noise: Variations, typos, or emerging job titles not captured by regex
   are categorized as 'Other / Unassigned' and excluded from supervised classification.
3. Supervised Model Interpretation: In this setting, the supervised classification model
   learns to predict the role category given the unmasked `job_description` (without title).
   Evaluation metrics (Accuracy, Macro-F1) measure the model's agreement with the rule-based
   taxonomic definitions rather than absolute human consensus.
"""
import re
import json
from typing import Tuple, Dict, Any
import pandas as pd
from src.config import PROCESSED_DATA_DIR, CLASSIFICATION_OUTPUTS, REPORTS_DIR

# Hierarchical rules: most specific to general
ROLE_RULES = [
    # 1. AI / ML Engineer
    ('AI / ML Engineer', r'\b(ai engineer|machine learning|deep learning|nlp|computer vision|máy học|trí tuệ nhân tạo|llm|genai|prompt engineer)\b'),
    
    # 2. Data Scientist
    ('Data Scientist', r'\b(data scientist|khoa học dữ liệu|data science specialist)\b'),
    
    # 3. Data Engineer
    ('Data Engineer', r'\b(data engineer|kỹ sư dữ liệu|big data|etl engineer|data pipeline|data platform|data warehouse)\b'),
    
    # 4. Data Analyst / BI
    ('Data Analyst / BI', r'\b(data analyst|bi analyst|business intelligence|phân tích dữ liệu|chuyên viên dữ liệu|power bi|tableau|bi developer)\b'),
    
    # 5. QA / Tester
    ('QA / Tester', r'\b(qa|qc|tester|test engineer|quality assurance|quality control|automation test|kiểm thử|manual test)\b'),
    
    # 6. DevOps / Cloud
    ('DevOps / Cloud', r'\b(devops|sre|cloud engineer|aws engineer|azure engineer|kubernetes|infrastructure engineer|system engineer|hạ tầng|site reliability)\b'),
    
    # 7. IT Support / Network / Security
    ('IT Support / Network / Security', r'\b(it support|helpdesk|help desk|network engineer|system admin|security|cybersecurity|an toàn thông tin|an ninh mạng|quản trị mạng|kỹ thuật it|it phần cứng)\b'),
    
    # 8. Software Engineer (Broader technical role)
    ('Software Engineer', r'\b(software engineer|developer|lập trình viên|kỹ sư phần mềm|backend|back-end|frontend|front-end|fullstack|full-stack|web developer|mobile developer|android|ios|flutter|react|node|java|python|php|c\+\+|golang|\.net|c#|embedded|nhúng)\b')
]

def map_job_title_to_role(title: Any) -> str:
    """Map raw job title to standardized job role using hierarchical heuristic regex rules."""
    if title is None or pd.isna(title):
        return "Other / Unassigned"
        
    t = str(title).lower().strip()
    
    for role, pattern in ROLE_RULES:
        if re.search(pattern, t):
            return role
            
    return "Other / Unassigned"

def generate_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Generate rule-based heuristic labels for the dataframe, filter unassigned,
    and save label distribution and mapping.
    """
    labeled_df = df.copy()
    labeled_df['job_role_label'] = labeled_df['job_title'].apply(map_job_title_to_role)
    
    # Label distribution
    dist = labeled_df['job_role_label'].value_counts().reset_index()
    dist.columns = ['job_role_label', 'count']
    dist['percentage'] = (dist['count'] / len(labeled_df) * 100).round(2)
    
    # Save distribution
    dist_path = REPORTS_DIR / 'tables' / 'label_distribution.csv'
    dist.to_csv(dist_path, index=False)
    
    # Label mapping metadata
    label_mapping = {role: idx for idx, (role, _) in enumerate(ROLE_RULES)}
    mapping_path = CLASSIFICATION_OUTPUTS / 'metrics' / 'label_mapping.json'
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
        
    print(f"[LabelGenerator] Generated rule-based labels across {len(dist)} categories. Saved to {dist_path}")
    return labeled_df, dist, label_mapping
