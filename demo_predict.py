"""Interactive demo script for JobLens Vietnam live inference.
Demonstrates live prediction for a new Job Description:
- Task 1: Job Role Classification
- Task 2: Salary Estimation (Million VND/month)
- Task 3: Cluster Assignment & Skill Recommendations
"""
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

import src.config as config
from src.data.cleaner import clean_text, normalize_location, normalize_job_position, normalize_job_type
from src.features.sbert_encoder import load_sbert_model

def load_artifacts():
    print("[Demo] Loading trained models and SBERT encoder...")
    # Load Models
    cls_artifact = joblib.load(config.CLASSIFICATION_OUTPUTS / 'models' / 'best_classification_model.joblib')
    reg_artifact = joblib.load(config.REGRESSION_OUTPUTS / 'models' / 'best_regression_model.joblib')
    cl_artifact = joblib.load(config.CLUSTERING_OUTPUTS / 'models' / 'best_clustering_model.joblib')
    
    # Load SBERT
    sbert_model = load_sbert_model()
    
    # Load regression training template to rebuild tabular features with exact column alignment
    df_reg = pd.read_csv(config.PROCESSED_DATA_DIR / "regression_data.csv", nrows=10)
    
    return cls_artifact, reg_artifact, cl_artifact, sbert_model

def predict_job(
    job_title: str,
    job_description: str,
    requirements: str,
    experience_years: float = 2.0,
    location: str = "Hà Nội",
    position: str = "Nhân viên",
    job_type: str = "Toàn thời gian",
    year: int = 2026,
    artifacts = None
):
    cls_artifact, reg_artifact, cl_artifact, sbert_model = artifacts
    
    # 1. Clean Inputs
    clean_desc = clean_text(job_description)
    clean_reqs = clean_text(requirements)
    norm_loc = normalize_location(location)
    norm_pos = normalize_job_position(position)
    norm_jtype = normalize_job_type(job_type)
    
    # 2. Encode with SBERT
    desc_emb = sbert_model.encode([clean_desc], normalize_embeddings=True).astype(np.float32)
    
    composite_text = f"{clean_text(job_title)}. {clean_desc} Yêu cầu: {clean_reqs}"
    comp_emb = sbert_model.encode([composite_text], normalize_embeddings=True).astype(np.float32)
    
    # 3. TASK 1: Job Role Classification
    cls_model = cls_artifact["model"]
    le = cls_artifact["label_encoder"]
    pred_class_idx = cls_model.predict(desc_emb)[0]
    predicted_role = le.inverse_transform([pred_class_idx])[0]
    
    # 4. TASK 2: Salary Prediction
    # Build tabular features matching the training feature names
    feature_names = reg_artifact["feature_names"]
    
    # Construct input row dictionary
    tab_features = {f: 0.0 for f in feature_names}
    
    # Fill SBERT features
    for i in range(desc_emb.shape[1]):
        feat_key = f"sbert_{i}"
        if feat_key in tab_features:
            tab_features[feat_key] = float(desc_emb[0, i])
            
    # Fill Categorical One-Hot features
    role_key = f"cat__job_role_label_{predicted_role}"
    loc_key = f"cat__norm_location_{norm_loc}"
    pos_key = f"cat__norm_position_{norm_pos}"
    type_key = f"cat__norm_job_type_{norm_jtype}"
    
    for k in [role_key, loc_key, pos_key, type_key]:
        if k in tab_features:
            tab_features[k] = 1.0
            
    # Numerical features (exp_years and year_clean were standardized during training)
    # Using sensible approximate z-scores or raw fallback
    if "num__exp_years" in tab_features:
        tab_features["num__exp_years"] = (experience_years - 2.5) / 1.5
    if "num__year_clean" in tab_features:
        tab_features["num__year_clean"] = (year - 2025.8) / 0.5
        
    X_reg = pd.DataFrame([tab_features])[feature_names].values
    predicted_salary_mil = float(reg_artifact["model"].predict(X_reg)[0])
    # Bound between sensible range
    predicted_salary_mil = max(8.0, min(80.0, predicted_salary_mil))
    
    # 5. TASK 3: Clustering & Skill Archetype
    km_model = cl_artifact["model"]
    cluster_idx = int(km_model.predict(comp_emb)[0])
    cluster_name = cl_artifact["cluster_names"].get(cluster_idx, f"Cluster {cluster_idx}")
    top_kws = cl_artifact["cluster_keywords"].get(cluster_idx, [])
    
    return {
        "predicted_role": predicted_role,
        "estimated_salary_mean": round(predicted_salary_mil, 1),
        "estimated_salary_range": f"{round(predicted_salary_mil * 0.85, 1)} - {round(predicted_salary_mil * 1.15, 1)} Triệu VNĐ",
        "cluster_archetype": cluster_name,
        "recommended_cluster_skills": ", ".join(top_kws[:8])
    }

def main():
    print("=" * 70)
    print(" JOBLENS VIETNAM - LIVE RECRUITMENT INFERENCE DEMO ")
    print("=" * 70)
    
    artifacts = load_artifacts()
    
    # Test Sample 1: AI / ML Engineer
    sample_jd = (
        "Chúng tôi đang tìm kiếm kỹ sư Machine Learning / AI Engineer tham gia nghiên cứu và "
        "triển khai các mô hình Deep Learning, Large Language Models (LLM), Computer Vision. "
        "Xây dựng pipeline huấn luyện mô hình, tối ưu hóa hiệu năng inference trên môi trường production."
    )
    sample_reqs = (
        "Tốt nghiệp Đại học chuyên ngành CNTT, Toán Tin hoặc Khoa học Máy tính. "
        "Thành thạo lập trình Python, sử dụng thành thạo PyTorch hoặc TensorFlow. "
        "Có kinh nghiệm làm việc với Docker, FastAPI, CI/CD, Git. Hiểu biết về Transformer, NLP."
    )
    
    print("\n[Input Sample Job Posting]")
    print(f"Title: Senior AI / Machine Learning Engineer")
    print(f"Location: Hà Nội | Experience: 3.0 năm | Position: Nhân viên")
    print(f"Description: {sample_jd[:120]}...")
    
    res = predict_job(
        job_title="Senior AI / Machine Learning Engineer",
        job_description=sample_jd,
        requirements=sample_reqs,
        experience_years=3.0,
        location="Hà Nội",
        position="Nhân viên",
        job_type="Toàn thời gian",
        year=2026,
        artifacts=artifacts
    )
    
    print("\n" + "-" * 50)
    print(" INFERENCE RESULTS ")
    print("-" * 50)
    print(f"1. Predicted Job Role:     {res['predicted_role']}")
    print(f"2. Estimated Monthly Salary: {res['estimated_salary_mean']} Triệu VNĐ/tháng ({res['estimated_salary_range']})")
    print(f"3. Natural Cluster Archetype: {res['cluster_archetype']}")
    print(f"4. Associated Skills:       {res['recommended_cluster_skills']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
