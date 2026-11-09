"""Supervised Job Role Classification models with 80/10/10 stratified split and TF-IDF baseline."""
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from src.config import (
    RANDOM_STATE,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    CLASSIFICATION_OUTPUTS
)
from src.evaluation.classification_metrics import (
    compute_classification_metrics,
    save_classification_reports
)

def run_classification_experiments(
    X_embeddings: np.ndarray,
    y_labels: pd.Series,
    raw_descriptions: List[str],
    unique_job_ids: List[str],
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, Any, Dict[str, Any], pd.DataFrame]:
    """
    Run classification experiments with:
    1. Stratified 80/10/10 split (Train: 80%, Validation: 10%, Test: 10%).
    2. Classical Baseline: TF-IDF + Logistic Regression (fit strictly on train).
    3. Multilingual SBERT models:
       - Logistic Regression
       - Linear SVM
       - K-Nearest Neighbors (k=5, cosine metric)
       - Random Forest
       - XGBoost Classifier
    4. Model selection on Validation set; final evaluation on Test set.
    """
    # Filter classes with >= 10 samples for robust stratification
    class_counts = y_labels.value_counts()
    valid_classes = class_counts[class_counts >= 10].index.tolist()
    valid_mask = y_labels.isin(valid_classes).values
    
    X = X_embeddings[valid_mask]
    y = y_labels[valid_mask].values
    raw_desc = np.array(raw_descriptions)[valid_mask]
    job_ids = np.array(unique_job_ids)[valid_mask]
    
    # Encode string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)
    
    # Stratified 80/10/10 split:
    # Step 1: 80% train, 20% temp
    indices = np.arange(len(y_encoded))
    train_idx, temp_idx = train_test_split(
        indices, test_size=(VAL_SIZE + TEST_SIZE), random_state=random_state, stratify=y_encoded
    )
    # Step 2: 50% val, 50% test from temp (10% and 10% of total)
    temp_y = y_encoded[temp_idx]
    val_rel_idx, test_rel_idx = train_test_split(
        np.arange(len(temp_idx)), test_size=0.5, random_state=random_state, stratify=temp_y
    )
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]
    
    # Create split manifest for classification
    split_records = []
    for idx in train_idx:
        split_records.append({"unique_job_id": job_ids[idx], "task": "classification", "split": "train"})
    for idx in val_idx:
        split_records.append({"unique_job_id": job_ids[idx], "task": "classification", "split": "val"})
    for idx in test_idx:
        split_records.append({"unique_job_id": job_ids[idx], "task": "classification", "split": "test"})
    cls_split_manifest = pd.DataFrame(split_records)
    
    # Embeddings splits
    X_train, y_train = X[train_idx], y_encoded[train_idx]
    X_val, y_val = X[val_idx], y_encoded[val_idx]
    X_test, y_test = X[test_idx], y_encoded[test_idx]
    
    # Text splits for TF-IDF baseline
    desc_train = raw_desc[train_idx]
    desc_val = raw_desc[val_idx]
    desc_test = raw_desc[test_idx]
    
    # -------------------------------------------------------------
    # 1. Classical Baseline: TF-IDF + Logistic Regression
    # -------------------------------------------------------------
    print("[Classification] Fitting TF-IDF Vectorizer strictly on train split...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words=None)
    X_train_tfidf = tfidf.fit_transform(desc_train)
    X_val_tfidf = tfidf.transform(desc_val)
    X_test_tfidf = tfidf.transform(desc_test)
    
    print("[Classification] Training Baseline: TF-IDF + Logistic Regression...")
    tfidf_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state)
    tfidf_lr.fit(X_train_tfidf, y_train)
    val_preds_tfidf = tfidf_lr.predict(X_val_tfidf)
    test_preds_tfidf = tfidf_lr.predict(X_test_tfidf)
    
    val_metrics_tfidf = compute_classification_metrics(y_val, val_preds_tfidf, "TF-IDF + Logistic Regression (Baseline)", split_name="Validation")
    test_metrics_tfidf = compute_classification_metrics(y_test, test_preds_tfidf, "TF-IDF + Logistic Regression (Baseline)", split_name="Test")
    test_metrics_tfidf["Val Macro-F1"] = val_metrics_tfidf["Macro-F1"]
    test_metrics_tfidf["Val Accuracy"] = val_metrics_tfidf["Accuracy"]
    
    # -------------------------------------------------------------
    # 2. Multilingual SBERT Classifiers
    # -------------------------------------------------------------
    sbert_models = {
        "SBERT + Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=random_state
        ),
        "SBERT + Linear SVM": LinearSVC(
            C=1.0, class_weight='balanced', random_state=random_state, dual=False, max_iter=2000
        ),
        "SBERT + K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, metric='cosine', weights='distance'
        ),
        "SBERT + Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=25, class_weight='balanced', random_state=random_state, n_jobs=-1
        ),
        "SBERT + XGBoost Classifier": XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1, random_state=random_state, n_jobs=-1,
            eval_metric='mlogloss'
        )
    }
    
    results_list = [test_metrics_tfidf]
    trained_models = {"TF-IDF + Logistic Regression (Baseline)": tfidf_lr}
    all_val_preds = {"TF-IDF + Logistic Regression (Baseline)": val_preds_tfidf}
    all_test_preds = {"TF-IDF + Logistic Regression (Baseline)": test_preds_tfidf}
    
    best_val_f1 = val_metrics_tfidf["Macro-F1"]
    best_model_name = "TF-IDF + Logistic Regression (Baseline)"
    best_model = tfidf_lr
    best_test_preds = test_preds_tfidf
    
    for name, clf in sbert_models.items():
        print(f"[Classification] Training {name} on SBERT train split...")
        clf.fit(X_train, y_train)
        
        # Validation evaluation (for model selection)
        val_preds = clf.predict(X_val)
        val_metrics = compute_classification_metrics(y_val, val_preds, name, split_name="Validation")
        
        # Test evaluation (unbiased held-out test)
        test_preds = clf.predict(X_test)
        test_metrics = compute_classification_metrics(y_test, test_preds, name, split_name="Test")
        test_metrics["Val Macro-F1"] = val_metrics["Macro-F1"]
        test_metrics["Val Accuracy"] = val_metrics["Accuracy"]
        
        results_list.append(test_metrics)
        trained_models[name] = clf
        all_val_preds[name] = val_preds
        all_test_preds[name] = test_preds
        
        if val_metrics["Macro-F1"] > best_val_f1:
            best_val_f1 = val_metrics["Macro-F1"]
            best_model_name = name
            best_model = clf
            best_test_preds = test_preds
            
    print(f"[Classification] Selected Best Model by Val Macro-F1: {best_model_name} (Val Macro-F1: {best_val_f1:.4f})")
    
    # Save reports, confusion matrix, and comparison charts on Test set
    y_test_names = le.inverse_transform(y_test)
    best_preds_names = le.inverse_transform(best_test_preds)
    
    results_df = save_classification_reports(
        results_list, y_test_names, best_preds_names, class_names
    )
    
    # Save best model artifact
    model_artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "label_encoder": le,
        "class_names": class_names,
        "tfidf_vectorizer": tfidf if "TF-IDF" in best_model_name else None,
        "best_val_macro_f1": best_val_f1
    }
    model_save_path = CLASSIFICATION_OUTPUTS / 'models' / 'best_classification_model.joblib'
    joblib.dump(model_artifact, model_save_path)
    print(f"[Classification] Saved best model to {model_save_path}")
    
    metadata = {
        "best_model_name": best_model_name,
        "best_val_macro_f1": best_val_f1,
        "class_names": class_names,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx)
    }
    
    return results_df, best_model, metadata, cls_split_manifest
