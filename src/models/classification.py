"""Supervised Job Role Classification models."""
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from src.config import (
    RANDOM_STATE,
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
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
    """
    Run all 5 required classification algorithms:
    1. Logistic Regression
    2. Linear SVM
    3. K-Nearest Neighbors
    4. Random Forest Classifier
    5. XGBoost Classifier
    
    Uses stratified 80/20 train/test split.
    """
    # Filter classes with >= 10 samples for robust stratification
    class_counts = y_labels.value_counts()
    valid_classes = class_counts[class_counts >= 10].index.tolist()
    valid_mask = y_labels.isin(valid_classes).values
    
    X = X_embeddings[valid_mask]
    y = y_labels[valid_mask].values
    
    # Encode string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=random_state, stratify=y_encoded
    )
    
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=random_state
        ),
        "Linear SVM": LinearSVC(
            C=1.0, class_weight='balanced', random_state=random_state, dual=False, max_iter=2000
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, metric='cosine', weights='distance'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=25, class_weight='balanced', random_state=random_state, n_jobs=-1
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1, random_state=random_state, n_jobs=-1,
            eval_metric='mlogloss'
        )
    }
    
    results_list = []
    trained_models = {}
    best_f1 = -1.0
    best_model_name = ""
    best_model = None
    best_preds = None
    
    for name, clf in models.items():
        print(f"[Classification] Training {name}...")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        metrics = compute_classification_metrics(y_test, preds, name)
        results_list.append(metrics)
        trained_models[name] = clf
        
        if metrics["Macro-F1"] > best_f1:
            best_f1 = metrics["Macro-F1"]
            best_model_name = name
            best_model = clf
            best_preds = preds
            
    print(f"[Classification] Best Model: {best_model_name} with Macro-F1: {best_f1:.4f}")
    
    # Save reports, confusion matrix and comparison charts
    y_test_names = le.inverse_transform(y_test)
    best_preds_names = le.inverse_transform(best_preds)
    
    results_df = save_classification_reports(
        results_list, y_test_names, best_preds_names, class_names
    )
    
    # Save best model and label encoder
    model_artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "label_encoder": le,
        "class_names": class_names,
        "macro_f1": best_f1
    }
    model_save_path = CLASSIFICATION_OUTPUTS / 'models' / 'best_classification_model.joblib'
    joblib.dump(model_artifact, model_save_path)
    print(f"[Classification] Saved best model to {model_save_path}")
    
    metadata = {
        "best_model_name": best_model_name,
        "best_macro_f1": best_f1,
        "class_names": class_names,
        "test_size": len(y_test),
        "train_size": len(y_train)
    }
    
    return results_df, best_model, metadata

