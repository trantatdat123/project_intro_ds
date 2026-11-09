"""Feature builder for tabular and multi-modal ablation experiments with strict zero leakage."""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

STRUCTURED_CATEGORICAL = ['job_role_label', 'norm_location', 'norm_position', 'norm_job_type']
STRUCTURED_NUMERICAL = ['exp_years']
STRUCTURED_YEAR = ['year_clean']

def build_ablation_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    sbert_train: np.ndarray,
    sbert_val: np.ndarray,
    sbert_test: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Construct feature matrices for all four required ablation configurations:
    - E1: Structured features without Year
    - E2: Structured features with Year
    - E3: SBERT + structured features without Year
    - E4: SBERT + structured features with Year
    
    Fits encoders strictly on `df_train` to prevent data leakage.
    Transforms `df_val` and `df_test`.
    """
    # 1. Pipeline for E1 (Structured without Year)
    preprocessor_e1 = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), STRUCTURED_CATEGORICAL),
            ('num', StandardScaler(), STRUCTURED_NUMERICAL)
        ]
    )
    X_train_e1 = preprocessor_e1.fit_transform(df_train)
    X_val_e1 = preprocessor_e1.transform(df_val)
    X_test_e1 = preprocessor_e1.transform(df_test)
    
    # 2. Pipeline for E2 (Structured with Year)
    preprocessor_e2 = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), STRUCTURED_CATEGORICAL),
            ('num', StandardScaler(), STRUCTURED_NUMERICAL + STRUCTURED_YEAR)
        ]
    )
    X_train_e2 = preprocessor_e2.fit_transform(df_train)
    X_val_e2 = preprocessor_e2.transform(df_val)
    X_test_e2 = preprocessor_e2.transform(df_test)
    
    # 3. Features for E3 (SBERT + Structured without Year)
    X_train_e3 = np.hstack([sbert_train, X_train_e1])
    X_val_e3 = np.hstack([sbert_val, X_val_e1])
    X_test_e3 = np.hstack([sbert_test, X_test_e1])
    
    # 4. Features for E4 (SBERT + Structured with Year)
    X_train_e4 = np.hstack([sbert_train, X_train_e2])
    X_val_e4 = np.hstack([sbert_val, X_val_e2])
    X_test_e4 = np.hstack([sbert_test, X_test_e2])
    
    feature_names_e1 = (
        list(preprocessor_e1.named_transformers_['cat'].get_feature_names_out(STRUCTURED_CATEGORICAL)) +
        STRUCTURED_NUMERICAL
    )
    feature_names_e2 = (
        list(preprocessor_e2.named_transformers_['cat'].get_feature_names_out(STRUCTURED_CATEGORICAL)) +
        STRUCTURED_NUMERICAL + STRUCTURED_YEAR
    )
    feature_names_e4 = [f"sbert_{i}" for i in range(sbert_train.shape[1])] + feature_names_e2

    return {
        "E1": {"train": X_train_e1, "val": X_val_e1, "test": X_test_e1, "names": feature_names_e1},
        "E2": {"train": X_train_e2, "val": X_val_e2, "test": X_test_e2, "names": feature_names_e2},
        "E3": {"train": X_train_e3, "val": X_val_e3, "test": X_test_e3, "names": [f"sbert_{i}" for i in range(sbert_train.shape[1])] + feature_names_e1},
        "E4": {"train": X_train_e4, "val": X_val_e4, "test": X_test_e4, "names": feature_names_e4}
    }
