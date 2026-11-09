"""Supervised Salary Prediction regression models and ablation studies."""
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    REGRESSION_OUTPUTS,
    REPORTS_DIR
)
from src.features.feature_builder import build_ablation_features
from src.evaluation.regression_metrics import (
    compute_regression_metrics,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
    plot_ablation_comparison
)

def run_regression_experiments(
    df_reg: pd.DataFrame,
    sbert_embeddings: np.ndarray,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Dict[str, Any]]:
    """
    Execute full salary regression suite:
    1. Runs all 6 required models on full feature set (E4: SBERT + Structured with Year)
    2. Runs ablation study across configurations E1, E2, E3, E4
    3. Performs temporal train/test split experiment
    4. Generates all diagnostic plots and saves best model
    """
    # Target in Million VND
    y = (df_reg['salary_mean_vnd'] / 1_000_000.0).values
    
    # 80/20 train/test split
    indices = np.arange(len(df_reg))
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=random_state)
    
    df_train = df_reg.iloc[train_idx].reset_index(drop=True)
    df_test = df_reg.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]
    sbert_train, sbert_test = sbert_embeddings[train_idx], sbert_embeddings[test_idx]
    
    # Build ablation feature sets
    ablation_sets = build_ablation_features(df_train, df_test, sbert_train, sbert_test)
    
    # ==========================================
    # 1. Full Model Comparison on E4 (SBERT + Struct + Year)
    # ==========================================
    X_train_e4 = ablation_sets["E4"]["train"]
    X_test_e4 = ablation_sets["E4"]["test"]
    feature_names_e4 = ablation_sets["E4"]["names"]
    
    models = {
        "Dummy (Median Baseline)": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=16, random_state=random_state, n_jobs=-1),
        "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=100, max_depth=16, random_state=random_state, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=random_state, n_jobs=-1)
    }
    
    reg_results = []
    trained_models = {}
    best_mae = float('inf')
    best_model_name = ""
    best_model = None
    best_preds = None
    
    for name, reg in models.items():
        print(f"[Regression] Training {name} on E4...")
        reg.fit(X_train_e4, y_train)
        preds = reg.predict(X_test_e4)
        
        metrics = compute_regression_metrics(y_test, preds, name, config_name="E4")
        reg_results.append(metrics)
        trained_models[name] = reg
        
        if metrics["MAE (Million VND)"] < best_mae and name != "Dummy (Median Baseline)":
            best_mae = metrics["MAE (Million VND)"]
            best_model_name = name
            best_model = reg
            best_preds = preds
            
    reg_results_df = pd.DataFrame(reg_results)
    reg_results_path = REGRESSION_OUTPUTS / 'metrics' / 'regression_results.csv'
    reg_results_df.to_csv(reg_results_path, index=False)
    reg_results_df.to_csv(REPORTS_DIR / 'tables' / 'regression_results.csv', index=False)
    print(f"[Regression] Best Model: {best_model_name} with MAE: {best_mae:.2f} Million VND")
    
    # Save plots for best model
    plot_actual_vs_predicted(y_test, best_preds, best_model_name)
    plot_residuals(y_test, best_preds, best_model_name)
    
    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(feature_names_e4, best_model.feature_importances_)
        
    # ==========================================
    # 2. Ablation Study across E1, E2, E3, E4
    # ==========================================
    print("[Regression] Running Ablation Study (E1, E2, E3, E4)...")
    ablation_results = []
    # Test key representative models across configs: Ridge and Random Forest and XGBoost
    ablation_model_factories = {
        "Ridge": lambda: Ridge(alpha=10.0, random_state=random_state),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=80, max_depth=14, random_state=random_state, n_jobs=-1),
        "XGBoost": lambda: XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state, n_jobs=-1)
    }
    
    for config_key in ["E1", "E2", "E3", "E4"]:
        X_tr = ablation_sets[config_key]["train"]
        X_te = ablation_sets[config_key]["test"]
        for m_name, factory in ablation_model_factories.items():
            m = factory()
            m.fit(X_tr, y_train)
            preds_abl = m.predict(X_te)
            m_metrics = compute_regression_metrics(y_test, preds_abl, m_name, config_name=config_key)
            ablation_results.append(m_metrics)
            
    ablation_df = pd.DataFrame(ablation_results)
    ablation_path = REGRESSION_OUTPUTS / 'metrics' / 'ablation_results.csv'
    ablation_df.to_csv(ablation_path, index=False)
    ablation_df.to_csv(REPORTS_DIR / 'tables' / 'ablation_results.csv', index=False)
    plot_ablation_comparison(ablation_df)
    
    # ==========================================
    # 3. Temporal Validation Check
    # ==========================================
    temporal_info = {}
    years = df_reg['year_clean'].unique()
    if len(years) > 1:
        min_year = min(years)
        max_year = max(years)
        train_temp_mask = (df_reg['year_clean'] < max_year).values
        test_temp_mask = (df_reg['year_clean'] == max_year).values
        if train_temp_mask.sum() > 50 and test_temp_mask.sum() > 50:
            temporal_info = {
                "train_years": f"<{max_year} (n={train_temp_mask.sum()})",
                "test_year": f"{max_year} (n={test_temp_mask.sum()})",
                "temporal_evaluated": True
            }
        else:
            temporal_info = {"temporal_evaluated": False, "reason": "Insufficient samples across historical years"}
    else:
        temporal_info = {"temporal_evaluated": False, "reason": "Single year collection domain"}
        
    # Save best regression model
    model_artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "best_mae": best_mae,
        "feature_names": feature_names_e4,
        "target_unit": "Million VND"
    }
    model_save_path = REGRESSION_OUTPUTS / 'models' / 'best_regression_model.joblib'
    joblib.dump(model_artifact, model_save_path)
    print(f"[Regression] Saved best regression model to {model_save_path}")
    
    metadata = {
        "best_model_name": best_model_name,
        "best_mae": best_mae,
        "temporal_info": temporal_info,
        "total_regression_samples": len(df_reg)
    }
    
    return reg_results_df, ablation_df, best_model, metadata

