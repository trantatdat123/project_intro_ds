"""Supervised Salary Prediction regression models, ablation studies, and temporal validation."""
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
    TRAIN_SIZE,
    VAL_SIZE,
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any, Dict[str, Any], pd.DataFrame]:
    """
    Execute full salary regression suite:
    1. 80/10/10 train/val/test split with zero leakage.
    2. Full model comparison on E4 (SBERT + Structured with Year) evaluated on held-out Test set.
    3. Reconciled ablation study across E1, E2, E3, E4.
    4. Temporal validation (Train: <=2025, Test: 2026).
    5. Saves split manifest and best model artifact.
    """
    # Target in Million VND
    y = (df_reg['salary_mean_vnd'] / 1_000_000.0).values
    job_ids = df_reg['unique_job_id'].values
    
    # 80/10/10 split:
    # 80% train, 20% temp
    indices = np.arange(len(df_reg))
    train_idx, temp_idx = train_test_split(indices, test_size=(VAL_SIZE + TEST_SIZE), random_state=random_state)
    # 50/50 split of temp -> 10% val, 10% test
    val_rel, test_rel = train_test_split(np.arange(len(temp_idx)), test_size=0.5, random_state=random_state)
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    
    # Split manifest
    split_records = []
    for idx in train_idx:
        split_records.append({"unique_job_id": job_ids[idx], "task": "regression", "split": "train"})
    for idx in val_idx:
        split_records.append({"unique_job_id": job_ids[idx], "task": "regression", "split": "val"})
    for idx in test_idx:
        split_records.append({"unique_job_id": job_ids[idx], "task": "regression", "split": "test"})
    reg_split_manifest = pd.DataFrame(split_records)
    
    df_train = df_reg.iloc[train_idx].reset_index(drop=True)
    df_val = df_reg.iloc[val_idx].reset_index(drop=True)
    df_test = df_reg.iloc[test_idx].reset_index(drop=True)
    
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    
    sbert_train = sbert_embeddings[train_idx]
    sbert_val = sbert_embeddings[val_idx]
    sbert_test = sbert_embeddings[test_idx]
    
    # Build ablation feature sets strictly fitting encoders on df_train
    ablation_sets = build_ablation_features(
        df_train, df_val, df_test, sbert_train, sbert_val, sbert_test
    )
    
    # Standardized Model Specifications (Reconciled across benchmarks and ablation)
    xgb_params = {
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.08,
        "random_state": random_state,
        "n_jobs": -1
    }
    rf_params = {
        "n_estimators": 100,
        "max_depth": 16,
        "random_state": random_state,
        "n_jobs": -1
    }
    ridge_params = {
        "alpha": 10.0,
        "random_state": random_state
    }
    
    # ==========================================
    # 1. Full Model Comparison on E4 (SBERT + Struct + Year)
    # ==========================================
    X_train_e4 = ablation_sets["E4"]["train"]
    X_val_e4 = ablation_sets["E4"]["val"]
    X_test_e4 = ablation_sets["E4"]["test"]
    feature_names_e4 = ablation_sets["E4"]["names"]
    
    models = {
        "Dummy (Median Baseline)": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(**ridge_params),
        "Random Forest Regressor": RandomForestRegressor(**rf_params),
        "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=100, max_depth=16, random_state=random_state, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(**xgb_params)
    }
    
    reg_results = []
    trained_models = {}
    best_val_mae = float('inf')
    best_model_name = ""
    best_model = None
    best_test_preds = None
    
    for name, reg in models.items():
        print(f"[Regression] Training {name} on E4...")
        reg.fit(X_train_e4, y_train)
        
        # Validation evaluation
        val_preds = reg.predict(X_val_e4)
        val_metrics = compute_regression_metrics(y_val, val_preds, name, config_name="E4")
        
        # Held-out Test evaluation
        test_preds = reg.predict(X_test_e4)
        test_metrics = compute_regression_metrics(y_test, test_preds, name, config_name="E4")
        test_metrics["Val MAE (Million VND)"] = val_metrics["MAE (Million VND)"]
        test_metrics["Val R2 Score"] = val_metrics["R2 Score"]
        
        reg_results.append(test_metrics)
        trained_models[name] = reg
        
        if val_metrics["MAE (Million VND)"] < best_val_mae and name != "Dummy (Median Baseline)":
            best_val_mae = val_metrics["MAE (Million VND)"]
            best_model_name = name
            best_model = reg
            best_test_preds = test_preds
            
    reg_results_df = pd.DataFrame(reg_results)
    reg_results_path = REGRESSION_OUTPUTS / 'metrics' / 'regression_results.csv'
    reg_results_df.to_csv(reg_results_path, index=False)
    reg_results_df.to_csv(REPORTS_DIR / 'tables' / 'regression_results.csv', index=False)
    
    best_test_mae = next(r["MAE (Million VND)"] for r in reg_results if r["Model"] == best_model_name)
    print(f"[Regression] Best Model on Val: {best_model_name} (Val MAE: {best_val_mae:.2f}M, Test MAE: {best_test_mae:.2f}M VND)")
    
    # Save plots for best model on Test set
    plot_actual_vs_predicted(y_test, best_test_preds, best_model_name)
    plot_residuals(y_test, best_test_preds, best_model_name)
    
    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(feature_names_e4, best_model.feature_importances_)
        
    # ==========================================
    # 2. Ablation Study across E1, E2, E3, E4
    # ==========================================
    print("[Regression] Running Reconciled Ablation Study (E1, E2, E3, E4)...")
    ablation_results = []
    
    # Reconciled model factories
    ablation_model_factories = {
        "Ridge": lambda: Ridge(**ridge_params),
        "Random Forest": lambda: RandomForestRegressor(**rf_params),
        "XGBoost": lambda: XGBRegressor(**xgb_params)
    }
    
    for config_key in ["E1", "E2", "E3", "E4"]:
        X_tr = ablation_sets[config_key]["train"]
        X_va = ablation_sets[config_key]["val"]
        X_te = ablation_sets[config_key]["test"]
        
        for m_name, factory in ablation_model_factories.items():
            m = factory()
            m.fit(X_tr, y_train)
            
            val_preds_abl = m.predict(X_va)
            val_abl_metrics = compute_regression_metrics(y_val, val_preds_abl, m_name, config_name=config_key)
            
            test_preds_abl = m.predict(X_te)
            test_abl_metrics = compute_regression_metrics(y_test, test_preds_abl, m_name, config_name=config_key)
            test_abl_metrics["Val MAE (Million VND)"] = val_abl_metrics["MAE (Million VND)"]
            test_abl_metrics["Val R2 Score"] = val_abl_metrics["R2 Score"]
            
            ablation_results.append(test_abl_metrics)
            
    ablation_df = pd.DataFrame(ablation_results)
    ablation_path = REGRESSION_OUTPUTS / 'metrics' / 'ablation_results.csv'
    ablation_df.to_csv(ablation_path, index=False)
    ablation_df.to_csv(REPORTS_DIR / 'tables' / 'ablation_results.csv', index=False)
    plot_ablation_comparison(ablation_df)
    
    # ==========================================
    # 3. Longitudinal / Temporal Validation Check
    # ==========================================
    print("[Regression] Running Temporal Validation (Train: <=2025, Test: 2026)...")
    temporal_train_mask = (df_reg['year_clean'] <= 2025).values
    temporal_test_mask = (df_reg['year_clean'] == 2026).values
    
    temporal_df = pd.DataFrame()
    if temporal_train_mask.sum() > 100 and temporal_test_mask.sum() > 100:
        df_temp_tr = df_reg[temporal_train_mask].reset_index(drop=True)
        df_temp_te = df_reg[temporal_test_mask].reset_index(drop=True)
        y_temp_tr = y[temporal_train_mask]
        y_temp_te = y[temporal_test_mask]
        sbert_temp_tr = sbert_embeddings[temporal_train_mask]
        sbert_temp_te = sbert_embeddings[temporal_test_mask]
        
        # Build features for E4 under temporal split
        temp_features = build_ablation_features(
            df_temp_tr, df_temp_te, df_temp_te, sbert_temp_tr, sbert_temp_te, sbert_temp_te
        )
        X_temp_tr_e4 = temp_features["E4"]["train"]
        X_temp_te_e4 = temp_features["E4"]["test"]
        
        temp_results = []
        for m_name, factory in ablation_model_factories.items():
            tm = factory()
            tm.fit(X_temp_tr_e4, y_temp_tr)
            preds_tm = tm.predict(X_temp_te_e4)
            tm_metrics = compute_regression_metrics(y_temp_te, preds_tm, m_name, config_name="Temporal (<=2025 -> 2026)")
            temp_results.append(tm_metrics)
            
        temporal_df = pd.DataFrame(temp_results)
        temp_path = REGRESSION_OUTPUTS / 'metrics' / 'temporal_validation_results.csv'
        temporal_df.to_csv(temp_path, index=False)
        temporal_df.to_csv(REPORTS_DIR / 'tables' / 'temporal_validation_results.csv', index=False)
        print(f"[Regression] Temporal validation completed. Saved to {temp_path}")
        
    # Save best regression model
    model_artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "best_test_mae": best_test_mae,
        "feature_names": feature_names_e4,
        "target_unit": "Million VND"
    }
    model_save_path = REGRESSION_OUTPUTS / 'models' / 'best_regression_model.joblib'
    joblib.dump(model_artifact, model_save_path)
    print(f"[Regression] Saved best regression model to {model_save_path}")
    
    metadata = {
        "best_model_name": best_model_name,
        "best_val_mae": best_val_mae,
        "best_test_mae": best_test_mae,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "temporal_evaluated": not temporal_df.empty
    }
    
    return reg_results_df, ablation_df, temporal_df, best_model, metadata, reg_split_manifest
