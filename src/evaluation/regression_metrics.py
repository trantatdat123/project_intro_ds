"""Regression evaluation metrics, ablation comparisons, and residual plots."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import REGRESSION_OUTPUTS, REPORTS_DIR

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, config_name: str = "Default") -> Dict[str, Any]:
    """Compute MAE, RMSE, and R2 (assuming targets are in million VND)."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "Config": config_name,
        "Model": model_name,
        "MAE (Million VND)": round(mae, 2),
        "RMSE (Million VND)": round(rmse, 2),
        "R2 Score": round(r2, 4)
    }

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, filename: str = "actual_vs_predicted.png") -> str:
    """Plot actual vs predicted scatter plot with 45-degree reference line."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, color='#1f77b4', edgecolors='none', s=25)
    
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y = x)')
    
    plt.title(f'Actual vs Predicted Salary ({model_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Monthly Salary (Million VND)', fontsize=12)
    plt.ylabel('Predicted Monthly Salary (Million VND)', fontsize=12)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    save_path = REGRESSION_OUTPUTS / 'plots' / filename
    plt.savefig(save_path, dpi=300)
    plt.savefig(REPORTS_DIR / 'figures' / filename, dpi=300)
    plt.close()
    return str(save_path)

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, filename: str = "residual_plot.png") -> str:
    """Plot distribution of residuals (errors)."""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.3, color='#2ca02c', s=25)
    ax1.axhline(0, color='red', linestyle='--', lw=2)
    ax1.set_title(f'Residuals vs Predicted ({model_name})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Salary (Million VND)', fontsize=11)
    ax1.set_ylabel('Residual (Actual - Predicted)', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Residual Distribution Histogram
    sns.histplot(residuals, kde=True, ax=ax2, color='#2ca02c', bins=30)
    ax2.axvline(0, color='red', linestyle='--', lw=2)
    ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Residual (Million VND)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    
    plt.tight_layout()
    save_path = REGRESSION_OUTPUTS / 'plots' / filename
    plt.savefig(save_path, dpi=300)
    plt.savefig(REPORTS_DIR / 'figures' / filename, dpi=300)
    plt.close()
    return str(save_path)

def plot_feature_importance(feature_names: List[str], importances: np.ndarray, top_n: int = 15, filename: str = "feature_importance.png") -> str:
    """Plot top N feature importances for tree-based models."""
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df, x='Importance', y='Feature', palette='mako')
    plt.title(f'Top {top_n} Feature Importances for Salary Prediction', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    save_path = REGRESSION_OUTPUTS / 'plots' / filename
    plt.savefig(save_path, dpi=300)
    plt.savefig(REPORTS_DIR / 'figures' / filename, dpi=300)
    plt.close()
    return str(save_path)

def plot_ablation_comparison(ablation_df: pd.DataFrame, filename: str = "ablation_comparison.png") -> str:
    """Plot MAE and R2 across ablation configurations E1-E4."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.barplot(data=ablation_df, x='Config', y='MAE (Million VND)', hue='Model', ax=ax1, palette='tab10')
    ax1.set_title('Ablation Study: MAE (Lower is better)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Configuration', fontsize=11)
    ax1.set_ylabel('MAE (Million VND)', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    sns.barplot(data=ablation_df, x='Config', y='R2 Score', hue='Model', ax=ax2, palette='tab10')
    ax2.set_title('Ablation Study: R2 Score (Higher is better)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Configuration', fontsize=11)
    ax2.set_ylabel('R2 Score', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    save_path = REGRESSION_OUTPUTS / 'plots' / filename
    plt.savefig(save_path, dpi=300)
    plt.savefig(REPORTS_DIR / 'figures' / filename, dpi=300)
    plt.close()
    return str(save_path)

