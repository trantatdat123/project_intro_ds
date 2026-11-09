"""Classification evaluation metrics and visualizations."""
import os
import matplotlib
# Use non-interactive Agg backend to avoid display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from src.config import CLASSIFICATION_OUTPUTS, REPORTS_DIR

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
    """Compute comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision (Macro)": round(prec_macro, 4),
        "Recall (Macro)": round(rec_macro, 4),
        "Macro-F1": round(f1_macro, 4),
        "Weighted-F1": round(f1_weighted, 4)
    }

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], filename: str = "confusion_matrix.png") -> str:
    """Plot and save normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix (Best Model)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Job Role', fontsize=12)
    plt.ylabel('True Job Role', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = CLASSIFICATION_OUTPUTS / 'plots' / filename
    plt.savefig(save_path, dpi=300)
    # Also save to reports figures
    plt.savefig(REPORTS_DIR / 'figures' / filename, dpi=300)
    plt.close()
    return str(save_path)

def plot_model_comparison(results_df: pd.DataFrame, filename: str = "model_comparison.png") -> str:
    """Plot bar chart comparing model metrics."""
    plt.figure(figsize=(10, 6))
    plot_df = results_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'Macro-F1', 'Weighted-F1'],
                             var_name='Metric', value_name='Score')
    sns.barplot(data=plot_df, x='Model', y='Score', hue='Metric', palette='viridis')
    plt.title('Classification Models Comparison (Macro-F1 & Accuracy)', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20, ha='right')
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    save_path = CLASSIFICATION_OUTPUTS / 'plots' / filename
    plt.savefig(save_path, dpi=300)
    plt.savefig(REPORTS_DIR / 'figures' / filename, dpi=300)
    plt.close()
    return str(save_path)

def save_classification_reports(
    results_list: List[Dict[str, Any]],
    best_y_true: np.ndarray,
    best_y_pred: np.ndarray,
    class_names: List[str]
) -> pd.DataFrame:
    """Save metrics table and per-class classification report."""
    results_df = pd.DataFrame(results_list)
    results_path = CLASSIFICATION_OUTPUTS / 'metrics' / 'classification_results.csv'
    results_df.to_csv(results_path, index=False)
    results_df.to_csv(REPORTS_DIR / 'tables' / 'classification_results.csv', index=False)
    
    # Detailed per-class report for best model
    cr_dict = classification_report(best_y_true, best_y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cr_df = pd.DataFrame(cr_dict).transpose().reset_index()
    cr_df.rename(columns={'index': 'Class'}, inplace=True)
    cr_path = CLASSIFICATION_OUTPUTS / 'metrics' / 'classification_report.csv'
    cr_df.to_csv(cr_path, index=False)
    cr_df.to_csv(REPORTS_DIR / 'tables' / 'classification_report.csv', index=False)
    
    plot_model_comparison(results_df)
    plot_confusion_matrix(best_y_true, best_y_pred, class_names)
    
    return results_df

