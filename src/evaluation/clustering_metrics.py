"""Clustering evaluation metrics and visualizations."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.decomposition import PCA
from src.config import CLUSTERING_OUTPUTS, REPORTS_DIR

def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    algorithm_name: str,
    metric: str = 'cosine',
    ground_truth_labels: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute Silhouette (cosine), Davies-Bouldin, and Calinski-Harabasz scores,
    and external validation metrics (ARI, NMI) if weak labels are supplied.
    """
    # Filter noise points (label == -1 in DBSCAN)
    valid_mask = labels != -1
    n_clusters = len(set(labels[valid_mask]))
    
    if n_clusters < 2:
        return {
            "Algorithm": algorithm_name,
            "Clusters": n_clusters,
            "Silhouette (Cosine)": np.nan,
            "Davies-Bouldin Index": np.nan,
            "Calinski-Harabasz": np.nan,
            "ARI (vs Weak Labels)": np.nan,
            "NMI (vs Weak Labels)": np.nan
        }
        
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    # Subsample if dataset is large for silhouette speed
    if len(X_valid) > 4000:
        np.random.seed(42)
        idx = np.random.choice(len(X_valid), 4000, replace=False)
        sil = silhouette_score(X_valid[idx], labels_valid[idx], metric=metric)
    else:
        sil = silhouette_score(X_valid, labels_valid, metric=metric)
        
    db = davies_bouldin_score(X_valid, labels_valid)
    ch = calinski_harabasz_score(X_valid, labels_valid)
    
    ari = np.nan
    nmi = np.nan
    if ground_truth_labels is not None:
        gt_valid = ground_truth_labels[valid_mask]
        ari = adjusted_rand_score(gt_valid, labels_valid)
        nmi = normalized_mutual_info_score(gt_valid, labels_valid)
    
    return {
        "Algorithm": algorithm_name,
        "Clusters": n_clusters,
        "Silhouette (Cosine)": round(sil, 4),
        "Davies-Bouldin Index": round(db, 4),
        "Calinski-Harabasz": round(ch, 2),
        "ARI (vs Weak Labels)": round(ari, 4) if not np.isnan(ari) else "N/A",
        "NMI (vs Weak Labels)": round(nmi, 4) if not np.isnan(nmi) else "N/A"
    }

def plot_elbow_curve(k_values: List[int], inertias: List[float], filename: str = "elbow_curve.png") -> str:
    """Plot K-Means Inertia elbow curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, 'bo-', lw=2, markersize=8)
    plt.title('K-Means Elbow Method (Inertia vs Number of Clusters)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Sum of squared distances)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    save_path = CLUSTERING_OUTPUTS / 'plots' / filename
    plt.savefig(save_path, dpi=300)
    plt.savefig(REPORTS_DIR / 'figures' / filename, dpi=300)
    plt.close()
    return str(save_path)

def plot_silhouette_comparison(results_df: pd.DataFrame, filename: str = "silhouette_comparison.png") -> str:
    """Plot bar chart comparing Silhouette scores across algorithms/k."""
    plt.figure(figsize=(10, 5))
    plot_df = results_df.dropna(subset=['Silhouette (Cosine)'])
    sns.barplot(data=plot_df, x='Algorithm', y='Silhouette (Cosine)', palette='crest')
    plt.title('Silhouette Score Comparison (Cosine Metric, Higher is Better)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    plt.grid(True, axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    save_path = CLUSTERING_OUTPUTS / 'plots' / filename
    plt.savefig(save_path, dpi=300)
    plt.savefig(REPORTS_DIR / 'figures' / filename, dpi=300)
    plt.close()
    return str(save_path)

def plot_pca_clusters(X: np.ndarray, labels: np.ndarray, cluster_names: Dict[int, str] = None, filename: str = "pca_clusters.png") -> str:
    """Project high-dimensional SBERT embeddings to 2D via PCA and plot clusters."""
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    
    plt.figure(figsize=(11, 8))
    unique_labels = sorted(set(labels))
    palette = sns.color_palette('tab10', len(unique_labels))
    
    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = cluster_names[lbl] if (cluster_names and lbl in cluster_names) else f"Cluster {lbl}"
        if lbl == -1:
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label="Noise (-1)", color='gray', alpha=0.2, s=15)
        else:
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=name, color=palette[idx], alpha=0.5, s=25)
            
    plt.title(f'2D PCA Projection of Job Profile Clusters (Explained Var: {pca.explained_variance_ratio_.sum()*100:.1f}%)', fontsize=13, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=11)
    plt.ylabel('Principal Component 2', fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = CLUSTERING_OUTPUTS / 'plots' / filename
    plt.savefig(save_path, dpi=300)
    plt.savefig(REPORTS_DIR / 'figures' / filename, dpi=300)
    plt.close()
    return str(save_path)

def plot_cluster_distribution(labels: np.ndarray, cluster_names: Dict[int, str] = None, filename: str = "cluster_distribution.png") -> str:
    """Plot distribution of job postings across discovered clusters."""
    counts = pd.Series(labels).value_counts().sort_index()
    labels_names = [cluster_names[k] if (cluster_names and k in cluster_names) else f"Cluster {k}" for k in counts.index]
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=labels_names, y=counts.values, palette='mako')
    plt.title('Job Postings Distribution Across Discovered Clusters', fontsize=14, fontweight='bold')
    plt.xlabel('Cluster Name', fontsize=12)
    plt.ylabel('Number of Job Postings', fontsize=12)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    
    save_path = CLUSTERING_OUTPUTS / 'plots' / filename
    plt.savefig(save_path, dpi=300)
    plt.savefig(REPORTS_DIR / 'figures' / filename, dpi=300)
    plt.close()
    return str(save_path)
