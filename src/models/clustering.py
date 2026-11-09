"""Unsupervised Job and Skill Clustering models."""
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    RANDOM_STATE,
    CLUSTERING_OUTPUTS,
    REPORTS_DIR
)
from src.evaluation.clustering_metrics import (
    compute_clustering_metrics,
    plot_elbow_curve,
    plot_silhouette_comparison,
    plot_pca_clusters,
    plot_cluster_distribution
)

def extract_cluster_keywords(texts: List[str], labels: np.ndarray, top_n: int = 10) -> Dict[int, List[str]]:
    """Extract top distinguishing keywords for each cluster using TF-IDF."""
    tfidf = TfidfVectorizer(max_features=2000, stop_words=None, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(texts)
    terms = np.array(tfidf.get_feature_names_out())
    
    cluster_keywords = {}
    for cl in sorted(set(labels)):
        if cl == -1: # Skip noise
            continue
        mask = labels == cl
        if mask.sum() == 0:
            continue
        mean_tfidf = np.asarray(X_tfidf[mask].mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[::-1][:top_n]
        cluster_keywords[cl] = list(terms[top_indices])
        
    return cluster_keywords

def run_clustering_experiments(
    df_cluster: pd.DataFrame,
    composite_embeddings: np.ndarray,
    composite_texts: List[str],
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Dict[str, Any]]:
    """
    Run 4 clustering algorithm families:
    1. K-Means across K = [3, 4, 5, 6, 8, 10]
    2. Agglomerative Clustering
    3. DBSCAN
    4. Gaussian Mixture Model (on PCA-reduced embeddings)
    """
    print(f"[Clustering] Starting clustering on {len(composite_embeddings)} records...")
    
    # 1. K-Means Sweep across K = [3, 4, 5, 6, 8, 10]
    k_range = [3, 4, 5, 6, 8, 10]
    inertias = []
    kmeans_results = []
    kmeans_models = {}
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km_labels = km.fit_predict(composite_embeddings)
        inertias.append(km.inertia_)
        metrics = compute_clustering_metrics(composite_embeddings, km_labels, f"K-Means (k={k})")
        kmeans_results.append(metrics)
        kmeans_models[k] = (km, km_labels)
        
    plot_elbow_curve(k_range, inertias)
    
    # Select best K based on silhouette score
    valid_km = [r for r in kmeans_results if not np.isnan(r["Silhouette Score"])]
    best_k_entry = max(valid_km, key=lambda x: x["Silhouette Score"])
    best_k = best_k_entry["Clusters"]
    best_kmeans, best_kmeans_labels = kmeans_models[best_k]
    print(f"[Clustering] Best K-Means cluster count: k={best_k} with Silhouette: {best_k_entry['Silhouette Score']:.4f}")
    
    # 2. Agglomerative Clustering (on a representative sample of 3000 to keep memory O(N^2) tractable)
    agg_sample_size = min(3000, len(composite_embeddings))
    agg_indices = np.random.RandomState(random_state).choice(len(composite_embeddings), agg_sample_size, replace=False)
    agg_emb = composite_embeddings[agg_indices]
    
    agg = AgglomerativeClustering(n_clusters=best_k, metric='cosine', linkage='average')
    agg_labels = agg.fit_predict(agg_emb)
    agg_metrics = compute_clustering_metrics(agg_emb, agg_labels, f"Agglomerative (k={best_k})")
    
    # 3. DBSCAN (cosine metric)
    dbscan = DBSCAN(eps=0.45, min_samples=15, metric='cosine')
    db_labels = dbscan.fit_predict(agg_emb)
    db_metrics = compute_clustering_metrics(agg_emb, db_labels, "DBSCAN (eps=0.45)")
    
    # 4. Gaussian Mixture Model (GMM on PCA 25-dim reduced embeddings)
    pca = PCA(n_components=25, random_state=random_state)
    X_pca = pca.fit_transform(composite_embeddings)
    gmm = GaussianMixture(n_components=best_k, random_state=random_state, covariance_type='diag')
    gmm_labels = gmm.fit_predict(X_pca)
    gmm_metrics = compute_clustering_metrics(composite_embeddings, gmm_labels, f"GMM (k={best_k})")
    
    # Combine results
    all_clustering_results = kmeans_results + [agg_metrics, db_metrics, gmm_metrics]
    clustering_results_df = pd.DataFrame(all_clustering_results)
    res_path = CLUSTERING_OUTPUTS / 'metrics' / 'clustering_results.csv'
    clustering_results_df.to_csv(res_path, index=False)
    clustering_results_df.to_csv(REPORTS_DIR / 'tables' / 'clustering_results.csv', index=False)
    
    plot_silhouette_comparison(clustering_results_df)
    
    # ==========================================
    # Interpret Clusters of Best Model (K-Means)
    # ==========================================
    cluster_keywords = extract_cluster_keywords(composite_texts, best_kmeans_labels, top_n=10)
    
    # Name clusters based on keywords
    cluster_names = {}
    for cl, kws in cluster_keywords.items():
        kw_str = " ".join(kws).lower()
        if any(w in kw_str for w in ['ai', 'learning', 'python', 'model', 'trí tuệ']):
            cluster_names[cl] = f"Cluster {cl}: AI & Machine Learning"
        elif any(w in kw_str for w in ['data', 'analyst', 'bi', 'sql', 'power', 'tableau']):
            cluster_names[cl] = f"Cluster {cl}: Data Analytics & BI"
        elif any(w in kw_str for w in ['java', 'spring', 'backend', 'api', 'microservice']):
            cluster_names[cl] = f"Cluster {cl}: Backend / Software Engineering"
        elif any(w in kw_str for w in ['tester', 'test', 'qa', 'qc', 'kiểm thử']):
            cluster_names[cl] = f"Cluster {cl}: QA / QC & Testing"
        elif any(w in kw_str for w in ['cloud', 'devops', 'aws', 'docker', 'system']):
            cluster_names[cl] = f"Cluster {cl}: DevOps & Cloud Infrastructure"
        else:
            cluster_names[cl] = f"Cluster {cl}: {kws[0].title()} / {kws[1].title()}"
            
    # Save keywords table
    kw_records = []
    for cl, kws in cluster_keywords.items():
        kw_records.append({
            "Cluster_ID": cl,
            "Interpreted_Name": cluster_names[cl],
            "Top_10_Keywords": ", ".join(kws),
            "Job_Count": int((best_kmeans_labels == cl).sum())
        })
    kw_df = pd.DataFrame(kw_records)
    kw_path = CLUSTERING_OUTPUTS / 'metrics' / 'cluster_keywords.csv'
    kw_df.to_csv(kw_path, index=False)
    kw_df.to_csv(REPORTS_DIR / 'tables' / 'cluster_keywords.csv', index=False)
    
    # Save cluster assignments
    assignments_df = df_cluster[['unique_job_id', 'job_title', 'company_name', 'location']].copy()
    assignments_df['cluster_id'] = best_kmeans_labels
    assignments_df['cluster_name'] = assignments_df['cluster_id'].map(cluster_names)
    assign_path = CLUSTERING_OUTPUTS / 'metrics' / 'cluster_assignments.csv'
    assignments_df.to_csv(assign_path, index=False)
    
    # Save plots
    plot_pca_clusters(composite_embeddings, best_kmeans_labels, cluster_names)
    plot_cluster_distribution(best_kmeans_labels, cluster_names)
    
    # Save best model
    model_artifact = {
        "model": best_kmeans,
        "best_k": best_k,
        "cluster_names": cluster_names,
        "cluster_keywords": cluster_keywords
    }
    model_save_path = CLUSTERING_OUTPUTS / 'models' / 'best_clustering_model.joblib'
    joblib.dump(model_artifact, model_save_path)
    print(f"[Clustering] Saved best clustering model to {model_save_path}")
    
    metadata = {
        "best_k": best_k,
        "best_silhouette": best_k_entry["Silhouette Score"],
        "cluster_names": cluster_names
    }
    
    return clustering_results_df, assignments_df, best_kmeans, metadata

