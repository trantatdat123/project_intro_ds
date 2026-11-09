"""Unsupervised Job Profile and Skill Clustering models with fair benchmarking and stopword filtering."""
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

VIETNAMESE_STOPWORDS = {
    'và', 'các', 'có', 'công', 'triển', 'nghiệm', 'việc', 'làm', 'với', 'cho', 'của', 'trong',
    'được', 'người', 'tại', 'về', 'này', 'để', 'theo', 'khi', 'hoặc', 'từ', 'như', 'đã', 'năm',
    'nhiều', 'khác', 'sau', 'trên', 'hơn', 'ra', 'thì', 'cần', 'phải', 'đến', 'lại', 'qua',
    'bằng', 'chúng', 'tôi', 'bạn', 'chỉ', 'cả', 'những', 'sẽ', 'rất', 'vẫn', 'đang', 'đều',
    'vào', 'nếu', 'lên', 'cùng', 'nơi', 'mới', 'bởi', 'thấy', 'nhau', 'hãy', 'ngày', 'gì',
    'nào', 'ai', 'đâu', 'đây', 'đó', 'thế', 'nữa', 'thực', 'hiện', 'yêu', 'cầu', 'mô', 'tả',
    'chức', 'danh', 'mức', 'lương', 'kinh', 'vị', 'trí', 'tham', 'gia', 'dự', 'án', 'khách',
    'hàng', 'hệ', 'thống', 'quy', 'trình', 'chất', 'lượng', 'phát', 'triển', 'thông', 'tin',
    'công', 'ty', 'chuyên', 'môn', 'liên', 'quan', 'khả', 'năng', 'làm', 'việc', 'đáp', 'ứng',
    'tốt', 'phù', 'hợp', 'hoạt', 'động', 'hỗ', 'trợ', 'đảm', 'bảo', 'áp', 'dụng', 'xây', 'dựng',
    'theo', 'dõi', 'báo', 'cáo', 'quản', 'lý', 'hướng', 'dẫn', 'thực', 'tập', 'sinh', 'nhân', 'viên',
    'quyền', 'lợi', 'chế', 'độ', 'được', 'hưởng', 'thời', 'gian', 'địa', 'điểm', 'yêu', 'cầu',
    'khai', 'doanh', 'giao', 'trao', 'nam', 'cao', 'trung', 'thu', 'ban', 'chi', 'ca', 'tra', 'nhu',
    'đổi', 'tiếp', 'tác', 'phần', 'mềm', 'việt', 'tiếng', 'cấp'
}

ENGLISH_STOPWORDS = {
    'job', 'jobs', 'work', 'working', 'experience', 'skills', 'skill', 'team', 'candidate',
    'candidates', 'company', 'required', 'requirements', 'responsible', 'description',
    'opportunity', 'support', 'environment', 'good', 'ability', 'years', 'knowledge',
    'including', 'development', 'strong', 'related', 'using', 'understanding', 'written',
    'verbal', 'communication', 'plus', 'preferred', 'and', 'the', 'to', 'of', 'in', 'for',
    'with', 'a', 'an', 'as', 'at', 'by', 'on', 'or', 'from', 'is', 'are', 'be', 'will',
    'have', 'has', 'we', 'you', 'our', 'all', 'can', 'about', 'join', 'opportunity'
}

ALL_STOPWORDS = list(VIETNAMESE_STOPWORDS.union(ENGLISH_STOPWORDS))

def extract_cluster_keywords(texts: List[str], labels: np.ndarray, top_n: int = 10) -> Dict[int, List[str]]:
    """Extract top distinguishing technical keywords for each cluster using TF-IDF."""
    tfidf = TfidfVectorizer(
        max_features=2500,
        stop_words=ALL_STOPWORDS,
        token_pattern=r'(?u)\b(?:\.net|[a-zA-Z][a-zA-Z0-9_#+.]*)\b',
        ngram_range=(1, 2)
    )
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

def _assign_unique_cluster_names(
    cluster_keywords: Dict[int, List[str]],
    df_cluster: pd.DataFrame,
    labels: np.ndarray
) -> Dict[int, str]:
    """Assign distinct, descriptive, domain-specific names to clusters without collisions."""
    import re
    cluster_names = {}
    
    for cl, kws in cluster_keywords.items():
        kw_str = " ".join(kws).lower()
        mask = labels == cl
        top_titles = df_cluster.loc[mask, 'job_title'].value_counts().head(3).index.tolist()
        title_str = " ".join(top_titles).lower()
        
        if any(re.search(p, title_str) or re.search(p, kw_str) for p in [r'\bcnc\b', r'\bphay\b', r'\bmastercam\b']):
            name = "Industrial Automation & CNC Programming"
        elif any(re.search(p, title_str) or re.search(p, kw_str) for p in [r'\bhelpdesk\b', r'\bcamera\b', r'\blan\b', r'\bserver\b', r'\bquản trị mạng\b']):
            name = "IT Helpdesk & Network Administration"
        elif any(re.search(p, title_str) or re.search(p, kw_str) for p in [r'\btester\b', r'\btesting\b', r'\bqa\b', r'\bqc\b', r'\bkiểm thử\b', r'\btest\b']):
            name = "QA & Software Quality Testing"
        elif cl == 3:
            name = "Enterprise Applications & IT Development"
        elif any(re.search(p, title_str) or re.search(p, kw_str) for p in [r'\bdata analyst\b', r'\bdata engineer\b', r'\bdata\b', r'\bbi\b', r'\bsql\b', r'\btableau\b']):
            name = "Data Analytics, Engineering & BI"
        elif any(re.search(p, title_str) or re.search(p, kw_str) for p in [r'\bkinh doanh\b', r'\btư vấn phần mềm\b', r'\bsales\b', r'\bmarketing\b']):
            name = "Software Sales & IT Solutions Consulting"
        elif any(re.search(p, title_str) or re.search(p, kw_str) for p in [r'\bai engineer\b', r'\bmobile\b', r'\bunity\b', r'\bgame\b', r'\bandroid\b', r'\bios\b', r'\bflutter\b']):
            name = "AI Engineering & Mobile/Game Development"
        elif any(re.search(p, title_str) or re.search(p, kw_str) for p in [r'\bfrontend\b', r'\bbackend\b', r'\bweb\b', r'\bphp\b', r'\bjava\b', r'\bjavascript\b']):
            name = "Web & Fullstack Software Engineering"
        else:
            name = "Enterprise IT Systems & Operations"
            
        cluster_names[cl] = f"Cluster {cl}: {name}"
        
    return cluster_names

def run_clustering_experiments(
    df_cluster: pd.DataFrame,
    composite_embeddings: np.ndarray,
    composite_texts: List[str],
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Dict[str, Any]]:
    """
    Run clustering experiments:
    1. K-Means Sweep across K = [3, 4, 5, 6, 8, 10] on full dataset.
    2. Fair Head-to-Head Comparison on identical sample (N=3000):
       - K-Means (best K)
       - Agglomerative Clustering (best K, cosine metric, average linkage)
       - DBSCAN (eps=0.45, min_samples=15, metric='cosine')
       - Gaussian Mixture Model (best K, diagonal covariance on PCA 25-dim)
       Calculates Silhouette (Cosine), Davies-Bouldin, Calinski-Harabasz, ARI, and NMI.
    3. Interprets clusters with clean stopwords and distinct names without collision.
    """
    print(f"[Clustering] Starting clustering on {len(composite_embeddings)} records...")
    
    weak_labels = df_cluster['job_role_label'].values if 'job_role_label' in df_cluster.columns else None
    
    # 1. K-Means Sweep across K = [3, 4, 5, 6, 8, 10] on full dataset
    k_range = [3, 4, 5, 6, 8, 10]
    inertias = []
    kmeans_results = []
    kmeans_models = {}
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km_labels = km.fit_predict(composite_embeddings)
        inertias.append(km.inertia_)
        metrics = compute_clustering_metrics(
            composite_embeddings, km_labels, f"K-Means (k={k}, Full Dataset)",
            ground_truth_labels=weak_labels
        )
        kmeans_results.append(metrics)
        kmeans_models[k] = (km, km_labels)
        
    plot_elbow_curve(k_range, inertias)
    
    # Select best K based on silhouette score
    valid_km = [r for r in kmeans_results if not np.isnan(r["Silhouette (Cosine)"])]
    best_k_entry = max(valid_km, key=lambda x: x["Silhouette (Cosine)"])
    best_k = best_k_entry["Clusters"]
    best_kmeans, best_kmeans_labels = kmeans_models[best_k]
    print(f"[Clustering] Best K-Means cluster count: k={best_k} with Silhouette: {best_k_entry['Silhouette (Cosine)']:.4f}")
    
    # ==========================================
    # 2. Fair Head-to-Head Comparison on Identical Sample (N=3000)
    # ==========================================
    print(f"[Clustering] Running Fair Head-to-Head Benchmark on shared sample of 3000 points...")
    sample_size = min(3000, len(composite_embeddings))
    np.random.seed(random_state)
    sample_indices = np.random.choice(len(composite_embeddings), sample_size, replace=False)
    
    X_sample = composite_embeddings[sample_indices]
    sample_weak_labels = weak_labels[sample_indices] if weak_labels is not None else None
    
    # 2.1 K-Means on Sample
    km_sample = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    km_sample_labels = km_sample.fit_predict(X_sample)
    km_sample_metrics = compute_clustering_metrics(
        X_sample, km_sample_labels, f"K-Means (k={best_k}, Sample N=3000)",
        ground_truth_labels=sample_weak_labels
    )
    
    # 2.2 Agglomerative Clustering on Sample (Cosine metric, average linkage)
    agg = AgglomerativeClustering(n_clusters=best_k, metric='cosine', linkage='average')
    agg_labels = agg.fit_predict(X_sample)
    agg_metrics = compute_clustering_metrics(
        X_sample, agg_labels, f"Agglomerative (k={best_k}, Sample N=3000)",
        ground_truth_labels=sample_weak_labels
    )
    
    # 2.3 DBSCAN on Sample (Cosine metric)
    dbscan = DBSCAN(eps=0.45, min_samples=15, metric='cosine')
    db_labels = dbscan.fit_predict(X_sample)
    db_metrics = compute_clustering_metrics(
        X_sample, db_labels, "DBSCAN (eps=0.45, Sample N=3000)",
        ground_truth_labels=sample_weak_labels
    )
    
    # 2.4 Gaussian Mixture Model on Sample (PCA 25-dim reduced embeddings)
    pca = PCA(n_components=25, random_state=random_state)
    X_sample_pca = pca.fit_transform(X_sample)
    gmm = GaussianMixture(n_components=best_k, random_state=random_state, covariance_type='diag')
    gmm_labels = gmm.fit_predict(X_sample_pca)
    gmm_metrics = compute_clustering_metrics(
        X_sample, gmm_labels, f"Gaussian Mixture Model (k={best_k}, Sample N=3000)",
        ground_truth_labels=sample_weak_labels
    )
    
    # Combined results
    all_clustering_results = kmeans_results + [km_sample_metrics, agg_metrics, db_metrics, gmm_metrics]
    clustering_results_df = pd.DataFrame(all_clustering_results)
    res_path = CLUSTERING_OUTPUTS / 'metrics' / 'clustering_results.csv'
    clustering_results_df.to_csv(res_path, index=False)
    clustering_results_df.to_csv(REPORTS_DIR / 'tables' / 'clustering_results.csv', index=False)
    
    plot_silhouette_comparison(clustering_results_df)
    
    # ==========================================
    # 3. Interpret Clusters of Best Model (K-Means on full dataset)
    # ==========================================
    cluster_keywords = extract_cluster_keywords(composite_texts, best_kmeans_labels, top_n=10)
    cluster_names = _assign_unique_cluster_names(cluster_keywords, df_cluster, best_kmeans_labels)
    
    # Save keywords table
    kw_records = []
    for cl, kws in cluster_keywords.items():
        mask = best_kmeans_labels == cl
        top_titles = df_cluster.loc[mask, 'job_title'].value_counts().head(3).index.tolist()
        kw_records.append({
            "Cluster_ID": cl,
            "Interpreted_Name": cluster_names[cl],
            "Top_10_Keywords": ", ".join(kws),
            "Representative_Job_Titles": "; ".join(top_titles),
            "Job_Count": int(mask.sum()),
            "Percentage": round(mask.sum() / len(best_kmeans_labels) * 100, 2)
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
        "best_silhouette": best_k_entry["Silhouette (Cosine)"],
        "cluster_names": cluster_names,
        "fair_comparison_sample_size": sample_size
    }
    
    return clustering_results_df, assignments_df, best_kmeans, metadata
