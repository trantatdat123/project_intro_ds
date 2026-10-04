"""Đồ thị kỹ năng và gom cụm JD bằng UMAP + KMeans/HDBSCAN."""

from collections import Counter
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def build_skill_cooccurrence_graph(skills_list, min_skill_count=20, min_cooccurrence=10):
    rows = [set(skills) for skills in skills_list]
    frequencies = Counter(skill for skills in rows for skill in skills)
    keep = {skill for skill, count in frequencies.items() if count >= min_skill_count}
    pairs = Counter(pair for skills in rows for pair in combinations(sorted(skills & keep), 2))
    graph = nx.Graph()
    graph.add_nodes_from((skill, {"job_count": frequencies[skill]}) for skill in sorted(keep))
    for (a, b), count in pairs.items():
        if count >= min_cooccurrence:
            graph.add_edge(a, b, weight=count, jaccard=count / (frequencies[a] + frequencies[b] - count),
                           lift=count * len(rows) / (frequencies[a] * frequencies[b]))
    return graph


def skill_graph_statistics(graph):
    """Centrality không trọng số và sức mạnh cạnh; count không phải khoảng cách."""
    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, weight=None)
    table = pd.DataFrame([{"skill": node, "job_count": graph.nodes[node]["job_count"],
                          "degree": graph.degree(node), "weighted_degree": graph.degree(node, weight="weight"),
                          "degree_centrality": degree[node], "betweenness_centrality": betweenness[node]}
                         for node in graph])
    if not table.empty:
        table = table.sort_values("weighted_degree", ascending=False)
    summary = {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
               "density": nx.density(graph), "components": nx.number_connected_components(graph)}
    return table, summary


def cluster_embeddings(embeddings, n_clusters=8, *, method="kmeans", random_state=42,
                       n_neighbors=20, n_components=15, min_cluster_size=35):
    import umap
    values = np.asarray(embeddings)
    if len(values) < 5:
        raise ValueError("Cần ít nhất 5 JD để gom cụm")
    reducer = umap.UMAP(n_neighbors=min(n_neighbors, len(values) - 1),
                        n_components=min(n_components, len(values) - 2), metric="cosine",
                        min_dist=0.0, random_state=random_state, n_jobs=1)
    reduced = reducer.fit_transform(values)
    if method == "kmeans":
        model = KMeans(n_clusters=min(n_clusters, len(values) - 1), n_init=10, random_state=random_state)
    elif method == "hdbscan":
        from hdbscan import HDBSCAN
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10, prediction_data=True)
    else:
        raise ValueError("method phải là kmeans hoặc hdbscan")
    labels = model.fit_predict(reduced)
    eligible = labels >= 0
    count = len(set(labels[eligible]))
    silhouette = None
    if 1 < count < int(eligible.sum()):
        silhouette = float(silhouette_score(reduced[eligible], labels[eligible], sample_size=min(2000, int(eligible.sum())), random_state=random_state))
    return {"labels": labels, "reduced": reduced, "reducer": reducer, "model": model,
            "metrics": {"method": method, "clusters": count, "noise_rows": int((labels < 0).sum()),
                        "silhouette_reduced_space": silhouette, "rows": len(labels)}}


def summarize_clusters(frame, labels):
    frame = frame.copy()
    frame["cluster"] = labels
    rows = []
    for label, group in frame.groupby("cluster"):
        skills = Counter(skill for values in group.extracted_skills for skill in set(values))
        rows.append({"cluster": int(label), "rows": len(group),
                     "top_skills": ", ".join(skill for skill, _ in skills.most_common(8)),
                     "top_roles": ", ".join(f"{role}: {count}" for role, count in group.role_standard.value_counts().head(3).items()),
                     "median_salary_vnd": group.salary_mean.median()})
    return pd.DataFrame(rows)
