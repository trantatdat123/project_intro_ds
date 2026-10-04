"""Runner theo stage cho toàn bộ dự án TinixAI."""

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import platform
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, mean_absolute_error, silhouette_score

from src.data.tinixai import download_dataset, filter_dataset
from src.features.build_features import create_skill_matrix, generate_text_embeddings
from src.models.classifier import train_role_classifier, evaluate_classifier
from src.models.cluster import build_skill_cooccurrence_graph, cluster_embeddings, summarize_clusters, skill_graph_statistics
from src.models.regressor import INPUT_FEATURES, train_salary_regressor, evaluate_regressor, explain_salary_features
from src.models.splitting import assign_company_splits
from src.preprocessing.prepare import prepare_jobs
from src.reporting import (generate_eda, generate_report, classification_figures,
                           salary_figures, graph_figure, cluster_figure)
from src.utils import ROOT, read_frame, save_frame, write_json, fingerprint


PROCESSED = ROOT / "data/processed"
FEATURES = ROOT / "data/features"
MODELS = ROOT / "artifacts/models"
REPORTS = ROOT / "reports"


def load_config(path=ROOT / "configs/pipeline.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataset_signature(frame):
    columns = ["id", "text_clean", "role_standard", "salary_mean", "extracted_skills", "split"]
    return fingerprint("\t".join(map(str, row)) for row in frame[columns].itertuples(index=False, name=None))


def acquire(config, refresh=False):
    raw = download_dataset(refresh=refresh)
    return filter_dataset(raw, ROOT / config["input"])


def prepare(config, limit=None):
    input_path = ROOT / config["input"]
    if not input_path.exists():
        acquire(config)
    frame, quality = prepare_jobs(input_path, usd_to_vnd_rate=config["usd_to_vnd_rate"], limit=limit)
    frame["split"] = assign_company_splits(frame, config["random_state"])
    save_frame(frame, PROCESSED / "jobs_cleaned.parquet")
    quality["split_counts"] = frame.split.value_counts().to_dict()
    quality["split_companies"] = frame.groupby("split").company_group.nunique().to_dict()
    quality["dataset_fingerprint"] = dataset_signature(frame)
    write_json(REPORTS / "data_quality.json", quality)
    print(f"Preprocessing: {len(frame):,} tin; {quality['classification_eligible_rows']:,} nhãn nghề đơn", flush=True)
    return frame


def features(frame, config):
    FEATURES.mkdir(parents=True, exist_ok=True)
    matrix = create_skill_matrix(frame, min_frequency=config["skill_min_frequency"])
    matrix.insert(0, "id", frame.id.to_numpy())
    matrix.to_parquet(FEATURES / "skill_matrix.parquet", index=False)
    write_json(FEATURES / "skill_vocabulary.json", list(matrix.columns[1:]))
    embeddings = None
    if config["embeddings"]["enabled"]:
        options = config["embeddings"]
        embeddings = generate_text_embeddings(frame.text_clean, model_name=options["model"],
                      batch_size=options["batch_size"], max_seq_length=options["max_seq_length"],
                      revision=options.get("revision"),
                      cache_path=FEATURES / "text_embeddings.npy", model_cache=ROOT / "artifacts/embedding_model")
    write_json(REPORTS / "features.json", {"rows": len(frame), "skill_columns": matrix.shape[1] - 1,
               "embedding_shape": list(embeddings.shape) if embeddings is not None else None,
               "embedding_model": config["embeddings"]["model"] if embeddings is not None else None})
    print(f"Features: {matrix.shape[1] - 1} kỹ năng; embeddings: {None if embeddings is None else embeddings.shape}", flush=True)
    return embeddings


def classify(frame, config):
    eligible = frame.loc[frame.classification_eligible].copy()
    subsets = {name: eligible.loc[eligible.split.eq(name)] for name in ("train", "validation", "test")}
    if any(part.empty for part in subsets.values()):
        raise ValueError("Tập phân loại train/validation/test có phần rỗng")
    train, validation, test = (subsets[name] for name in ("train", "validation", "test"))
    missing = set(eligible.role_standard) - set(train.role_standard)
    if missing:
        raise ValueError(f"Các lớp không có trong train: {missing}; điều chỉnh seed/chia tập")
    MODELS.mkdir(parents=True, exist_ok=True)
    results, fitted = {}, {}
    for name in config["classification_models"]:
        print(f"Train classifier: {name}", flush=True)
        model = train_role_classifier(train.text_clean, train.role_standard, name, random_state=config["random_state"])
        results[name] = {"validation": evaluate_classifier(model, validation.text_clean, validation.role_standard),
                         "test": evaluate_classifier(model, test.text_clean, test.role_standard)}
        joblib.dump(model, MODELS / f"classifier_{name}.joblib")
        fitted[name] = model
    selected = max(results, key=lambda name: results[name]["validation"]["macro_f1"])
    joblib.dump(fitted[selected], MODELS / "role_classifier.joblib")
    baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), train.role_standard)
    baseline_f1 = f1_score(test.role_standard, baseline.predict(np.zeros((len(test), 1))), average="macro", zero_division=0)
    payload = {"selected_model": selected, "selection_metric": "validation_macro_f1", "models": results,
               "dataset_fingerprint": dataset_signature(frame),
               "baseline_macro_f1": float(baseline_f1), "split_rows": {key: len(value) for key, value in subsets.items()},
               "split_class_counts": {key: value.role_standard.value_counts().to_dict() for key, value in subsets.items()},
               "label_source": "title_rule", "input": "description_and_requirements_only"}
    write_json(REPORTS / "classification_metrics.json", payload)
    predictions = test[["id", "company_group", "role_standard", "title_clean"]].copy()
    predictions["predicted_role"] = fitted[selected].predict(test.text_clean)
    predictions.to_csv(REPORTS / "classification_predictions.csv", index=False, encoding="utf-8-sig")
    classification_figures(results[selected]["test"])
    print(f"Classifier chọn: {selected}; macro-F1 test={results[selected]['test']['macro_f1']:.3f}", flush=True)
    return payload


def salary(frame, config):
    eligible = frame.loc[frame.salary_eligible & frame.classification_eligible].copy()
    train = eligible.loc[eligible.split.eq("train")]
    validation = eligible.loc[eligible.split.eq("validation")]
    test = eligible.loc[eligible.split.eq("test")]
    if any(part.empty for part in (train, validation, test)):
        raise ValueError("Tập hồi quy train/validation/test có phần rỗng")
    low, high = train.salary_mean.quantile(config["salary_train_quantiles"])
    training_count = len(train)
    train = train.loc[train.salary_mean.between(low, high)]
    results, fitted = {}, {}
    MODELS.mkdir(parents=True, exist_ok=True)
    for name in config["regression_models"]:
        print(f"Train salary model: {name}", flush=True)
        model = train_salary_regressor(train, train.salary_mean, name, random_state=config["random_state"])
        results[name] = {"validation": evaluate_regressor(model, validation, validation.salary_mean),
                         "test": evaluate_regressor(model, test, test.salary_mean)}
        joblib.dump(model, MODELS / f"salary_{name}.joblib")
        fitted[name] = model
    selected = min(results, key=lambda name: results[name]["validation"]["mae_vnd"])
    joblib.dump(fitted[selected], MODELS / "salary_regressor.joblib")
    payload = {"selected_model": selected, "selection_metric": "validation_mae_vnd", "models": results,
               "dataset_fingerprint": dataset_signature(frame),
               "baseline_mae_vnd": float(mean_absolute_error(test.salary_mean, np.full(len(test), train.salary_mean.median()))),
               "training_bounds_vnd": [float(low), float(high)], "training_rows_removed": training_count - len(train),
               "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
               "target": "advertised_salary_midpoint_vnd_month", "feature_columns": INPUT_FEATURES}
    # SHAP của mô hình cây tốt nhất theo validation, kể cả khi Ridge thắng chung.
    tree_names = [name for name in fitted if name in {"lightgbm", "random_forest"}]
    if not tree_names:
        raise ValueError("Cần ít nhất một mô hình cây để tạo SHAP theo plan")
    tree_name = min(tree_names, key=lambda name: results[name]["validation"]["mae_vnd"])
    explanation = explain_salary_features(fitted[tree_name], test, max_samples=config["shap_sample_size"], random_state=config["random_state"])
    payload["shap_model"] = tree_name
    payload["shap_scale"] = "log1p_salary"
    payload["shap_rows"] = len(explanation["values"])
    explanation["summary"].to_csv(REPORTS / "salary_shap_importance.csv", index=False, encoding="utf-8-sig")
    np.save(MODELS / "salary_shap_values.npy", explanation["values"])
    write_json(REPORTS / "salary_metrics.json", payload)
    predictions = test[["id", "role_standard", "year", "salary_mean", "salary_bound"]].copy()
    predictions["predicted_salary_vnd"] = fitted[selected].predict(test[INPUT_FEATURES])
    predictions.to_csv(REPORTS / "salary_predictions.csv", index=False, encoding="utf-8-sig")
    predictions["absolute_error_vnd"] = (predictions.salary_mean - predictions.predicted_salary_vnd).abs()
    predictions.groupby("role_standard").agg(rows=("id", "size"), mae_vnd=("absolute_error_vnd", "mean"),
            median_error_vnd=("absolute_error_vnd", "median")).to_csv(REPORTS / "salary_errors_by_role.csv", encoding="utf-8-sig")
    salary_figures(test.salary_mean, predictions.predicted_salary_vnd, explanation["summary"])
    print(f"Salary chọn: {selected}; MAE test={results[selected]['test']['mae_vnd'] / 1e6:.2f} triệu", flush=True)
    return payload


def cluster(frame, config, embeddings=None):
    graph = build_skill_cooccurrence_graph(frame.extracted_skills, **config["graph"])
    MODELS.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, MODELS / "skill_network.graphml")
    centrality, topology = skill_graph_statistics(graph)
    centrality.to_csv(REPORTS / "skill_centrality.csv", index=False, encoding="utf-8-sig")
    matrix = nx.to_pandas_adjacency(graph, weight="weight")
    for node in graph:
        matrix.loc[node, node] = graph.nodes[node]["job_count"]
    matrix.to_csv(REPORTS / "skill_cooccurrence_matrix.csv", encoding="utf-8-sig")
    edges = pd.DataFrame([{"skill_a": a, "skill_b": b, **attributes} for a, b, attributes in graph.edges(data=True)])
    edges.to_csv(REPORTS / "skill_cooccurrence.csv", index=False, encoding="utf-8-sig")
    graph_figure(graph)
    payload = {"graph": topology, "semantic": {}, "dataset_fingerprint": dataset_signature(frame)}
    if config["embeddings"]["enabled"]:
        if embeddings is None:
            embeddings = features(frame, config)
        options = config["clustering"]
        print("UMAP + KMeans/HDBSCAN", flush=True)
        result = cluster_embeddings(embeddings, **options, random_state=config["random_state"])
        from hdbscan import HDBSCAN
        density_model = HDBSCAN(min_cluster_size=options["min_cluster_size"], min_samples=10, prediction_data=True)
        density_labels = density_model.fit_predict(result["reduced"])
        non_noise = density_labels >= 0
        n_clusters = len(set(density_labels[non_noise]))
        density_score = None
        if 1 < n_clusters < int(non_noise.sum()):
            density_score = float(silhouette_score(result["reduced"][non_noise], density_labels[non_noise],
                                                  sample_size=min(2000, int(non_noise.sum())), random_state=config["random_state"]))
        payload["semantic"] = {"kmeans": result["metrics"], "hdbscan": {
            "method": "hdbscan", "clusters": n_clusters, "noise_rows": int((~non_noise).sum()),
            "silhouette_reduced_space": density_score, "rows": len(frame)}}
        labels = frame[["id", "title_clean", "role_standard"]].copy()
        labels["kmeans_cluster"] = result["labels"]
        labels["hdbscan_cluster"] = density_labels
        labels.to_csv(REPORTS / "cluster_assignments.csv", index=False, encoding="utf-8-sig")
        summarize_clusters(frame, result["labels"]).to_csv(REPORTS / "cluster_summary.csv", index=False, encoding="utf-8-sig")
        joblib.dump({"umap": result["reducer"], "kmeans": result["model"], "hdbscan": density_model}, MODELS / "semantic_clustering.joblib")
        np.save(FEATURES / "umap_coordinates.npy", result["reduced"])
        cluster_figure(result["reduced"], result["labels"])
    write_json(REPORTS / "clustering_metrics.json", payload)
    return payload


def report(frame, config):
    signature = dataset_signature(frame)
    def optional(filename):
        path = REPORTS / filename
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if result and result.get("dataset_fingerprint") not in (None, signature):
            raise ValueError(f"{filename} thuộc dữ liệu khác; chạy lại stage tương ứng hoặc all")
        return result
    path = generate_report(frame, config, optional("data_quality.json"), optional("classification_metrics.json"),
                           optional("salary_metrics.json"), optional("clustering_metrics.json"))
    samples = pd.concat([group.sample(min(8, len(group)), random_state=config["random_state"])
                         for _, group in frame.groupby("role_standard")])
    annotation = samples[["id", "title_clean", "role_standard", "description_clean"]].copy()
    annotation["description_clean"] = annotation.description_clean.str.slice(0, 1600)
    annotation["human_role"] = ""
    annotation["notes"] = ""
    annotation.to_csv(REPORTS / "annotation_template.csv", index=False, encoding="utf-8-sig")
    print("Báo cáo:", path, flush=True)
    return path


def run_pipeline(stage="all", config_path=ROOT / "configs/pipeline.json", *, no_embeddings=False, limit=None, refresh=False):
    config = load_config(config_path)
    if no_embeddings:
        config["embeddings"]["enabled"] = False
    for folder in (PROCESSED, FEATURES, MODELS, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)
    if stage == "acquire":
        return acquire(config, refresh)
    cleaned = PROCESSED / "jobs_cleaned.parquet"
    frame = prepare(config, limit) if stage in {"all", "prepare"} or not cleaned.exists() else read_frame(cleaned)
    if "split" not in frame:
        frame = prepare(config, limit)
    embeddings = None
    if stage in {"all", "features"}:
        embeddings = features(frame, config)
    if stage in {"all", "eda"}:
        generate_eda(frame)
    if stage in {"all", "classify"}:
        classify(frame, config)
    if stage in {"all", "salary"}:
        salary(frame, config)
    if stage in {"all", "cluster"}:
        cluster(frame, config, embeddings)
    if stage in {"all", "report"}:
        report(frame, config)
    libraries = {}
    for package in ("pandas", "numpy", "scikit-learn", "lightgbm", "shap", "sentence-transformers", "torch", "umap-learn", "hdbscan", "flashtext"):
        try:
            libraries[package] = version(package)
        except PackageNotFoundError:
            libraries[package] = None
    input_metadata = (ROOT / config["input"]).with_suffix(".metadata.json")
    provenance = json.loads(input_metadata.read_text(encoding="utf-8")) if input_metadata.exists() else {}
    write_json(REPORTS / "run_manifest.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "stage": stage, "rows": len(frame), "config": config, "limit": limit,
               "dataset_fingerprint": dataset_signature(frame), "source_revision": provenance.get("revision"),
               "python": platform.python_version(), "libraries": libraries})
    return frame


def main():
    parser = argparse.ArgumentParser(description="JobLens Vietnam: TinixAI pipeline")
    parser.add_argument("--stage", choices=["all", "acquire", "prepare", "features", "eda", "classify", "salary", "cluster", "report"], default="all")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.json")
    parser.add_argument("--no-embeddings", action="store_true", help="Bỏ sentence model/semantic clustering trong lượt này")
    parser.add_argument("--limit", type=int, help="Giới hạn số dòng khi prepare/all; mặc định toàn bộ")
    parser.add_argument("--refresh", action="store_true", help="Cập nhật raw từ Hub khi stage acquire")
    args = parser.parse_args()
    run_pipeline(args.stage, args.config, no_embeddings=args.no_embeddings, limit=args.limit, refresh=args.refresh)


if __name__ == "__main__":
    main()
