"""Tải TinixAI và tái tạo bộ lọc title từ notebook, không crawl website."""

import html
import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.utils import ROOT, fold_text, write_json


def download_dataset(raw_dir=ROOT / "data/raw/tinixai_vietnamese_job_descriptions", revision="main", refresh=False):
    raw_dir = Path(raw_dir)
    local = raw_dir / "data.parquet"
    if local.exists() and not refresh:
        return local
    from huggingface_hub import HfApi, hf_hub_download
    repo = "tinixai/vietnamese-job-descriptions"
    sha = HfApi().dataset_info(repo, revision=revision).sha
    path = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename="data.parquet",
                               revision=sha, local_dir=str(raw_dir)))
    write_json(raw_dir / "source.json", {"dataset": repo, "revision": sha, "license": "CC-BY-NC-4.0"})
    return path


def filter_dataset(source_path, output_path=ROOT / "data/processed/tinixai_it_ai_data_jobs.parquet",
                   filters_path=ROOT / "configs/title_filters.json", batch_size=10000):
    if batch_size <= 0:
        raise ValueError("batch_size phải > 0")
    config = json.loads(Path(filters_path).read_text(encoding="utf-8"))
    rules = {}
    for group, phrases in config["phrases"].items():
        terms = sorted({fold_text(value) for value in phrases}, key=lambda value: (-len(value), value))
        raw = sorted(config["acronyms"].get(group, []), key=lambda value: (-len(value), value))
        rules[group] = (
            re.compile(r"(?<![a-z0-9+#])(?:" + "|".join(map(re.escape, terms)) + r")(?![a-z0-9+#])"),
            re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, raw)) + r")(?!\w)") if raw else re.compile(r"(?!)"),
        )

    @lru_cache(maxsize=20000)
    def match(title):
        normalized = fold_text(title)
        original = unicodedata.normalize("NFKC", html.unescape(title)).casefold()
        groups, matched = [], []
        for group, (phrases, acronyms) in rules.items():
            hits = set(phrases.findall(normalized)) | set(acronyms.findall(original))
            if hits:
                groups.append(group)
                matched.extend(f"{group}: {term}" for term in sorted(hits))
        return normalized, " | ".join(groups), "; ".join(matched)

    output = Path(output_path).with_suffix(".parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    source = pq.ParquetFile(source_path)
    extra = ["title_normalized", "matched_groups", "matched_phrases"]
    if "job_title" not in source.schema_arrow.names or set(extra) & set(source.schema_arrow.names):
        raise ValueError("Schema raw không phù hợp cho lọc TinixAI")
    schema = source.schema_arrow.remove_metadata()
    for column in extra:
        schema = schema.append(pa.field(column, pa.string()))
    temp = output.with_name(output.name + ".tmp")
    counts = Counter()
    kept = 0
    with pq.ParquetWriter(temp, schema, compression="snappy") as writer:
        for batch in tqdm(source.iter_batches(batch_size=batch_size, use_threads=False), desc="Lọc TinixAI"):
            frame = batch.to_pandas(ignore_metadata=True)
            matches = pd.DataFrame([match(value) for value in frame.job_title.fillna("").astype(str)], columns=extra)
            mask = matches.matched_groups.ne("")
            selected = pd.concat([frame.loc[mask], matches.loc[mask]], axis=1)
            if not selected.empty:
                writer.write_table(pa.Table.from_pandas(selected, schema=schema, preserve_index=False))
                kept += len(selected)
                for groups in selected.matched_groups:
                    counts.update(groups.split(" | "))
    temp.replace(output)
    # CSV cũng ghi từng batch để không nạp toàn bộ JD vào RAM.
    csv = output.with_suffix(".csv")
    csv_temp = csv.with_name(csv.name + ".tmp")
    with csv_temp.open("w", encoding="utf-8-sig", newline="") as handle:
        pd.DataFrame(columns=schema.names).to_csv(handle, index=False)
        for batch in pq.ParquetFile(output).iter_batches(batch_size=batch_size):
            batch.to_pandas(ignore_metadata=True).to_csv(handle, index=False, header=False)
    csv_temp.replace(csv)
    provenance_path = Path(source_path).parent / "source.json"
    previous_metadata = output.with_suffix(".metadata.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else (
        json.loads(previous_metadata.read_text(encoding="utf-8")) if previous_metadata.exists() else {})
    write_json(output.with_suffix(".metadata.json"), {"dataset": "tinixai/vietnamese-job-descriptions",
               "revision": provenance.get("revision"), "license": "CC-BY-NC-4.0",
               "source_file": str(source_path), "source_rows": source.metadata.num_rows, "kept_rows": kept,
               "filter_column": "job_title", "group_counts": dict(counts), **config})
    return output
