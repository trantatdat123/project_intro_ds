"""Các hàm dùng chung cho văn bản, file và cấu hình."""

import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def fold_text(value) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", html.unescape(value)).casefold()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9+#]+", " ", text.replace("đ", "d")).strip()


def fingerprint(texts) -> str:
    digest = hashlib.sha256()
    for text in texts:
        value = str(text).encode("utf-8")
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")

    def convert(value):
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Không thể ghi JSON: {type(value).__name__}")

    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=convert, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_frame(path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    for column in ("extracted_skills", "role_candidates"):
        if column in frame:
            frame[column] = frame[column].map(
                lambda value: json.loads(value) if isinstance(value, str) else list(value) if value is not None else []
            )
    return frame


def save_frame(frame: pd.DataFrame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    csv = frame.copy()
    for column in ("extracted_skills", "role_candidates"):
        if column in csv:
            csv[column] = csv[column].map(lambda value: json.dumps(list(value), ensure_ascii=False))
    csv.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
