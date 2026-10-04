"""Multi-hot kỹ năng và sentence embeddings với cache kiểm tra nội dung."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils import fingerprint, write_json


class SkillBinarizer(TransformerMixin, BaseEstimator):
    def __init__(self, min_frequency=5):
        self.min_frequency = min_frequency

    def fit(self, X, y=None):
        values = self._values(X)
        counts = Counter(skill for skills in values for skill in set(skills))
        self.skills_ = sorted(skill for skill, count in counts.items() if count >= self.min_frequency)
        return self

    @staticmethod
    def _values(X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].tolist()
        if isinstance(X, pd.Series):
            return X.tolist()
        return [value[0] if isinstance(value, np.ndarray) else value for value in X]

    def transform(self, X):
        index = {skill: i for i, skill in enumerate(self.skills_)}
        values = self._values(X)
        result = np.zeros((len(values), len(index)), dtype=np.float32)
        for row, skills in enumerate(values):
            for skill in set(skills):
                if skill in index:
                    result[row, index[skill]] = 1
        return result

    def get_feature_names_out(self, input_features=None):
        return np.array([f"skill_{skill}" for skill in self.skills_], dtype=object)


def create_skill_matrix(df, skill_col="extracted_skills", min_frequency=10):
    if min_frequency < 1:
        raise ValueError("min_frequency phải >= 1")
    encoder = SkillBinarizer(min_frequency).fit(df[[skill_col]])
    return pd.DataFrame(encoder.transform(df[[skill_col]]), columns=encoder.get_feature_names_out(), index=df.index)


def generate_text_embeddings(texts, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                             *, batch_size=16, max_seq_length=256, cache_path=None, model_cache=None, device=None,
                             revision=None, local_files_only=False):
    """Mean-pooled sentence embedding; mô hình cắt token sau max_seq_length."""
    texts = [str(text) if text is not None else "" for text in texts]
    signature = {"texts_sha256": fingerprint(texts), "model": model_name,
                 "revision": revision, "max_seq_length": max_seq_length, "rows": len(texts)}
    cache = Path(cache_path) if cache_path else None
    metadata = cache.with_suffix(".json") if cache else None
    if cache and cache.exists() and metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if all(saved.get(key) == value for key, value in signature.items()):
            values = np.load(cache, allow_pickle=False)
            if values.ndim == 2 and len(values) == len(texts):
                return values
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(min(4, torch.get_num_threads()))
    # oneDNN lưu primitive theo nhiều độ dài token, có thể tăng RAM trên CPU.
    # Dense float32 vẫn được giữ; tắt cache backend này cho lượt CPU dài.
    if device == "cpu" or (device is None and not torch.cuda.is_available()):
        torch.backends.mkldnn.enabled = False
    model = SentenceTransformer(model_name, cache_folder=str(model_cache) if model_cache else None,
                                device=device, revision=revision, local_files_only=local_files_only)
    model.max_seq_length = max_seq_length
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    if cache:
        from tqdm import tqdm
        cache.parent.mkdir(parents=True, exist_ok=True)
        partial = cache.with_name(cache.stem + ".partial.npy")
        progress_path = cache.with_name(cache.stem + ".progress.json")
        dimension = model.get_sentence_embedding_dimension()
        completed = 0
        if partial.exists() and progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if all(progress.get(key) == value for key, value in signature.items()):
                completed = int(progress.get("completed_rows", 0))
        values = np.lib.format.open_memmap(partial, mode="r+" if completed else "w+",
                                          dtype=np.float32, shape=(len(texts), dimension))
        chunk_size = max(batch_size, 128)
        with tqdm(total=len(texts), initial=completed, desc="Embeddings (checkpoint)") as bar:
            for start in range(completed, len(texts), chunk_size):
                stop = min(start + chunk_size, len(texts))
                values[start:stop] = model.encode(texts[start:stop], batch_size=batch_size, show_progress_bar=False,
                                                   normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
                values.flush()
                write_json(progress_path, {**signature, "completed_rows": stop, "dimension": dimension})
                bar.update(stop - start)
                import gc
                gc.collect()
        del values
        partial.replace(cache)
        write_json(metadata, {**signature, "dimension": dimension})
        progress_path.unlink(missing_ok=True)
        values = np.load(cache, allow_pickle=False)
    else:
        values = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
    return values
