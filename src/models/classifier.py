"""Phân loại JD thành nhãn nghề sơ bộ: TF-IDF + Logistic/SVM/MLP."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


class EncodedLabelClassifier(ClassifierMixin, BaseEstimator):
    """Để MLP early stopping làm việc với nhãn số, trả lại tên nghề ở API ngoài."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.estimator_ = clone(self.estimator).fit(X, self.encoder_.transform(y))
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(self.estimator_.predict(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def train_role_classifier(X_train, y_train, model_type="tfidf_logistic", *, random_state=42):
    if len(set(y_train)) < 2:
        raise ValueError("Cần ít nhất hai nhóm nghề trong tập train")
    models = {
        "tfidf_logistic": LogisticRegression(C=2, max_iter=1000, class_weight="balanced", random_state=random_state),
        "tfidf_svm": LinearSVC(C=1, class_weight="balanced", random_state=random_state),
        "tfidf_mlp": EncodedLabelClassifier(MLPClassifier(hidden_layer_sizes=(64,), max_iter=80, batch_size=128,
                                   early_stopping=True, n_iter_no_change=8, random_state=random_state)),
    }
    if model_type not in models:
        raise ValueError(f"model_type phải thuộc {list(models)}")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2 if len(X_train) >= 20 else 1,
                                  max_df=0.98, max_features=20000, sublinear_tf=True, dtype=np.float32)),
        ("model", models[model_type]),
    ])
    return model.fit(X_train, y_train)


def evaluate_classifier(model, X_test, y_test) -> dict:
    predicted = model.predict(X_test)
    labels = sorted(set(y_test) | set(predicted))
    return {
        "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_test, predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predicted, average="macro", zero_division=0)),
        "classification_report": classification_report(y_test, predicted, output_dict=True, zero_division=0),
        "labels": labels, "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
        "rows": len(y_test), "target_source": "title_rule",
    }
