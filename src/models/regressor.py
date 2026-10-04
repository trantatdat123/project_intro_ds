"""Dự đoán midpoint lương niêm yết; fit preprocessing và vocabulary chỉ trên train."""

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import SkillBinarizer


NUMERIC_FEATURES = ["experience_min", "year", "skill_count"]
CATEGORICAL_FEATURES = ["role_standard", "location_group", "seniority", "education_level", "job_position"]
INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["extracted_skills"]


def train_salary_regressor(X_train, y_train, model_name="lightgbm", *, random_state=42):
    models = {
        "random_forest": RandomForestRegressor(n_estimators=180, max_depth=18, min_samples_leaf=5,
                                               n_jobs=4, random_state=random_state),
        "ridge": Ridge(alpha=15),
    }
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(n_estimators=250, learning_rate=0.04, num_leaves=20,
                                  min_child_samples=35, reg_lambda=2, verbosity=-1, n_jobs=4,
                                  random_state=random_state)
    elif model_name in models:
        estimator = models[model_name]
    else:
        raise ValueError("model_name phải là lightgbm, random_forest hoặc ridge")
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                               ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                                   ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES),
        ("skills", SkillBinarizer(min_frequency=5), ["extracted_skills"]),
    ], sparse_threshold=0)
    model = TransformedTargetRegressor(
        regressor=Pipeline([("preprocess", preprocessing), ("model", estimator)]),
        func=np.log1p, inverse_func=np.expm1,
    )
    return model.fit(X_train[INPUT_FEATURES], y_train)


def evaluate_regressor(model, X_test, y_test) -> dict:
    actual = np.asarray(y_test, dtype=float)
    predicted = np.asarray(model.predict(X_test[INPUT_FEATURES]), dtype=float)
    return {"mae_vnd": float(mean_absolute_error(actual, predicted)),
            "rmse_vnd": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
            "median_absolute_error_vnd": float(np.median(np.abs(actual - predicted))),
            "rows": len(actual), "target": "advertised_salary_midpoint_vnd_month"}


def explain_salary_features(model, X_sample, feature_names=None, *, max_samples=200, random_state=42):
    """SHAP trên log1p(lương), mô tả liên hệ trong mô hình; không phải hiệu ứng nhân quả."""
    import pandas as pd
    import shap
    pipeline = model.regressor_
    estimator = pipeline.named_steps["model"]
    if not isinstance(estimator, RandomForestRegressor) and estimator.__class__.__name__ != "LGBMRegressor":
        raise ValueError("Tree SHAP chỉ hỗ trợ LightGBM/RandomForest trong module này")
    sample = X_sample[INPUT_FEATURES].sample(n=min(max_samples, len(X_sample)), random_state=random_state)
    encoded = pipeline.named_steps["preprocess"].transform(sample)
    names = feature_names if feature_names is not None else pipeline.named_steps["preprocess"].get_feature_names_out()
    values = np.asarray(shap.TreeExplainer(estimator).shap_values(encoded))
    summary = pd.DataFrame({"feature": names, "mean_abs_shap_log": np.abs(values).mean(axis=0),
                            "mean_shap_log": values.mean(axis=0)}).sort_values("mean_abs_shap_log", ascending=False)
    return {"summary": summary, "values": values, "encoded": encoded, "feature_names": list(names)}
