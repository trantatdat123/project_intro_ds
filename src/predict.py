"""CLI áp dụng model đã lưu cho CSV/Parquet có schema dữ liệu sạch."""
import argparse
from pathlib import Path

import joblib

from src.models.regressor import INPUT_FEATURES
from src.utils import ROOT, read_frame


def predict_jobs(input_path, output_path, models_dir=ROOT / "artifacts/models"):
    frame = read_frame(input_path)
    required = set(INPUT_FEATURES + ["text_clean"])
    if not required.issubset(frame.columns):
        raise ValueError(f"Cần dữ liệu đã prepare; thiếu {sorted(required - set(frame.columns))}")
    role = joblib.load(Path(models_dir) / "role_classifier.joblib")
    salary = joblib.load(Path(models_dir) / "salary_regressor.joblib")
    result = frame[[column for column in ("id", "title_clean", "role_standard") if column in frame]].copy()
    result["predicted_role_5_classes"] = role.predict(frame.text_clean.fillna(""))
    in_scope = frame.role_standard.isin(role.classes_)
    result["salary_prediction_in_scope"] = in_scope
    result["predicted_salary_vnd_month"] = float("nan")
    if in_scope.any():
        result.loc[in_scope, "predicted_salary_vnd_month"] = salary.predict(frame.loc[in_scope, INPUT_FEATURES])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return result


def main():
    parser = argparse.ArgumentParser(description="Dự đoán từ dữ liệu TinixAI đã làm sạch")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/inference_predictions.csv")
    args = parser.parse_args()
    result = predict_jobs(args.input, args.output)
    print(f"Đã xuất {len(result):,} dự đoán: {args.output}")


if __name__ == "__main__":
    main()
