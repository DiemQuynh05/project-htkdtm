"""
Sinh dự báo nguy cơ không hoàn thành khóa học cho toàn bộ học viên (batch scoring).
Phụ trách: Nguyễn Anh Quân - 2351160541 (AI & Machine Learning, Nhóm 05)

Đầu ra: data/processed/ml/student_risk_predictions.csv
    - risk_probability : xác suất không hoàn thành (0-1)
    - predicted_label  : 1 nếu risk_probability >= ngưỡng quyết định của model
    - predicted_risk_level : High (>=0.60) / Medium (0.30-0.60) / Low (<0.30)
                             / Insufficient Data (học viên chưa có log hành vi)
Tệp này được ETL Data Warehouse nạp vào dw.fact_risk_prediction (xem
src/database/dw_ml_risk_predictions.sql và etl_warehouse.py).

Chạy:
    python src/ml/predict_risk.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import BASE_DIR, build_feature_table  # noqa: E402

ML_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ML_DIR, "models", "risk_model.joblib")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "ml")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "student_risk_predictions.csv")


def risk_band(p: float) -> str:
    if p >= 0.60:
        return "High"
    if p >= 0.30:
        return "Medium"
    return "Low"


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chưa có model tại {path}. Chạy train_risk_model.py trước.")
    bundle = joblib.load(path)
    return bundle["model"], bundle["metadata"]


def predict(df: pd.DataFrame, model, meta: dict) -> pd.DataFrame:
    feats = meta["numeric_features"] + meta["categorical_features"]
    out = df[["code_module", "code_presentation", "id_student", "has_behaviour_data",
              "vle_total_clicks", "vle_active_days", "asm_submitted", "asm_due",
              "asm_avg_score"]].copy()
    proba = model.predict_proba(df[feats])[:, 1]
    in_scope = df["has_behaviour_data"].values == 1
    out["risk_probability"] = np.where(in_scope, np.round(proba, 4), np.nan)
    out["predicted_label"] = np.where(in_scope, (proba >= meta["decision_threshold"]).astype(int), -1)
    out["predicted_risk_level"] = [risk_band(p) if s else "Insufficient Data"
                                   for p, s in zip(proba, in_scope)]
    out["cutoff_day"] = meta["cutoff_day"]
    out["model_name"] = meta["model_name"]
    out["model_version"] = meta["model_version"]
    return out


def main():
    model, meta = load_model()
    df = build_feature_table(cutoff_day=meta["cutoff_day"])
    out = predict(df, model, meta)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Đã dự báo {len(out)} học viên -> {OUTPUT_CSV}")
    print(out["predicted_risk_level"].value_counts().to_string())


if __name__ == "__main__":
    main()
