"""
Huấn luyện & đánh giá mô hình dự báo nguy cơ học viên KHÔNG hoàn thành khóa học.
Phụ trách: Nguyễn Anh Quân - 2351160541 (AI & Machine Learning, Nhóm 05)

Bài toán: phân loại nhị phân
    y = 1 nếu final_result ∈ {Fail, Withdrawn}, y = 0 nếu ∈ {Pass, Distinction}
Đặc trưng: nhân khẩu học + hành vi VLE + bài đánh giá tính đến ngày `cutoff_day`.

Quy trình:
1. Chia train/test 80/20 (stratified, random_state cố định).
2. 5-fold Stratified CV trên tập train cho 3-4 mô hình (LogReg, Random Forest,
   HistGradientBoosting, XGBoost nếu đã cài) -> chọn mô hình có F1 OOF cao nhất,
   đồng thời chọn ngưỡng quyết định tối ưu F1 trên xác suất OOF.
3. Huấn luyện lại mô hình tốt nhất trên toàn tập train, đánh giá trên tập test.
4. Đánh giá thêm theo nhiều mốc thời gian (early-warning curve).
5. Lưu model (.joblib), metadata, báo cáo đánh giá (JSON/CSV).

Chạy:
    python src/ml/train_risk_model.py                 # cutoff mặc định 90 ngày
    python src/ml/train_risk_model.py --cutoff 60
"""

import argparse
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_table  # noqa: E402

ML_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ML_DIR, "models")
REPORT_DIR = os.path.join(ML_DIR, "reports")
SEED = 42
TARGET = "target_not_completed"
F1_TARGET = 0.75  # chỉ tiêu theo đề cương


def make_preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])


_XGB_WARNED = False


def candidate_models() -> dict:
    global _XGB_WARNED
    models = {
        "LogisticRegression": Pipeline([
            ("prep", make_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "RandomForest": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                           class_weight="balanced", n_jobs=-1, random_state=SEED)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                   class_weight="balanced", random_state=SEED)),
        ]),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4,
                                  subsample=0.9, colsample_bytree=0.9,
                                  eval_metric="logloss", random_state=SEED)),
        ])
    except ImportError:
        if not _XGB_WARNED:
            print("[i] Chưa cài xgboost -> bỏ qua XGBoost (pip install xgboost để thử thêm).")
            _XGB_WARNED = True
    return models


def best_threshold(y_true, proba) -> float:
    grid = np.arange(0.10, 0.91, 0.01)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(round(grid[int(np.argmax(scores))], 2))


def metrics(y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
    }


def rule_baseline(df: pd.DataFrame) -> np.ndarray:
    """Luật thủ công để so sánh: nộp < 50% bài đến hạn HOẶC điểm TB < 40."""
    low_submit = df["asm_submit_rate"] < 0.5
    low_score = df["asm_avg_score"].fillna(0) < 40
    return (low_submit | low_score).astype(float).values


def prepare(cutoff: int):
    df = build_feature_table(cutoff_day=cutoff)
    scope = df[df["has_behaviour_data"] == 1].reset_index(drop=True)
    return df, scope


def run_selection(scope: pd.DataFrame, verbose: bool = True):
    X = scope[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = scope[TARGET].values
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, scope.index.values, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, model in candidate_models().items():
        oof = cross_val_predict(clone(model), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr = best_threshold(y_tr, oof)
        cv_m = metrics(y_tr, oof, thr)
        final = clone(model).fit(X_tr, y_tr)
        te_m = metrics(y_te, final.predict_proba(X_te)[:, 1], thr)
        fitted[name] = (final, thr)
        rows.append({"model": name, "cv_f1": cv_m["f1"], "cv_roc_auc": cv_m["roc_auc"],
                     "cv_pr_auc": cv_m["pr_auc"], "threshold": thr,
                     "test_f1": te_m["f1"], "test_precision": te_m["precision"],
                     "test_recall": te_m["recall"], "test_roc_auc": te_m["roc_auc"],
                     "test_pr_auc": te_m["pr_auc"]})
        if verbose:
            print(f"  - {name:<22} CV F1={cv_m['f1']:.4f} AUC={cv_m['roc_auc']:.4f} | "
                  f"Test F1={te_m['f1']:.4f} AUC={te_m['roc_auc']:.4f} (thr={thr})")

    # Baseline luật thủ công
    base_pred = rule_baseline(scope.loc[idx_te])
    rows.append({"model": "RuleBaseline (submit<50% | score<40)",
                 "test_f1": round(f1_score(y_te, base_pred), 4),
                 "test_precision": round(precision_score(y_te, base_pred, zero_division=0), 4),
                 "test_recall": round(recall_score(y_te, base_pred), 4)})

    comparison = pd.DataFrame(rows)
    ml_rows = comparison[comparison["model"].isin(fitted.keys())]
    best_name = ml_rows.sort_values(["cv_f1", "cv_roc_auc"], ascending=False).iloc[0]["model"]
    return comparison, best_name, fitted, (X_tr, X_te, y_tr, y_te, idx_tr, idx_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=90,
                    help="Ngày mốc của khóa học dùng để dự báo sớm (mặc định 90).")
    ap.add_argument("--curve", default="30,60,90,120,150,240",
                    help="Các mốc ngày để đánh giá early-warning curve.")
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print(f"=== HUẤN LUYỆN MÔ HÌNH DỰ BÁO NGUY CƠ (cutoff = ngày {args.cutoff}) ===")
    df, scope = prepare(args.cutoff)
    print(f"[i] Tổng học viên: {len(df)} | Có dữ liệu hành vi (phạm vi mô hình): {len(scope)} "
          f"| Tỷ lệ không hoàn thành: {scope[TARGET].mean():.2%}")

    comparison, best_name, fitted, split = run_selection(scope)
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = split
    best_model, thr = fitted[best_name]
    proba_te = best_model.predict_proba(X_te)[:, 1]
    test_m = metrics(y_te, proba_te, thr)
    print(f"[*] Mô hình được chọn: {best_name} | Test F1={test_m['f1']} "
          f"({'ĐẠT' if test_m['f1'] >= F1_TARGET else 'CHƯA ĐẠT'} chỉ tiêu F1 >= {F1_TARGET})")

    # Recall theo từng loại kết quả trên tập test
    te_df = scope.loc[idx_te].assign(pred=(proba_te >= thr).astype(int))
    recall_by_result = (te_df.groupby("final_result")["pred"].mean().round(4).to_dict())

    # Tầm quan trọng đặc trưng (permutation importance trên tập test, theo ROC-AUC)
    pi = permutation_importance(best_model, X_te, y_te, scoring="roc_auc",
                                n_repeats=10, random_state=SEED, n_jobs=-1)
    importance = (pd.DataFrame({"feature": X_te.columns, "importance_mean": pi.importances_mean,
                                "importance_std": pi.importances_std})
                  .sort_values("importance_mean", ascending=False))
    importance.to_csv(os.path.join(REPORT_DIR, "feature_importance.csv"), index=False)

    # Early-warning curve: hiệu năng theo mốc thời gian
    curve = []
    for c in [int(x) for x in args.curve.split(",")]:
        _, sc = prepare(c)
        comp_c, best_c, fit_c, sp_c = run_selection(sc, verbose=False)
        m_c, t_c = fit_c[best_c]
        curve.append({"cutoff_day": c, "best_model": best_c,
                      **{k: v for k, v in metrics(sp_c[3], m_c.predict_proba(sp_c[1])[:, 1], t_c).items()
                         if k in ("f1", "precision", "recall", "roc_auc", "threshold")}})
        print(f"  [curve] ngày {c:>3}: {best_c:<22} F1={curve[-1]['f1']:.4f} AUC={curve[-1]['roc_auc']:.4f}")
    pd.DataFrame(curve).to_csv(os.path.join(REPORT_DIR, "early_warning_curve.csv"), index=False)

    # Huấn luyện lại trên toàn bộ dữ liệu trong phạm vi để triển khai
    X_all = scope[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    deploy_model = clone(best_model).fit(X_all, scope[TARGET].values)

    metadata = {
        "model_name": best_name,
        "model_version": datetime.now().strftime("%Y%m%d%H%M"),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "cutoff_day": args.cutoff,
        "decision_threshold": thr,
        "risk_bands": {"Low": "< 0.30", "Medium": "0.30 - < 0.60", "High": ">= 0.60"},
        "target": "1 = Fail/Withdrawn (không hoàn thành), 0 = Pass/Distinction",
        "scope": "Học viên có ít nhất 1 bản ghi VLE hoặc bài đánh giá",
        "n_scope": int(len(scope)),
        "n_total": int(len(df)),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "sklearn_version": sklearn.__version__,
        "random_state": SEED,
    }
    joblib.dump({"model": deploy_model, "metadata": metadata},
                os.path.join(MODEL_DIR, "risk_model.joblib"))
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    comparison.to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"), index=False)
    report = {
        "metadata": metadata,
        "selected_model": best_name,
        "test_metrics": test_m,
        "f1_target": F1_TARGET,
        "f1_target_met": bool(test_m["f1"] >= F1_TARGET),
        "test_recall_by_final_result": recall_by_result,
        "model_comparison": comparison.replace({np.nan: None}).to_dict(orient="records"),
        "early_warning_curve": curve,
        "top_features": importance.head(10).round(4).to_dict(orient="records"),
    }
    with open(os.path.join(REPORT_DIR, "model_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] Đã lưu model: {os.path.join(MODEL_DIR, 'risk_model.joblib')}")
    print(f"[OK] Đã lưu báo cáo đánh giá: {REPORT_DIR}")


if __name__ == "__main__":
    main()
