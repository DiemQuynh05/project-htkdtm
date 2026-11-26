"""
Feature engineering cho mô hình dự báo nguy cơ không hoàn thành khóa học.
Phụ trách: Nguyễn Anh Quân - 2351160541 (AI & Machine Learning, Nhóm 05)

Nguyên tắc chống rò rỉ nhãn (data leakage):
- KHÔNG dùng `final_result`, `risk_level` (risk_level trong student_features_analytics.csv
  chỉ là ánh xạ trực tiếp từ final_result) làm đặc trưng.
- Chỉ dùng log VLE và bài đánh giá có ngày <= `cutoff_day` (cảnh báo sớm giữa khóa).
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OULAD_DIR = os.path.join(BASE_DIR, "data", "processed", "oulad")

KEY_COLS = ["code_module", "code_presentation", "id_student"]
CATEGORICAL_FEATURES = [
    "code_module", "gender", "region", "highest_education",
    "imd_band", "age_band", "disability",
]
NUMERIC_FEATURES = [
    "num_of_prev_attempts", "studied_credits",
    # VLE
    "vle_total_clicks", "vle_active_days", "vle_n_records", "vle_clicks_per_day",
    "vle_last_active_day", "vle_clicks_forumng", "vle_clicks_oucontent", "vle_clicks_quiz",
    "vle_clicks_resource", "vle_clicks_url", "vle_clicks_subpage",
    # Assessment
    "asm_due", "asm_submitted", "asm_submit_rate", "asm_avg_score", "asm_min_score",
    "asm_weighted_score", "asm_late_count", "asm_avg_delay_days",
]
ACTIVITY_TYPES = ["forumng", "oucontent", "quiz", "resource", "url", "subpage"]
NON_COMPLETION = {"Fail", "Withdrawn"}


def load_raw_tables(oulad_dir: str = OULAD_DIR):
    return {
        "info": pd.read_csv(os.path.join(oulad_dir, "studentInfo_cleaned.csv")),
        "vle": pd.read_csv(os.path.join(oulad_dir, "studentVle_cleaned.csv")),
        "stu_asm": pd.read_csv(os.path.join(oulad_dir, "studentAssessment_cleaned.csv")),
        "asm": pd.read_csv(os.path.join(oulad_dir, "assessments_cleaned.csv")),
    }


def _vle_features(vle: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    v = vle[vle["date"] <= cutoff_day]
    g = v.groupby(KEY_COLS)
    feats = g.agg(
        vle_total_clicks=("sum_click", "sum"),
        vle_active_days=("date", "nunique"),
        vle_n_records=("sum_click", "size"),
        vle_last_active_day=("date", "max"),
    )
    feats["vle_clicks_per_day"] = feats["vle_total_clicks"] / feats["vle_active_days"].clip(lower=1)
    by_type = v.pivot_table(index=KEY_COLS, columns="activity_type", values="sum_click",
                            aggfunc="sum", fill_value=0)
    for t in ACTIVITY_TYPES:
        feats[f"vle_clicks_{t}"] = by_type[t] if t in by_type.columns else 0
    return feats.reset_index()


def _assessment_features(info: pd.DataFrame, stu_asm: pd.DataFrame, asm: pd.DataFrame,
                         cutoff_day: int) -> pd.DataFrame:
    # Bài đánh giá đã đến hạn trước mốc cutoff (mỗi học phần)
    due = asm[asm["date"] <= cutoff_day]
    due_count = due.groupby(["code_module", "code_presentation"]).size().rename("asm_due").reset_index()

    sa = stu_asm.merge(asm, on="id_assessment", how="inner")
    sa = sa[(sa["date_submitted"] <= cutoff_day)].copy()
    sa["delay"] = sa["date_submitted"] - sa["date"]
    sa["is_late"] = (sa["delay"] > 0).astype(int)
    sa["w_score"] = sa["score"] * sa["weight"]

    g = sa.groupby(KEY_COLS)
    feats = g.agg(
        asm_submitted=("id_assessment", "nunique"),
        asm_avg_score=("score", "mean"),
        asm_min_score=("score", "min"),
        asm_late_count=("is_late", "sum"),
        asm_avg_delay_days=("delay", "mean"),
        _w_sum=("w_score", "sum"),
        _weight_sum=("weight", "sum"),
    )
    feats["asm_weighted_score"] = feats["_w_sum"] / feats["_weight_sum"].replace(0, np.nan)
    feats = feats.drop(columns=["_w_sum", "_weight_sum"]).reset_index()

    out = info[KEY_COLS].merge(due_count, on=["code_module", "code_presentation"], how="left")
    out = out.merge(feats, on=KEY_COLS, how="left")
    out["asm_due"] = out["asm_due"].fillna(0)
    out["asm_submitted"] = out["asm_submitted"].fillna(0)
    out["asm_submit_rate"] = np.where(out["asm_due"] > 0, out["asm_submitted"] / out["asm_due"], 0.0)
    return out


def build_feature_table(cutoff_day: int = 90, oulad_dir: str = OULAD_DIR) -> pd.DataFrame:
    """Trả về bảng 1 dòng / học viên-học phần gồm đặc trưng, nhãn và cờ có dữ liệu hành vi."""
    t = load_raw_tables(oulad_dir)
    info = t["info"]

    vle_f = _vle_features(t["vle"], cutoff_day)
    asm_f = _assessment_features(info, t["stu_asm"], t["asm"], cutoff_day)

    df = info.merge(vle_f, on=KEY_COLS, how="left").merge(asm_f, on=KEY_COLS, how="left")

    # Học viên có bất kỳ log hành vi nào trong toàn khóa (dùng để xác định phạm vi mô hình)
    has_vle = info[KEY_COLS].merge(t["vle"][KEY_COLS].drop_duplicates(), on=KEY_COLS, how="inner")
    asm_keys = t["stu_asm"].merge(t["asm"], on="id_assessment")[KEY_COLS].drop_duplicates()
    behaviour_keys = pd.concat([has_vle, asm_keys]).drop_duplicates()
    behaviour_keys["has_behaviour_data"] = 1
    df = df.merge(behaviour_keys, on=KEY_COLS, how="left")
    df["has_behaviour_data"] = df["has_behaviour_data"].fillna(0).astype(int)

    count_cols = ["vle_total_clicks", "vle_active_days", "vle_n_records", "vle_clicks_per_day",
                  "asm_late_count"] + [f"vle_clicks_{a}" for a in ACTIVITY_TYPES]
    df[count_cols] = df[count_cols].fillna(0)
    # last_active_day / điểm / độ trễ: để NaN -> imputer xử lý trong pipeline

    df["target_not_completed"] = df["final_result"].isin(NON_COMPLETION).astype(int)
    df["cutoff_day"] = cutoff_day
    return df
