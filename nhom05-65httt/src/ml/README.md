# Module AI & Machine Learning – Dự báo nguy cơ không hoàn thành khóa học

Phụ trách: Nguyễn Anh Quân (2351160541). Tài liệu chi tiết: [docs/AI_ML_DU_BAO_NGUY_CO.md](../../docs/AI_ML_DU_BAO_NGUY_CO.md)

## Cấu trúc

| Tệp | Chức năng |
| :--- | :--- |
| `features.py` | Tạo đặc trưng từ `data/processed/oulad/` theo mốc ngày `cutoff_day` |
| `train_risk_model.py` | Huấn luyện, so sánh, chọn mô hình, đánh giá, lưu model và báo cáo |
| `predict_risk.py` | Chấm điểm toàn bộ học viên, xuất `data/processed/ml/student_risk_predictions.csv` |
| `models/` | `risk_model.joblib` + `model_metadata.json` |
| `reports/` | `model_evaluation.json`, `model_comparison.csv`, `early_warning_curve.csv`, `feature_importance.csv` |

## Chạy

Chạy từ thư mục `nhom05-65httt`:

```bash
pip install -r src/ml/requirements.txt
python src/ml/train_risk_model.py            # mặc định cutoff = ngày 90
python src/ml/predict_risk.py
python src/database/etl_warehouse.py         # nạp dự báo vào dw.fact_risk_prediction (cần PostgreSQL)
```

Tham số: `--cutoff 60` để huấn luyện mô hình cảnh báo sớm hơn; `--curve 30,60,90` để chọn các mốc đánh giá.

`risk_model.joblib` phụ thuộc phiên bản scikit-learn đã dùng khi huấn luyện (ghi trong `model_metadata.json`). Nếu khác phiên bản, hãy chạy lại `train_risk_model.py`.
