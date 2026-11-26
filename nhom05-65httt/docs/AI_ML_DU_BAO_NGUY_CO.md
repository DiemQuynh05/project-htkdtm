# Báo cáo AI & Machine Learning: Dự báo nguy cơ học viên không hoàn thành khóa học

- **Thành viên phụ trách:** Nguyễn Anh Quân (MSV: `2351160541`)
- **Vai trò:** AI & Machine Learning
- **Mã nguồn:** [src/ml/](../src/ml/)

---

## 1. Mục tiêu và chỉ tiêu

Theo đề cương (mục III.2.3 và IV.3):
- Xây dựng mô hình dự báo nguy cơ học viên không hoàn thành khóa học, sử dụng Random Forest hoặc XGBoost (nhóm so sánh thêm Logistic Regression và HistGradientBoosting).
- Phân nhóm nguy cơ **Thấp / Trung bình / Cao**.
- Chỉ tiêu nghiệm thu: **F1-score ≥ 0.75**.
- Tích hợp kết quả vào Data Warehouse, Superset và backend.

## 2. Định nghĩa bài toán

| Thành phần | Định nghĩa |
| :--- | :--- |
| Đơn vị dự báo | 1 học viên × 1 học phần × 1 đợt học |
| Nhãn `y = 1` | `final_result` ∈ {Fail, Withdrawn}, tức không hoàn thành |
| Nhãn `y = 0` | `final_result` ∈ {Pass, Distinction} |
| Thời điểm dự báo | Ngày **90** của khóa học (`cutoff_day`). Chỉ dùng log VLE và bài nộp có ngày ≤ 90 |
| Phạm vi | Học viên có ít nhất 1 bản ghi VLE hoặc bài đánh giá (1.500/3.000) |

**Chống rò rỉ nhãn:** cột `risk_level` trong `student_features_analytics.csv` được gán trực tiếp từ `final_result` (Withdrawn → High, Fail → Medium). Cột này **không** được dùng làm nhãn hay đặc trưng. Các cột tổng hợp toàn khóa (`total_clicks`, `avg_score`) cũng không dùng, vì chứa thông tin sau thời điểm dự báo.

## 3. Đặc trưng ([features.py](../src/ml/features.py))

| Nhóm | Đặc trưng |
| :--- | :--- |
| Nhân khẩu học | `code_module`, `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability`, `num_of_prev_attempts`, `studied_credits` |
| Tương tác VLE (≤ cutoff) | tổng click, số ngày hoạt động, số bản ghi, click/ngày, ngày hoạt động cuối, click theo 6 loại tài nguyên |
| Bài đánh giá (≤ cutoff) | số bài đến hạn, số bài đã nộp, tỷ lệ nộp, điểm TB, điểm thấp nhất, điểm có trọng số, số bài nộp trễ, độ trễ TB |

Tiền xử lý: điền giá trị thiếu bằng trung vị và thêm cột đánh dấu thiếu; mã hóa one-hot cho biến phân loại; chuẩn hóa thang đo (chỉ cho Logistic Regression).

## 4. Phương pháp huấn luyện và đánh giá ([train_risk_model.py](../src/ml/train_risk_model.py))

1. Chia train/test theo tỷ lệ 80/20, phân tầng theo nhãn, `random_state = 42`.
2. Chạy 5-fold Stratified Cross-Validation trên tập train cho từng mô hình. Xác suất out-of-fold được dùng để chọn **ngưỡng quyết định tối ưu F1**.
3. Chọn mô hình có F1 cross-validation cao nhất, rồi đánh giá một lần trên tập test.
4. Dùng một luật thủ công làm baseline: tỷ lệ nộp bài < 50% **hoặc** điểm TB < 40.
5. Đo permutation importance (theo ROC-AUC) trên tập test.
6. Vẽ early-warning curve: huấn luyện lại ở các mốc ngày 30 / 60 / 90 / 120 / 150 / 240.

## 5. Kết quả (cutoff = ngày 90)

### 5.1. So sánh mô hình

| Mô hình | F1 (CV) | ROC-AUC (CV) | Ngưỡng | F1 (test) | Precision (test) | Recall (test) | ROC-AUC (test) |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Logistic Regression | 0.9477 | 0.9884 | 0.56 | 0.9365 | 0.9291 | 0.9440 | 0.9791 |
| Random Forest | 0.9656 | 0.9942 | 0.71 | 0.9551 | 0.9750 | 0.9360 | 0.9928 |
| **HistGradientBoosting (chọn)** | **0.9770** | **0.9977** | **0.33** | **0.9799** | **0.9839** | **0.9760** | **0.9946** |
| Luật baseline | – | – | – | 0.7349 | 0.8778 | 0.6320 | – |

Ma trận nhầm lẫn trên tập test (n = 300): TN = 173, FP = 2, FN = 3, TP = 122.
Tỷ lệ phát hiện theo kết quả thực tế: Withdrawn 100%, Fail 96,1%. Tỷ lệ cảnh báo nhầm: Pass 1,5%, Distinction 0%.

**→ Đạt chỉ tiêu F1 ≥ 0.75.**

### 5.2. Early-warning curve

| Mốc ngày | Mô hình tốt nhất | F1 (test) | ROC-AUC (test) |
| :-: | :--- | :-: | :-: |
| 30 | Random Forest | 0.6642 | 0.8149 |
| 60 | Random Forest | 0.9091 | 0.9758 |
| 90 | HistGradientBoosting | 0.9799 | 0.9946 |
| 120 | HistGradientBoosting | 0.9960 | 0.9999 |
| 150 | HistGradientBoosting | 0.9880 | 0.9999 |
| 240 | HistGradientBoosting | 0.9921 | 1.0000 |

Ngày 90 là mốc cân bằng giữa dự báo sớm và độ chính xác. Ở ngày 30 (mới có 1 bài TMA), mô hình chưa đạt chỉ tiêu.

### 5.3. Đặc trưng quan trọng nhất

`asm_avg_score` (điểm TB) > `asm_min_score` > `asm_submitted` (số bài đã nộp) > `vle_clicks_per_day`. Các đặc trưng nhân khẩu học gần như không đóng góp.

## 6. Phân nhóm nguy cơ và kết quả dự báo ([predict_risk.py](../src/ml/predict_risk.py))

| Mức | Điều kiện |
| :--- | :--- |
| High | xác suất ≥ 0.60 |
| Medium | 0.30 ≤ xác suất < 0.60 |
| Low | xác suất < 0.30 |
| Insufficient Data | Học viên chưa có log VLE hoặc bài đánh giá; mô hình không dự báo |

Đầu ra là `data/processed/ml/student_risk_predictions.csv` (3.000 dòng): High 624, Low 876, Insufficient Data 1.500. Mức Medium không có học viên nào, vì xác suất mô hình trả về gần như chỉ nằm ở hai đầu 0 và 1.

## 7. Tích hợp hệ thống

| Thành phần | Nội dung |
| :--- | :--- |
| Data Warehouse | [dw_ml_risk_predictions.sql](../src/database/dw_ml_risk_predictions.sql): bảng `dw.fact_risk_prediction`, view `dw.v_bi_ml_risk_predictions` (danh sách cảnh báo kèm kết quả thực tế để đối chiếu) và view `dw.v_bi_ml_risk_summary` |
| ETL | [etl_warehouse.py](../src/database/etl_warehouse.py): hàm `load_risk_predictions_to_dw()` chạy ở cuối pipeline |
| Backend Django | `GET /api/ml/risk-predictions/?level=High&module=AAA&limit=100` và `GET /api/ml/risk-model-report/` (yêu cầu đăng nhập) |
| Superset | Có thể tạo dataset từ `dw.v_bi_ml_risk_predictions` và đưa vào Dashboard 1 (Quản lý học vụ) |

Thứ tự chạy:

```bash
python src/data_pipeline/run_pipeline.py     # dữ liệu sạch
python src/ml/train_risk_model.py            # huấn luyện + đánh giá
python src/ml/predict_risk.py                # dự báo
python src/database/etl_warehouse.py         # nạp DW + views
```

## 8. Hạn chế

1. **Dữ liệu OULAD hiện là dữ liệu mô phỏng** (`src/data_pipeline/collect_data.py`). Khi sinh dữ liệu, điểm số và số click đã được gán theo `final_result` (Pass/Distinction quanh 78 điểm, Fail/Withdrawn quanh 38 điểm; Withdrawn có 0–8 click). Vì vậy F1 ≈ 0.98 phản ánh cách sinh dữ liệu, **không** phản ánh hiệu năng trên dữ liệu thật. Trên OULAD gốc (Open University), mức hiệu năng thực tế thường thấp hơn đáng kể.
2. 1.500/3.000 học viên không có dữ liệu hành vi nên không được dự báo.
3. Bản ghi dự báo được sinh bằng mô hình đã huấn luyện trên chính các học viên đó (dự báo in-sample, dùng cho minh họa dashboard). Số liệu đánh giá khách quan là kết quả trên tập test ở mục 5.
4. Hướng tiếp theo: thay bằng OULAD gốc (tải từ analyse.kmi.open.ac.uk), chạy lại toàn bộ pipeline, hiệu chỉnh xác suất (calibration), và thử XGBoost (`pip install xgboost`; script tự động đưa vào so sánh).
