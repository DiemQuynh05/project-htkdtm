-- =====================================================================
-- Tích hợp kết quả AI/ML vào Data Warehouse
-- Phụ trách: Nguyễn Anh Quân - 2351160541 (AI & Machine Learning, Nhóm 05)
-- Nguồn: data/processed/ml/student_risk_predictions.csv (src/ml/predict_risk.py)
-- Bảng được nạp bởi etl_warehouse.py -> load_risk_predictions_to_dw()
-- =====================================================================

-- Fact_RiskPrediction: Kết quả dự báo nguy cơ không hoàn thành khóa học
-- Grain: 1 học viên x 1 học phần x 1 phiên bản mô hình
-- Tạo lại mỗi lần ETL (giống các bảng dw.* khác, vì dw_star_schema.sql DROP ... CASCADE các dimension)
DROP TABLE IF EXISTS dw.fact_risk_prediction CASCADE;
CREATE TABLE dw.fact_risk_prediction (
    prediction_id BIGSERIAL PRIMARY KEY,
    student_sk INT NOT NULL REFERENCES dw.dim_student(student_sk),
    course_sk INT NOT NULL REFERENCES dw.dim_course(course_sk),
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    cutoff_day INT NOT NULL,                     -- ngày mốc dữ liệu dùng để dự báo
    risk_probability NUMERIC(5, 4),              -- NULL nếu thiếu dữ liệu hành vi
    predicted_label SMALLINT NOT NULL,           -- 1 = dự báo không hoàn thành, 0 = hoàn thành, -1 = không đủ dữ liệu
    predicted_risk_level VARCHAR(20) NOT NULL,   -- High / Medium / Low / Insufficient Data
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (student_sk, course_sk, model_version)
);

CREATE INDEX idx_dw_riskpred_level ON dw.fact_risk_prediction(predicted_risk_level);
CREATE INDEX idx_dw_riskpred_course ON dw.fact_risk_prediction(course_sk);

-- View cho Superset / Django: danh sách cảnh báo sớm theo mô hình ML (phiên bản mới nhất)
CREATE OR REPLACE VIEW dw.v_bi_ml_risk_predictions AS
WITH latest AS (
    SELECT MAX(model_version) AS model_version FROM dw.fact_risk_prediction
)
SELECT
    ds.id_student,
    dc.code_module,
    dc.code_presentation,
    ds.gender,
    ds.age_band,
    ds.region,
    ds.highest_education,
    p.model_name,
    p.model_version,
    p.cutoff_day,
    p.risk_probability,
    p.predicted_label,
    p.predicted_risk_level,
    f.total_clicks,
    f.active_days,
    f.assessments_completed,
    f.avg_score,
    f.final_result,                                -- kết quả thực tế (để đối chiếu)
    CASE
        WHEN p.predicted_label = -1 THEN NULL
        WHEN p.predicted_label = f.is_dropout THEN 1 ELSE 0
    END AS is_prediction_correct
FROM dw.fact_risk_prediction p
JOIN latest l ON l.model_version = p.model_version
JOIN dw.dim_student ds ON ds.student_sk = p.student_sk
JOIN dw.dim_course dc ON dc.course_sk = p.course_sk
LEFT JOIN dw.fact_ket_qua_hoc_tap f
       ON f.student_sk = p.student_sk AND f.course_sk = p.course_sk
ORDER BY
    CASE p.predicted_risk_level WHEN 'High' THEN 1 WHEN 'Medium' THEN 2
                                WHEN 'Low' THEN 3 ELSE 4 END,
    p.risk_probability DESC NULLS LAST;

-- View tổng hợp phân bố mức nguy cơ theo học phần
CREATE OR REPLACE VIEW dw.v_bi_ml_risk_summary AS
SELECT
    code_module,
    code_presentation,
    predicted_risk_level,
    COUNT(*) AS student_count,
    ROUND(AVG(risk_probability), 4) AS avg_risk_probability
FROM dw.v_bi_ml_risk_predictions
GROUP BY code_module, code_presentation, predicted_risk_level;
