import json
import os

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import redirect, render

# Kết quả mô hình ML dự báo nguy cơ (Nguyễn Anh Quân) - sinh bởi src/ml/predict_risk.py
_GROUP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RISK_PREDICTIONS_CSV = os.path.join(_GROUP_DIR, "data", "processed", "ml", "student_risk_predictions.csv")
RISK_MODEL_REPORT = os.path.join(_GROUP_DIR, "src", "ml", "reports", "model_evaluation.json")


def home_redirect(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    return redirect("login")


@login_required
def dashboard(request):
    return render(request, "dashboard/dashboard.html", {"active_menu": "dashboard"})


@login_required
def learning(request):
    return render(request, "dashboard/empty_page.html", {
        "active_menu": "learning",
        "page_title": "Quản lý và phân tích học tập",
        "page_icon": "fa-solid fa-graduation-cap",
    })


@login_required
def quality(request):
    return render(request, "dashboard/empty_page.html", {
        "active_menu": "quality",
        "page_title": "Phân tích chất lượng đào tạo",
        "page_icon": "fa-solid fa-chart-pie",
    })


@login_required
def careers(request):
    return render(request, "dashboard/empty_page.html", {
        "active_menu": "careers",
        "page_title": "Phân tích xu hướng ngành nghề",
        "page_icon": "fa-solid fa-building-columns",
    })


@login_required
def skill_gap(request):
    return render(request, "dashboard/empty_page.html", {
        "active_menu": "skill_gap",
        "page_title": "Phân tích Skill Gap",
        "page_icon": "fa-solid fa-sliders",
    })


@login_required
def recommendations(request):
    return render(request, "dashboard/empty_page.html", {
        "active_menu": "recommendations",
        "page_title": "Đề xuất chương trình đào tạo",
        "page_icon": "fa-solid fa-lightbulb",
    })


@login_required
def api_risk_predictions(request):
    """API JSON: danh sách dự báo nguy cơ không hoàn thành khóa học.

    Tham số GET (tùy chọn): level=High|Medium|Low|Insufficient Data, module=AAA, limit=100
    """
    import pandas as pd

    if not os.path.exists(RISK_PREDICTIONS_CSV):
        return JsonResponse({"error": "Chưa có kết quả dự báo. Chạy src/ml/predict_risk.py."}, status=404)

    df = pd.read_csv(RISK_PREDICTIONS_CSV)
    level = request.GET.get("level")
    module = request.GET.get("module")
    if level:
        df = df[df["predicted_risk_level"] == level]
    if module:
        df = df[df["code_module"] == module]
    try:
        limit = max(1, min(int(request.GET.get("limit", 100)), 5000))
    except ValueError:
        limit = 100
    df = df.sort_values("risk_probability", ascending=False, na_position="last")

    summary = df["predicted_risk_level"].value_counts().to_dict()
    records = json.loads(df.head(limit).to_json(orient="records", force_ascii=False))
    return JsonResponse({"total": int(len(df)), "summary": summary, "results": records},
                        json_dumps_params={"ensure_ascii": False})


@login_required
def api_risk_model_report(request):
    """API JSON: báo cáo đánh giá mô hình (metrics, so sánh mô hình, early-warning curve)."""
    if not os.path.exists(RISK_MODEL_REPORT):
        return JsonResponse({"error": "Chưa có báo cáo. Chạy src/ml/train_risk_model.py."}, status=404)
    with open(RISK_MODEL_REPORT, encoding="utf-8") as f:
        return JsonResponse(json.load(f), json_dumps_params={"ensure_ascii": False})
