import tempfile
from pathlib import Path

import pandas as pd

from core import analysis, model, pipeline
from core import config as C

SAMPLE = C.SAMPLE_DIR / "sample_students.csv"


def _results():
    bundle = model.load_bundle(auto_retrain=False)
    val, res = pipeline.run(SAMPLE.read_bytes(), bundle)
    assert val.ok
    return res


def test_pipeline_adds_all_prediction_columns():
    res = _results()
    for col in C.PREDICTION_COLS:
        assert col in res.columns, col
    assert len(res) == 300


def test_saved_csv_is_separate_and_uploaded_file_untouched():
    before = SAMPLE.read_bytes()
    res = _results()
    with tempfile.TemporaryDirectory() as tmp:
        path = pipeline.save_predictions(res, "my students.csv", Path(tmp))
        assert path.exists() and path != SAMPLE
        back = pipeline.load_saved(path)
        assert len(back) == len(res) and "Risk_Category" in back.columns
        assert (Path(tmp) / "latest_predictions.csv").exists()
    assert SAMPLE.read_bytes() == before


def test_attention_areas_follow_benchmarks():
    row = {c: 85.0 for c in C.FEATURES}; row["Previous_Semester_GPA"] = 8.5
    row["Attendance_Percentage"] = 50.0
    df = pd.DataFrame([{C.ID_COL: "X1", **row, "Risk_Category": "High concern"}])
    out = analysis.enrich(df)
    assert "Attendance" in out.loc[0, "Attention_Areas"]
    assert out.loc[0, "Support_Priority"] == "Priority" and out.loc[0, "Suggested_Follow_Up_Days"] == 7


def test_student_report_has_subject_wise_findings_and_recommendations():
    res = _results()
    weak = res[res["Risk_Category"] == "High concern"].iloc[0]
    rep = analysis.student_report(weak, res)
    assert len(rep["subjects"]) == 6 and len(rep["indicators"]) == 5
    assert rep["attention"], "a high-concern student should have at least one attention area"
    assert rep["recommendations"] and all(r["actions"] for r in rep["recommendations"])
    md = analysis.report_to_markdown(rep)
    assert rep["id"] in md and "Subject-wise performance" in md and "not a grade" in md


def test_subject_status_bands():
    res = _results()
    row = res.iloc[0].copy()
    row["Math_Score"], row["English_Score"], row["Science_Score"] = 45.0, 55.0, 90.0
    rep = analysis.student_report(row, res)
    status = {s["name"]: s["status"] for s in rep["subjects"]}
    assert status["Mathematics"] == "Critical" and status["English"] == "Needs attention"
    assert status["Science"] == "On track"


def test_cohort_summaries_are_consistent():
    res = _results()
    counts = analysis.category_counts(res)
    assert counts["Students"].sum() == len(res)
    assert abs(counts["Share"].sum() - 100) < 0.5
    assert analysis.key_findings(res)
    ranked = analysis.rank_for_review(res)
    assert ranked["Risk_Index"].is_monotonic_decreasing
