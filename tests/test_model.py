import json

import numpy as np
import pandas as pd

from core import config as C
from core import model, validation


def _clean(n=200):
    raw = pd.read_csv(C.SAMPLE_DIR / "sample_students.csv").head(n)
    return validation.validate_and_clean(raw).df


def test_target_leakage_columns_are_not_features():
    assert "Risk_Score" not in C.FEATURES and "At_Risk" not in C.FEATURES
    assert not {"Gender", "Age", "Stream"} & set(C.FEATURES)   # no sensitive attributes in the model


def test_make_target_bins():
    y = model.make_target(pd.Series([0, 29, 30, 49, 50, 100]))
    assert y.tolist() == ["Low concern", "Low concern", "Moderate concern", "Moderate concern",
                          "High concern", "High concern"]


def test_predictions_are_well_formed():
    bundle = model.load_bundle(auto_retrain=False)
    clean = _clean()
    out = model.predict(clean, bundle)
    assert len(out) == len(clean)
    assert set(out["Risk_Category"]) <= set(C.CATEGORIES)
    total = out[["Prob_Low", "Prob_Moderate", "Prob_High"]].sum(axis=1)
    assert np.allclose(total, 1.0, atol=0.01)
    assert out["Risk_Index"].between(0, 100).all()


def test_obvious_cases_are_classified_sensibly():
    bundle = model.load_bundle(auto_retrain=False)
    good = {c: 90.0 for c in C.FEATURES}; good["Previous_Semester_GPA"] = 9.0
    bad = {c: 40.0 for c in C.FEATURES}; bad["Previous_Semester_GPA"] = 3.5
    bad["Attendance_Percentage"] = 45.0; bad["Assignment_Completion_Percentage"] = 35.0
    out = model.predict(pd.DataFrame([good, bad]), bundle)
    assert out["Risk_Category"].tolist() == ["Low concern", "High concern"]


def test_metrics_file_documents_a_clean_train_test_split_and_beats_baseline():
    m = json.loads(C.METRICS_PATH.read_text())
    assert m["n_train"] + m["n_test"] == m["n_rows"]
    final = m["models"][m["final_model"]]
    base = m["models"]["Majority-class baseline"]
    assert final["macro_f1"] > base["macro_f1"] + 0.3
    assert final["per_class"]["High concern"]["recall"] > 0.9      # false negatives matter most
    assert "confusion_matrix" in final and len(final["confusion_matrix"]) == 3


def test_low_confidence_rows_are_flagged_for_review():
    bundle = model.load_bundle(auto_retrain=False)
    clean = _clean(50)
    clean.loc[0, "Data_Notes"] = "Imputed (typical value used): Math_Score"
    out = model.predict(clean, bundle)
    assert bool(out.loc[0, "Needs_Manual_Review"])
