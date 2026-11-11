import io

import pandas as pd
import pytest

from core import config as C
from core import validation as V


def _sample(n=30) -> pd.DataFrame:
    return pd.read_csv(C.SAMPLE_DIR / "sample_students.csv").head(n)


def test_accepts_valid_sample():
    res = V.validate_and_clean(_sample())
    assert res.ok and not res.errors
    assert res.stats["rows_used"] == 30
    assert res.df[C.FEATURES].notna().all().all()


def test_missing_required_column_is_an_error():
    res = V.validate_and_clean(_sample().drop(columns=["Attendance_Percentage"]))
    assert not res.ok
    assert any("Attendance_Percentage" in e for e in res.errors)
    assert res.df is None


def test_flexible_column_names_are_matched():
    df = _sample().rename(columns={"Attendance_Percentage": "Attendance %", "Math_Score": "Maths",
                                   "Previous_Semester_GPA": "GPA"})
    res = V.validate_and_clean(df)
    assert res.ok
    assert {"Attendance_Percentage", "Math_Score", "Previous_Semester_GPA"} <= set(res.df.columns)


def test_bad_values_become_missing_and_are_imputed_and_noted():
    df = _sample().astype({"Math_Score": object})
    df.loc[df.index[0], "Attendance_Percentage"] = 150      # out of range
    df.loc[df.index[1], "Math_Score"] = "absent"            # not a number
    df.loc[df.index[2], "Previous_Semester_GPA"] = None     # missing
    res = V.validate_and_clean(df, fill_values={c: 50.0 for c in C.FEATURES})
    assert res.ok
    assert res.df[C.FEATURES].notna().all().all()
    assert (res.df["Data_Notes"] != "").sum() == 3
    assert any("outside" in w for w in res.warnings) and any("non-numeric" in w for w in res.warnings)


def test_percent_strings_are_parsed():
    df = _sample().astype({"Attendance_Percentage": object})
    df.loc[df.index[0], "Attendance_Percentage"] = "85%"
    res = V.validate_and_clean(df)
    assert res.ok and res.df.loc[0, "Attendance_Percentage"] == 85.0


def test_duplicate_and_blank_ids_removed():
    df = pd.concat([_sample(), _sample(2)])
    df.loc[df.index[3], C.ID_COL] = ""
    res = V.validate_and_clean(df)
    assert res.ok
    assert res.df[C.ID_COL].is_unique
    assert res.stats["rows_removed"] >= 3


def test_identifier_columns_are_removed():
    df = _sample()
    df["Student Name"] = "x"
    df["Email"] = "a@b.c"
    res = V.validate_and_clean(df)
    assert res.ok
    assert "Student Name" not in res.df.columns and "Email" not in res.df.columns
    assert any("identifiers" in w for w in res.warnings)


def test_existing_prediction_columns_are_replaced():
    df = _sample()
    df["Risk_Category"] = "whatever"
    res = V.validate_and_clean(df)
    assert res.ok and "Risk_Category" not in res.df.columns


def test_four_point_gpa_scale_warns():
    df = _sample()
    df["Previous_Semester_GPA"] = df["Previous_Semester_GPA"] / 3
    res = V.validate_and_clean(df)
    assert any("4.0" in w for w in res.warnings)


def test_empty_and_garbage_files():
    with pytest.raises(ValueError):
        V.read_csv_bytes(b"")
    with pytest.raises(ValueError):
        V.read_csv_bytes(b"onecolumn\n1\n2\n")
    assert not V.validate_and_clean(pd.DataFrame()).ok


def test_read_csv_with_semicolons_and_latin1():
    text = "Student_ID;Café;x\nA;1;2\n".encode("latin-1")
    df = V.read_csv_bytes(text)
    assert list(df.columns)[0] == "Student_ID" and len(df) == 1
