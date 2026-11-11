"""Reading and validating an uploaded student CSV.

The validator is forgiving where it safely can (column-name variants, '85%'
strings, a few missing values) and strict where it must be (missing required
columns, empty files).  Every automatic fix is reported back to the user.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------- #
# Column-name matching
# --------------------------------------------------------------------------- #
_ALIASES: dict[str, set[str]] = {
    "Student_ID": {"student_id", "studentid", "id", "student", "roll_no", "rollno",
                   "roll_number", "reg_no", "enrollment_no"},
    "Attendance_Percentage": {"attendance", "attendance_percentage", "attendance_pct",
                              "attendance_percent", "attendance_rate"},
    "Assignment_Completion_Percentage": {"assignment_completion", "assignment_completion_percentage",
                                         "assignments_completed", "assignment_pct",
                                         "assignment_completion_rate"},
    "Previous_Semester_GPA": {"previous_semester_gpa", "gpa", "prev_gpa", "previous_gpa",
                              "last_semester_gpa", "prev_semester_gpa"},
    "Previous_Assessment_Score": {"previous_assessment_score", "previous_assessment",
                                  "assessment_score", "prev_assessment", "prev_assessment_score"},
    "Math_Score": {"math_score", "math", "maths", "maths_score", "mathematics", "mathematics_score"},
    "Science_Score": {"science_score", "science"},
    "English_Score": {"english_score", "english"},
    "Social_Studies_Score": {"social_studies_score", "social_studies", "social_science", "social"},
    "Business_Studies_Score": {"business_studies_score", "business_studies", "business"},
    "Computer_Science_Score": {"computer_science_score", "computer_science", "computer",
                               "cs", "computer_studies"},
    "Stream": {"stream"},
    "Age": {"age"},
    "Gender": {"gender"},
}


def _norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


_LOOKUP: dict[str, str] = {}
for _canon, _names in _ALIASES.items():
    _LOOKUP[_norm(_canon)] = _canon
    for _n in _names:
        _LOOKUP[_n] = _canon


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    df: pd.DataFrame | None = None


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Parse raw bytes into a DataFrame, raising ValueError with a friendly message."""
    if not data or not data.strip():
        raise ValueError("The uploaded file is empty.")
    text = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("Could not decode the file. Please save it as a UTF-8 CSV.")
    try:
        sep = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep)
    except Exception as exc:  # noqa: BLE001 - pandas raises many exception types
        raise ValueError(f"The file could not be read as a CSV ({exc}).") from exc
    if df.shape[1] < 2:
        raise ValueError("The file has fewer than 2 columns - is it a valid, comma-separated CSV?")
    return df


# --------------------------------------------------------------------------- #
# Validation + cleaning
# --------------------------------------------------------------------------- #
def validate_and_clean(raw: pd.DataFrame, fill_values: dict[str, float] | None = None) -> ValidationResult:
    """Validate an uploaded table and return a cleaned copy ready for the model."""
    res = ValidationResult(ok=False)
    res.stats["rows_received"] = int(len(raw)) if raw is not None else 0

    if raw is None or raw.empty:
        res.errors.append("The file contains no data rows.")
        return res
    if len(raw) > C.MAX_ROWS:
        res.errors.append(f"The file has {len(raw):,} rows; the maximum supported is {C.MAX_ROWS:,}.")
        return res

    df = raw.copy()
    df = df.dropna(how="all")
    junk = [c for c in df.columns if str(c).startswith("Unnamed:") and df[c].isna().all()]
    df = df.drop(columns=junk)

    # ---- map column names to canonical names ---------------------------------
    rename: dict = {}
    taken: set[str] = set()
    for col in df.columns:
        canon = _LOOKUP.get(_norm(col))
        if canon is None:
            continue
        if canon in taken:
            res.warnings.append(f"Column '{col}' also matches {canon}; only the first match is used.")
            continue
        taken.add(canon)
        rename[col] = canon
        if str(col) != canon:
            res.info.append(f"Matched column '{col}' to {canon}.")
    df = df.rename(columns=rename)

    # ---- direct identifiers (names, e-mails ...) are removed ------------------
    ident = [c for c in df.columns
             if c not in C.REQUIRED_COLS and set(_norm(c).split("_")) & C.IDENTIFIER_TOKENS]
    if ident:
        df = df.drop(columns=ident)
        res.warnings.append("Possible direct identifiers were removed and will not be stored: "
                            + ", ".join(f"'{c}'" for c in ident) + ".")

    # ---- required columns -------------------------------------------------------
    missing = [c for c in C.REQUIRED_COLS if c not in df.columns]
    if missing:
        res.errors.append("Missing required column(s): " + ", ".join(missing) + ".")
        res.errors.append("Required columns: " + ", ".join(C.REQUIRED_COLS) + ".")
        return res

    # ---- prediction columns already present are replaced ------------------------
    clash = [c for c in C.PREDICTION_COLS if c in df.columns]
    if clash:
        df = df.drop(columns=clash)
        res.warnings.append("Existing prediction columns were replaced by new predictions: "
                            + ", ".join(clash) + ".")
    for c in ("Risk_Score", "At_Risk"):
        if c in df.columns:
            res.info.append(f"Column '{c}' was kept in the output but is NOT used for prediction.")

    # ---- Student_ID ----------------------------------------------------------------
    ids = df[C.ID_COL].where(df[C.ID_COL].notna(), "").astype(str).str.strip()
    ids = ids.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    blank = ids.eq("") | ids.str.lower().isin(["nan", "none", "null"])
    if blank.any():
        res.warnings.append(f"{int(blank.sum())} row(s) without a Student_ID were removed.")
    df[C.ID_COL] = ids
    df = df.loc[~blank]
    dup = df[C.ID_COL].duplicated(keep="first")
    if dup.any():
        res.warnings.append(f"{int(dup.sum())} duplicate Student_ID row(s) were removed (first occurrence kept).")
        df = df.loc[~dup]

    # ---- numeric conversion + range checks -------------------------------------------
    for col in C.FEATURES:
        s = df[col]
        had_value = s.notna() & s.astype(str).str.strip().ne("")
        if not pd.api.types.is_numeric_dtype(s):
            s = s.astype(str).str.replace("%", "", regex=False).str.strip()
        num = pd.to_numeric(s, errors="coerce").astype(float)
        bad_text = had_value & num.isna()
        if bad_text.any():
            res.warnings.append(f"{col}: {int(bad_text.sum())} non-numeric value(s) were treated as missing.")
        lo, hi = C.VALUE_RANGES[col]
        out = num.notna() & ((num < lo) | (num > hi))
        if out.any():
            res.warnings.append(f"{col}: {int(out.sum())} value(s) outside {lo}-{hi} were treated as missing.")
            num = num.mask(out)
        valid = num.dropna()
        if len(valid) > 1:
            if col in C.CORE_COLS and col != "Previous_Semester_GPA" and valid.max() <= 1.0:
                res.warnings.append(f"{col}: all values are between 0 and 1 - the model expects 0-100. "
                                    "Please check the scale.")
            if col == "Previous_Semester_GPA" and valid.max() <= 4.0:
                res.warnings.append("Previous_Semester_GPA: all values are <= 4.0 - the model expects a "
                                    "10-point GPA scale. Predictions may be unreliable on a 4-point scale.")
        df[col] = num

    # ---- missing values ------------------------------------------------------------------
    n_present = df[C.FEATURES].notna().sum(axis=1)
    too_empty = n_present < C.MIN_FEATURES_PRESENT
    if too_empty.any():
        res.warnings.append(f"{int(too_empty.sum())} row(s) had fewer than {C.MIN_FEATURES_PRESENT} usable "
                            "values and were removed.")
        df = df.loc[~too_empty]

    all_missing = [c for c in C.FEATURES if df[c].isna().all()]
    if all_missing and not fill_values:
        res.errors.append("These columns contain no usable values: " + ", ".join(all_missing) + ".")
        return res

    miss = df[C.FEATURES].isna()
    rows_with = miss.any(axis=1)
    df["Data_Notes"] = ""
    if rows_with.any():
        notes = miss.loc[rows_with].apply(
            lambda r: "Imputed (typical value used): " + ", ".join(c for c in C.FEATURES if r[c]), axis=1)
        df.loc[rows_with, "Data_Notes"] = notes
        res.warnings.append(f"{int(rows_with.sum())} row(s) had missing values that were filled with typical "
                            "values from the training data. Their predictions are less certain "
                            "(see the Data_Notes column).")
    fills = pd.Series(fill_values) if fill_values else df[C.FEATURES].median()
    df[C.FEATURES] = df[C.FEATURES].fillna(fills)

    df = df.reset_index(drop=True)
    if df.empty:
        res.errors.append("No valid rows remained after cleaning.")
        return res
    if len(df) < 5:
        res.warnings.append("Fewer than 5 students were found - cohort statistics will not be meaningful.")

    res.stats.update(rows_used=int(len(df)), rows_removed=int(res.stats["rows_received"] - len(df)),
                     columns_used=len(C.REQUIRED_COLS))
    res.df = df
    res.ok = True
    return res
