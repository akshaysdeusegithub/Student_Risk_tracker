"""Central configuration: paths, column names, benchmarks and wording.

Everything that a teacher/maintainer might want to tune lives here, so the
rest of the code never contains "magic numbers".
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "risk_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
REPORT_DIR = ROOT / "reports"
TRAINING_DATA_PATH = ROOT / "data" / "training" / "students_labeled.csv"
OUTPUT_DIR = ROOT / "outputs" / "predictions"
SAMPLE_DIR = ROOT / "sample_data"

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
ID_COL = "Student_ID"

# subject column -> display name
SUBJECTS: dict[str, str] = {
    "Math_Score": "Mathematics",
    "Science_Score": "Science",
    "English_Score": "English",
    "Social_Studies_Score": "Social Studies",
    "Business_Studies_Score": "Business Studies",
    "Computer_Science_Score": "Computer Science",
}
SUBJECT_COLS = list(SUBJECTS)

CORE_COLS = [
    "Attendance_Percentage",
    "Assignment_Completion_Percentage",
    "Previous_Semester_GPA",
    "Previous_Assessment_Score",
]

# The model only sees these 10 numeric columns. Gender, age, stream and other
# personal attributes are deliberately NOT used for prediction.
FEATURES = CORE_COLS + SUBJECT_COLS
REQUIRED_COLS = [ID_COL] + FEATURES

# Optional columns that are passed through to the output but never used by the model.
OPTIONAL_PASSTHROUGH = ["Stream", "Age", "Gender"]

# valid ranges used by the validator (inclusive)
VALUE_RANGES: dict[str, tuple[float, float]] = {
    "Attendance_Percentage": (0, 100),
    "Assignment_Completion_Percentage": (0, 100),
    "Previous_Semester_GPA": (0, 10),
    "Previous_Assessment_Score": (0, 100),
    **{c: (0, 100) for c in SUBJECT_COLS},
}

MAX_ROWS = 200_000
MIN_FEATURES_PRESENT = 5  # rows with fewer usable features than this are dropped

# --------------------------------------------------------------------------- #
# Risk categories (derived from the dataset's documented Risk_Score, 0-100)
#   Risk_Score <  30  -> Low concern
#   30 <= score < 50  -> Moderate concern
#   score >= 50       -> High concern   (identical to the dataset's At_Risk = 1)
# --------------------------------------------------------------------------- #
CATEGORIES = ["Low concern", "Moderate concern", "High concern"]
LOW_MAX = 30
HIGH_MIN = 50

CATEGORY_COLORS = {
    "Low concern": "#2e9e6b",
    "Moderate concern": "#f0a020",
    "High concern": "#d64545",
}

PROB_COLS = {
    "Low concern": "Prob_Low",
    "Moderate concern": "Prob_Moderate",
    "High concern": "Prob_High",
}
LOW_CONFIDENCE = 0.60  # below this, the record is flagged for manual review

# support priority per category -> (label, suggested follow-up in days)
SUPPORT = {
    "Low concern": ("Routine", 30),
    "Moderate concern": ("Targeted", 14),
    "High concern": ("Priority", 7),
}

PREDICTION_COLS = [
    "Risk_Category", "Risk_Index", "Prob_High", "Prob_Moderate", "Prob_Low",
    "Confidence", "Needs_Manual_Review", "Support_Priority",
    "Suggested_Follow_Up_Days", "Attention_Areas", "Weakest_Subject", "Subject_Average",
]

# --------------------------------------------------------------------------- #
# Indicator benchmarks.  cuts = (on-track, watch, needs-attention):
#   value >= cuts[0] -> On track | >= cuts[1] -> Watch
#   value >= cuts[2] -> Needs attention | otherwise -> Critical
# Bands follow the academic-risk methodology of the earlier project version.
# --------------------------------------------------------------------------- #
STATUSES = ["On track", "Watch", "Needs attention", "Critical"]
STATUS_POINTS = [0, 25, 60, 100]
STATUS_COLORS = {
    "On track": "#2e9e6b",
    "Watch": "#e0b100",
    "Needs attention": "#f0a020",
    "Critical": "#d64545",
}

INDICATORS: dict[str, dict] = {
    "attendance": dict(label="Attendance", column="Attendance_Percentage", unit="%",
                       cuts=(85, 75, 60), weight=0.16, target=85),
    "assignment": dict(label="Assignment completion", column="Assignment_Completion_Percentage",
                       unit="%", cuts=(85, 70, 50), weight=0.20, target=85),
    "gpa": dict(label="Previous semester GPA", column="Previous_Semester_GPA", unit=" / 10",
                cuts=(8.0, 6.5, 5.0), weight=0.20, target=8.0),
    "assessment": dict(label="Previous assessment", column="Previous_Assessment_Score", unit="%",
                       cuts=(75, 60, 40), weight=0.20, target=75),
    "subjects": dict(label="Subject average", column="Subject_Average", unit="%",
                     cuts=(75, 60, 50), weight=0.24, target=75),
}
SUBJECT_CUTS = (75, 60, 50)
SUBJECT_TARGET = 75
SUBJECT_IMBALANCE_GAP = 25  # strongest minus weakest subject, in points

IDENTIFIER_TOKENS = {"name", "email", "mail", "phone", "mobile", "contact", "address",
                     "aadhaar", "aadhar", "passport", "ssn"}
