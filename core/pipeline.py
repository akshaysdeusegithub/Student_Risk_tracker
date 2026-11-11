"""End-to-end orchestration: raw bytes -> validated -> predicted -> enriched -> saved CSV."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import analysis, config as C, model, validation


def run(data: bytes, bundle: dict) -> tuple[validation.ValidationResult, pd.DataFrame | None]:
    """Parse, validate and predict. Returns (validation result, result table or None)."""
    raw = validation.read_csv_bytes(data)
    val = validation.validate_and_clean(raw, bundle["medians"])
    if not val.ok:
        return val, None
    pred = model.predict(val.df, bundle)
    merged = pd.concat([val.df.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    return val, analysis.enrich(merged)


def save_predictions(df: pd.DataFrame, source_name: str, out_dir: Path | None = None) -> Path:
    """Write the predictions to a NEW csv file (never overwrites the uploaded file)."""
    out_dir = out_dir or C.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source_name).stem).strip("_") or "students"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{stem}_predicted_{stamp}.csv"
    df.to_csv(path, index=False)
    df.to_csv(out_dir / "latest_predictions.csv", index=False)
    return path


def load_saved(path: Path) -> pd.DataFrame:
    """Re-open a previously saved predictions file."""
    df = pd.read_csv(path)
    needed = [C.ID_COL, "Risk_Category", "Risk_Index", "Confidence", *C.PROB_COLS.values(),
              *C.FEATURES, "Subject_Average", "Attention_Areas"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("This file is not a saved predictions file (missing: " + ", ".join(missing) + ").")
    df[C.ID_COL] = df[C.ID_COL].astype(str)
    if "Data_Notes" in df:
        df["Data_Notes"] = df["Data_Notes"].fillna("")
    return df
