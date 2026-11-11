"""Training, evaluation, persistence and prediction for the risk model.

Target ("label"):  the dataset's documented ``Risk_Score`` (0-100) is binned into
three categories - Low (<30), Moderate (30-49) and High (>=50, identical to the
dataset's own ``At_Risk`` flag).  ``Risk_Score`` and ``At_Risk`` are NEVER used as
model inputs (that would be target leakage).

Models compared (all trained on the same 80 % split, tested on the other 20 %):
  * Majority-class dummy  - shows why accuracy alone is misleading
  * Logistic Regression   - simple, interpretable baseline
  * Gradient Boosting     - final model (HistGradientBoostingClassifier)
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Target construction
# --------------------------------------------------------------------------- #
def make_target(risk_score: pd.Series) -> pd.Series:
    """Bin the documented 0-100 Risk_Score into the three risk categories."""
    return pd.cut(risk_score, bins=[-np.inf, C.LOW_MAX - 1e-9, C.HIGH_MIN - 1e-9, np.inf],
                  labels=C.CATEGORIES).astype(str)


def _pipelines() -> dict[str, Pipeline]:
    imp = lambda: SimpleImputer(strategy="median")  # noqa: E731
    return {
        "Majority-class baseline": Pipeline([("imp", imp()), ("model", DummyClassifier(strategy="most_frequent"))]),
        "Logistic Regression": Pipeline([("imp", imp()), ("scale", StandardScaler()),
                                         ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Gradient Boosting": Pipeline([("imp", imp()),
                                       ("model", HistGradientBoostingClassifier(
                                           max_depth=4, learning_rate=0.08, max_iter=200,
                                           random_state=RANDOM_STATE))]),
    }


def _ordered_proba(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    proba = pipe.predict_proba(X)
    return pd.DataFrame(proba, columns=pipe.classes_, index=X.index)[C.CATEGORIES]


# --------------------------------------------------------------------------- #
# Training + evaluation
# --------------------------------------------------------------------------- #
def _evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = _ordered_proba(pipe, X_test)
    pred = proba.idxmax(axis=1)
    p, r, f, s = precision_recall_fscore_support(y_test, pred, labels=C.CATEGORIES, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=C.CATEGORIES)
    high_idx = C.CATEGORIES.index("High concern")
    high_missed = int(cm[high_idx].sum() - cm[high_idx, high_idx])
    # "needs support" = Moderate or High (binary view for educators)
    needs_true = y_test.ne("Low concern")
    needs_pred = pred.ne("Low concern")
    try:
        auc = float(roc_auc_score(y_test, proba.to_numpy(), multi_class="ovr", average="macro",
                                  labels=C.CATEGORIES))
    except ValueError:
        auc = None
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", labels=C.CATEGORIES, zero_division=0)),
        "roc_auc_ovr": auc,
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, c in enumerate(C.CATEGORIES)},
        "confusion_matrix": cm.tolist(),
        "high_concern_missed": high_missed,
        "high_concern_total": int(cm[high_idx].sum()),
        "needs_support_recall": float((needs_pred & needs_true).sum() / max(needs_true.sum(), 1)),
        "needs_support_precision": float((needs_pred & needs_true).sum() / max(needs_pred.sum(), 1)),
    }


def _threshold_table(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> list[dict]:
    """Precision/recall/F1 for detecting High concern at different probability cut-offs."""
    p_high = _ordered_proba(pipe, X_test)["High concern"].to_numpy()
    truth = y_test.eq("High concern").to_numpy()
    rows = []
    for t in np.arange(0.1, 0.95, 0.1):
        flag = p_high >= t
        tp = int((flag & truth).sum())
        prec = tp / flag.sum() if flag.sum() else 0.0
        rec = tp / truth.sum() if truth.sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"threshold": round(float(t), 2), "precision": prec, "recall": rec, "f1": f1,
                     "flagged": int(flag.sum())})
    return rows


def train_and_save(training_csv=None, save: bool = True) -> dict:
    """Train all models, evaluate on a held-out test set, save the final bundle."""
    path = training_csv or C.TRAINING_DATA_PATH
    df = pd.read_csv(path)
    needed = C.FEATURES + ["Risk_Score"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing columns: {missing}")

    y = make_target(df["Risk_Score"])
    X = df[C.FEATURES].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, pipe in _pipelines().items():
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results[name] = _evaluate(pipe, X_test, y_test)
        results[name]["cv_macro_f1_mean"] = float(cv_f1.mean())
        results[name]["cv_macro_f1_std"] = float(cv_f1.std())

    # model selection uses cross-validation on TRAIN data only (test set stays untouched)
    final_name = max(results, key=lambda k: results[k]["cv_macro_f1_mean"])
    final = fitted[final_name]

    sample = X_test.sample(min(3000, len(X_test)), random_state=RANDOM_STATE)
    pi = permutation_importance(final, sample, y_test.loc[sample.index], scoring="f1_macro",
                                n_repeats=3, random_state=RANDOM_STATE)
    importance = sorted(({"feature": f, "importance": float(m), "std": float(s)}
                         for f, m, s in zip(C.FEATURES, pi.importances_mean, pi.importances_std)),
                        key=lambda d: d["importance"], reverse=True)

    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "final_model": final_name,
        "n_rows": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "class_distribution": {k: int(v) for k, v in y.value_counts().reindex(C.CATEGORIES).items()},
        "features": C.FEATURES,
        "models": results,
        "thresholds_high_concern": _threshold_table(final, X_test, y_test),
        "permutation_importance": importance,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    bundle = {
        "pipeline": final,
        "features": C.FEATURES,
        "categories": C.CATEGORIES,
        "medians": X_train.median().to_dict(),
        "sklearn_version": sklearn.__version__,
        "final_model": final_name,
        "trained_at_utc": metrics["trained_at_utc"],
    }
    if save:
        C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, C.MODEL_PATH, compress=3)
        C.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_evaluation_report(metrics)
    return {"bundle": bundle, "metrics": metrics}


# --------------------------------------------------------------------------- #
# Loading / predicting
# --------------------------------------------------------------------------- #
def load_bundle(auto_retrain: bool = True) -> dict:
    """Load the saved model; retrain automatically if missing or built with another scikit-learn."""
    try:
        bundle = joblib.load(C.MODEL_PATH)
        if bundle.get("sklearn_version") != sklearn.__version__:
            raise RuntimeError("scikit-learn version mismatch")
        return bundle
    except Exception:  # noqa: BLE001 - any failure -> retrain from bundled data
        if not auto_retrain:
            raise
        return train_and_save()["bundle"]


def load_metrics() -> dict | None:
    if C.METRICS_PATH.exists():
        return json.loads(C.METRICS_PATH.read_text(encoding="utf-8"))
    return None


def predict(clean: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Return probability + category columns aligned with ``clean``'s index."""
    X = clean[bundle["features"]].astype(float)
    proba = _ordered_proba(bundle["pipeline"], X)
    cat = proba.idxmax(axis=1)
    out = pd.DataFrame(index=clean.index)
    out["Risk_Category"] = cat
    out["Risk_Index"] = (100 * (0.5 * proba["Moderate concern"] + proba["High concern"])).round(1)
    out["Prob_High"] = proba["High concern"].round(3)
    out["Prob_Moderate"] = proba["Moderate concern"].round(3)
    out["Prob_Low"] = proba["Low concern"].round(3)
    out["Confidence"] = proba.max(axis=1).round(3)
    imputed = clean["Data_Notes"].ne("") if "Data_Notes" in clean else pd.Series(False, index=clean.index)
    out["Needs_Manual_Review"] = (out["Confidence"] < C.LOW_CONFIDENCE) | imputed
    return out


# --------------------------------------------------------------------------- #
# Markdown evaluation report (a required deliverable)
# --------------------------------------------------------------------------- #
def write_evaluation_report(m: dict) -> None:
    final = m["final_model"]
    fm = m["models"][final]
    lines = [
        "# Evaluation report", "",
        f"_Generated {m['trained_at_utc']} UTC - scikit-learn {m['sklearn_version']}, Python {m['python_version']}_", "",
        "## Data and target", "",
        f"- Training file: `data/training/students_labeled.csv` ({m['n_rows']:,} students)",
        f"- Train / test split: {m['n_train']:,} / {m['n_test']:,} (stratified 80/20, random_state=42)",
        f"- Target: documented `Risk_Score` binned into Low (<{C.LOW_MAX}), Moderate ({C.LOW_MAX}-{C.HIGH_MIN - 1}) "
        f"and High (>={C.HIGH_MIN}) concern; `Risk_Score`/`At_Risk` are never used as inputs.",
        "- Class distribution: " + ", ".join(f"{k}: {v:,}" for k, v in m["class_distribution"].items()),
        "- Model inputs: " + ", ".join(m["features"]), "",
        "## Model comparison (held-out test set)", "",
        "| Model | CV macro-F1 (train) | Test accuracy | Test macro-F1 | High-concern recall | High-concern precision |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in m["models"].items():
        hc = r["per_class"]["High concern"]
        lines.append(f"| {name}{' (final)' if name == final else ''} | {r['cv_macro_f1_mean']:.3f} | {r['accuracy']:.3f} "
                     f"| {r['macro_f1']:.3f} | {hc['recall']:.3f} | {hc['precision']:.3f} |")
    lines += ["", f"## Final model: {final}", "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for c, v in fm["per_class"].items():
        lines.append(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
    lines += ["", "Confusion matrix (rows = actual, columns = predicted; order: " + ", ".join(C.CATEGORIES) + "):", "", "```"]
    for row in fm["confusion_matrix"]:
        lines.append("  ".join(f"{v:6d}" for v in row))
    lines += ["```", "",
              f"- High-concern students missed (false negatives): **{fm['high_concern_missed']}** of {fm['high_concern_total']}",
              f"- Needs-support (Moderate+High) recall: {fm['needs_support_recall']:.3f}, precision: {fm['needs_support_precision']:.3f}",
              "", "## Important limitation", "",
              "The training labels are derived from a documented scoring rule over attendance, assignment completion, "
              "GPA and assessment scores. The near-perfect scores therefore show that the model has learned that rule - "
              "NOT that it can forecast real-world outcomes. On real, independently labelled data expect lower performance, "
              "re-train, and re-evaluate before relying on it. Outputs are decision-support signals for educators only.", ""]
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORT_DIR / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    out = train_and_save()
    print(json.dumps({k: v for k, v in out["metrics"].items() if k in ("final_model", "n_rows")}, indent=2))
    print("Python", sys.version.split()[0])
