"""Model transparency: metrics, limitations, privacy and responsible use."""
import pandas as pd
import streamlit as st

import charts
import ui
from core import config as C, model

st.title("Model & Responsible Use")
st.caption("How the predictions are made, how well they perform and what they must not be used for.")

metrics = model.load_metrics()
if metrics is None:
    st.warning("No evaluation metrics found. Click **Re-train model** below to create them.")

tab_perf, tab_card, tab_data, tab_use = st.tabs(
    ["📈 Performance", "🪪 Model card", "🗂️ Data dictionary", "🤝 Responsible use"])

# --------------------------------------------------------------------------- #
with tab_perf:
    if metrics:
        final = metrics["final_model"]
        fm = metrics["models"][final]
        st.markdown(f"**Final model:** {final} &nbsp;·&nbsp; trained on {metrics['n_train']:,} students, "
                    f"tested on {metrics['n_test']:,} unseen students.")
        st.error("**Read this first:** the training labels come from a documented scoring rule over attendance, "
                 "assignments, GPA and assessments. Near-perfect scores mean the model learned that rule - they do "
                 "**not** prove it can forecast real-world outcomes. Re-train and re-evaluate on independently "
                 "labelled data before relying on it.")
        rows = []
        for name, r in metrics["models"].items():
            hc = r["per_class"]["High concern"]
            rows.append({"Model": name + (" (final)" if name == final else ""),
                         "CV macro-F1 (train)": round(r["cv_macro_f1_mean"], 3),
                         "Test accuracy": round(r["accuracy"], 3), "Test macro-F1": round(r["macro_f1"], 3),
                         "High-concern recall": round(hc["recall"], 3),
                         "High-concern precision": round(hc["precision"], 3),
                         "High-concern missed": r["high_concern_missed"]})
        st.markdown("##### Model comparison (held-out test set)")
        ui.show_df(pd.DataFrame(rows), hide_index=True)
        st.caption("The majority-class baseline reaches a high-looking accuracy while missing every High-concern "
                   "student - that is why precision, recall and the confusion matrix matter more than accuracy.")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            ui.plot(charts.confusion_matrix(fm["confusion_matrix"]), key="mi_cm")
            st.caption(f"High-concern students missed (false negatives): **{fm['high_concern_missed']}** of "
                       f"{fm['high_concern_total']:,}.")
        with c2:
            per = pd.DataFrame(fm["per_class"]).T.reset_index().rename(columns={"index": "Class"})
            per[["precision", "recall", "f1"]] = per[["precision", "recall", "f1"]].round(3)
            st.markdown("##### Per-class results (final model)")
            ui.show_df(per, hide_index=True)
            st.markdown(f"Needs-support (Moderate + High) recall: **{fm['needs_support_recall']:.3f}**, "
                        f"precision: **{fm['needs_support_precision']:.3f}**")
        c3, c4 = st.columns(2, gap="large")
        with c3:
            ui.plot(charts.importance_bar(metrics["permutation_importance"]), key="mi_imp")
        with c4:
            ui.plot(charts.threshold_lines(metrics["thresholds_high_concern"]), key="mi_thr")
            st.caption("Lower thresholds flag more students (higher recall, lower precision). Choosing a threshold "
                       "is a policy decision: missing a student who needs help usually costs more than an extra check-in.")

# --------------------------------------------------------------------------- #
with tab_card:
    st.markdown(f"""
**Model card - Student Risk Tracker**

| | |
|---|---|
| **Purpose** | Help educators decide who may benefit from a supportive check-in. |
| **Not for** | Grading, ranking students publicly, disciplinary action, admissions, or any automatic decision. |
| **Task** | 3-class classification: Low / Moderate / High concern. |
| **Inputs** | {len(C.FEATURES)} numeric columns: attendance, assignment completion, previous GPA, previous assessment and six subject scores. |
| **Not used** | Gender, age, stream, names or any identifier. |
| **Algorithm** | Gradient boosting (scikit-learn `HistGradientBoostingClassifier`) after a median imputer; compared with logistic regression and a majority-class baseline. |
| **Outputs** | Concern level, class probabilities, risk index (0-100), model confidence and a manual-review flag when uncertain. |
| **Label source** | Dataset `Risk_Score` binned into Low (<{C.LOW_MAX}), Moderate ({C.LOW_MAX}-{C.HIGH_MIN - 1}), High (>={C.HIGH_MIN}). |
| **Known risks** | Labels are rule-derived, so performance on real data will likely be lower; the model may under-serve groups not represented in the training data; 10-point GPA scale is assumed. |
""")
    if metrics:
        st.caption(f"Trained {metrics['trained_at_utc']} UTC · scikit-learn {metrics['sklearn_version']} · "
                   f"Python {metrics['python_version']}")

# --------------------------------------------------------------------------- #
with tab_data:
    st.markdown("**Required columns** (names are matched flexibly, e.g. `Attendance %`, `Maths`, `GPA`)")
    dd = pd.DataFrame([
        {"Column": C.ID_COL, "Type": "text", "Meaning": "De-identified student code (must be unique)", "Valid range": "-"},
        {"Column": "Attendance_Percentage", "Type": "number", "Meaning": "Share of classes attended", "Valid range": "0-100"},
        {"Column": "Assignment_Completion_Percentage", "Type": "number", "Meaning": "Share of assignments completed", "Valid range": "0-100"},
        {"Column": "Previous_Semester_GPA", "Type": "number", "Meaning": "GPA last semester (10-point scale)", "Valid range": "0-10"},
        {"Column": "Previous_Assessment_Score", "Type": "number", "Meaning": "Score in the previous assessment", "Valid range": "0-100"},
        *[{"Column": c, "Type": "number", "Meaning": f"{n} score", "Valid range": "0-100"} for c, n in C.SUBJECTS.items()],
    ])
    ui.show_df(dd, hide_index=True)
    st.markdown("**Optional columns:** `Stream`, `Age`, `Gender` are carried into the output and used only for the "
                "group-fairness view. Columns that look like names, e-mails, phone numbers or addresses are removed "
                "automatically.")
    st.markdown("**Output columns added to the saved CSV:** " + ", ".join(f"`{c}`" for c in C.PREDICTION_COLS) + ".")
    st.markdown(f"**Status bands:** On track / Watch / Needs attention / Critical use the benchmarks below.")
    bands = pd.DataFrame([{"Indicator": s["label"], "On track": f">= {s['cuts'][0]}", "Watch": f">= {s['cuts'][1]}",
                           "Needs attention": f">= {s['cuts'][2]}", "Critical": f"< {s['cuts'][2]}"}
                          for s in C.INDICATORS.values()])
    ui.show_df(bands, hide_index=True)

# --------------------------------------------------------------------------- #
with tab_use:
    st.markdown("""
**Intended use**
- A starting point for supportive, human conversations about who may need extra help.
- Aggregate planning: which areas (attendance, assignments, subjects) need group-level support.

**Not allowed / not recommended**
- Automatic grade changes, penalties or disciplinary action.
- Telling a student or parent that a student "is a failure" or "will fail".
- Using the output without checking the underlying data and context.

**Privacy**
- Upload de-identified data only. Direct identifiers found in column names are removed.
- Predictions are written to `outputs/predictions/` on the machine running the app. Restrict access to that folder,
  delete files you no longer need, and do not deploy the app publicly without authentication.

**Human review process**
1. Open the Review Queue and start with High concern and manual-review records.
2. Check the Student Report and the real context (health, family, workload, data errors).
3. Hold a supportive check-in; record the outcome in your own school system.
4. Re-check after the suggested follow-up window.

**Limitations**
- Predictions are probabilities, not facts; they can be wrong in both directions.
- Missing values are filled with typical training values, which lowers certainty (these rows are flagged).
- Subject benchmarks are generic; adapt them to your grading scheme.
""")

# --------------------------------------------------------------------------- #
st.divider()
with st.expander("Maintenance: re-train the model"):
    st.caption("Re-trains from `data/training/students_labeled.csv` (about 10-15 seconds) and refreshes the "
               "metrics. Already-loaded predictions are not changed - upload the file again to re-score it.")
    if st.button("Re-train model"):
        with st.spinner("Training..."):
            model.train_and_save()
            ui.get_bundle.clear()
        st.success("Model re-trained. Reloading...")
        st.rerun()
