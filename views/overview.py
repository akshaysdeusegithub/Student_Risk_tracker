"""Landing page: upload -> validate -> predict -> overall findings."""
import hashlib

import streamlit as st

import charts
import ui
from core import analysis, config as C, pipeline

bundle = ui.get_bundle()


def process(name: str, data: bytes) -> None:
    """Validate, predict and save; keeps the outcome in st.session_state."""
    report = {"name": name, "ok": False, "errors": [], "warnings": [], "info": [], "stats": {}}
    with st.status("Validating your file and predicting risk categories...", expanded=True) as status:
        try:
            st.write("Reading and validating the file...")
            val, result = pipeline.run(data, bundle)
            report.update(ok=val.ok, errors=val.errors, warnings=val.warnings, info=val.info, stats=val.stats)
            if result is None:
                status.update(label="Validation failed - see the messages below", state="error", expanded=False)
            else:
                st.write(f"Predicting risk for {len(result):,} students...")
                path = pipeline.save_predictions(result, name)
                report["saved_to"] = str(path.relative_to(C.ROOT))
                st.session_state["csv_bytes"] = result.to_csv(index=False).encode("utf-8")
                st.session_state["csv_name"] = path.name
                st.session_state["results"] = result
                st.session_state["source_name"] = name
                st.session_state.pop("focus_student", None)
                status.update(label="Done - results saved", state="complete", expanded=False)
        except ValueError as exc:
            report.update(ok=False, errors=[str(exc)])
            status.update(label="The file could not be read", state="error", expanded=False)
        except Exception as exc:  # noqa: BLE001 - never show a raw traceback to the user
            report.update(ok=False, errors=[f"Unexpected problem while processing the file: {exc}"])
            status.update(label="Something went wrong", state="error", expanded=False)
    if not report["ok"]:
        st.session_state.pop("results", None)
    st.session_state["validation"] = report
    st.rerun()


# --------------------------------------------------------------------------- #
# Header + upload
# --------------------------------------------------------------------------- #
st.title("🎓 Student Risk Tracker")
st.caption("Upload a student CSV and get an automatic validation, ML-based risk categories and clear findings. "
           "Outputs are supportive prompts for educators - never grades or final judgements.")

with st.container(border=True):
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("##### 1 · Upload student data")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")
        st.caption("Use de-identified data only (no names, e-mails or phone numbers). "
                   "Column names such as 'Attendance %' or 'Maths' are matched automatically.")
    with right:
        st.markdown("##### Required columns")
        st.caption(", ".join(C.REQUIRED_COLS))
        b1, b2 = st.columns(2)
        b1.download_button("⬇ CSV template", ui.template_csv(), "student_template.csv", "text/csv")
        use_sample = b2.button("▶ Try sample data")

if use_sample:
    sample = C.SAMPLE_DIR / "sample_students.csv"
    process(sample.name, sample.read_bytes())
elif uploaded is not None:
    data = uploaded.getvalue()
    sig = hashlib.md5(data).hexdigest()
    if st.session_state.get("last_upload_sig") != sig:  # process each new file once, not on every rerun
        st.session_state["last_upload_sig"] = sig
        process(uploaded.name, data)
else:
    st.session_state.pop("last_upload_sig", None)

with st.expander("Open a previously saved prediction file"):
    saved = sorted(C.OUTPUT_DIR.glob("*_predicted_*.csv"), reverse=True) if C.OUTPUT_DIR.exists() else []
    if not saved:
        st.caption("No saved runs yet. Every upload is saved automatically in outputs/predictions/.")
    else:
        pick = st.selectbox("Saved run", saved, format_func=lambda p: p.name)
        if st.button("Load this run"):
            try:
                st.session_state["results"] = pipeline.load_saved(pick)
                st.session_state["source_name"] = pick.name
                st.session_state["csv_bytes"] = pick.read_bytes()
                st.session_state["csv_name"] = pick.name
                st.session_state["validation"] = {"ok": True, "errors": [], "warnings": [], "info": [],
                                                  "stats": {}, "name": pick.name,
                                                  "saved_to": str(pick.relative_to(C.ROOT))}
                st.session_state.pop("focus_student", None)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

# --------------------------------------------------------------------------- #
# Validation feedback
# --------------------------------------------------------------------------- #
val = st.session_state.get("validation")
df = ui.get_results()

if val and val["errors"]:
    st.error("**We could not use this file.**")
    for msg in val["errors"]:
        st.markdown(f"- {msg}")
    st.caption("Fix the issues above and upload the file again, or start from the CSV template.")
elif val and df is not None:
    stats = val.get("stats", {})
    if stats:
        st.success(f"File accepted - **{stats.get('rows_used', len(df)):,}** students analysed "
                   f"({stats.get('rows_removed', 0)} row(s) removed).")
    else:
        st.success(f"Loaded saved run **{val.get('name')}** - {len(df):,} students.")
    if val["warnings"] or val["info"]:
        with st.expander(f"Validation report ({len(val['warnings'])} warning(s), {len(val['info'])} note(s))"):
            for w in val["warnings"]:
                st.markdown(f"- ⚠️ {w}")
            for i in val["info"]:
                st.markdown(f"- ℹ️ {i}")
    c1, c2 = st.columns([3, 1])
    c1.caption(f"Predictions saved to `{val.get('saved_to', 'outputs/predictions')}`")
    c2.download_button("⬇ Download predictions CSV", st.session_state["csv_bytes"],
                       st.session_state["csv_name"], "text/csv")

if df is None:
    if not (val and val["errors"]):
        st.info("Upload a CSV file, or click **Try sample data**, to see the overall risk picture.")
    st.stop()

# --------------------------------------------------------------------------- #
# Overall findings
# --------------------------------------------------------------------------- #
n = len(df)
counts = analysis.category_counts(df)
share = dict(zip(counts["Category"], counts["Share"]))
num = dict(zip(counts["Category"], counts["Students"]))
support_share = share["Moderate concern"] + share["High concern"]

st.markdown("### 2 · Overall picture")
k = st.columns(5)
with k[0]:
    ui.kpi("Students analysed", f"{n:,}", "in this file")
with k[1]:
    ui.kpi("Needing support", ui.pct(support_share), "Moderate + High concern", "#4f46e5")
with k[2]:
    ui.kpi("High concern", ui.pct(share["High concern"]), f"{num['High concern']:,} students",
           C.CATEGORY_COLORS["High concern"])
with k[3]:
    ui.kpi("Moderate concern", ui.pct(share["Moderate concern"]), f"{num['Moderate concern']:,} students",
           C.CATEGORY_COLORS["Moderate concern"])
with k[4]:
    ui.kpi("Low concern", ui.pct(share["Low concern"]), f"{num['Low concern']:,} students",
           C.CATEGORY_COLORS["Low concern"])

st.write("")
g1, g2, g3 = st.columns([1, 1, 1.2], gap="large")
with g1:
    ui.plot(charts.gauge(support_share, "Overall risk: students needing support"), key="ov_gauge")
with g2:
    ui.plot(charts.donut(counts), key="ov_donut")
with g3:
    st.markdown("#### Key findings")
    for line in analysis.key_findings(df):
        st.markdown(f"- {line}")

a1, a2 = st.columns(2, gap="large")
with a1:
    ui.plot(charts.attention_bar(analysis.attention_rates(df)), key="ov_attention")
with a2:
    ui.plot(charts.stacked_hist(df, "Attendance_Percentage", "Attendance distribution by concern level",
                                value_range=(40, 100), xlabel="Attendance %"), key="ov_att_hist")

st.markdown("### 3 · Students to check in with first")
top = analysis.rank_for_review(df).head(10)
cols = [C.ID_COL, "Risk_Category", "Risk_Index", "Confidence", "Attention_Areas", "Weakest_Subject"]
ui.show_df(top[cols], hide_index=True, column_config={
    "Risk_Index": st.column_config.ProgressColumn("Risk index", min_value=0, max_value=100, format="%.0f"),
    "Confidence": st.column_config.NumberColumn("Model confidence", format="%.2f"),
    "Risk_Category": "Concern level", "Attention_Areas": "Areas needing attention",
    "Weakest_Subject": "Weakest subject"})
n1, n2, n3 = st.columns(3)
n1.page_link("views/review_queue.py", label="Open the full Review Queue", icon=":material/checklist:")
n2.page_link("views/student_report.py", label="Open a detailed Student Report", icon=":material/person_search:")
n3.page_link("views/insights.py", label="Explore Cohort Insights", icon=":material/insights:")
st.caption("A high-ranking student is a prompt for a supportive conversation - not a verdict. "
           "Please review the context before acting.")
