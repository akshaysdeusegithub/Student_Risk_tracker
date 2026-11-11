"""Searchable, filterable list of students ordered by support priority."""
import streamlit as st

import ui
from core import analysis, config as C

df = ui.require_results()
ranked = analysis.rank_for_review(df)

st.title("Review Queue")
st.caption("Students ordered by estimated concern. This is a to-do list for supportive conversations - "
           "every flag needs human judgement.")

with st.container(border=True):
    f1, f2, f3 = st.columns([2, 2, 2])
    cats = f1.multiselect("Concern level", C.CATEGORIES, default=["High concern", "Moderate concern"])
    areas = ["Any"] + [s["label"] for s in C.INDICATORS.values()]
    area = f2.selectbox("Needs attention in", areas)
    search = f3.text_input("Search Student ID", placeholder="e.g. STU_00042")
    g1, g2, g3 = st.columns([2, 2, 2])
    only_review = g1.checkbox("Only records flagged for manual review")
    streams = sorted(df["Stream"].dropna().astype(str).unique()) if "Stream" in df.columns else []
    stream_sel = g2.multiselect("Stream", streams) if streams else []
    min_index = g3.slider("Minimum risk index", 0, 100, 0)

mask = ranked["Risk_Category"].isin(cats) & (ranked["Risk_Index"] >= min_index)
if area != "Any":
    mask &= ranked["Attention_Areas"].str.contains(area, regex=False)
if search.strip():
    mask &= ranked[C.ID_COL].str.contains(search.strip(), case=False, regex=False)
if only_review:
    mask &= ranked["Needs_Manual_Review"].astype(bool)
if stream_sel:
    mask &= ranked["Stream"].astype(str).isin(stream_sel)
view = ranked[mask]

cols = [C.ID_COL, "Risk_Category", "Risk_Index", "Confidence", "Support_Priority",
        "Suggested_Follow_Up_Days", "Attention_Areas", "Weakest_Subject", "Needs_Manual_Review"]
if "Stream" in view.columns:
    cols.insert(2, "Stream")
shown = view[cols].head(2000).reset_index(drop=True)

m1, m2 = st.columns([3, 1])
m1.markdown(f"**{len(view):,}** of {len(df):,} students match"
            + (f" - showing the first {len(shown):,}. Download the CSV for the complete list." if len(view) > len(shown) else "."))
m2.download_button("⬇ Download this list", view[cols].to_csv(index=False).encode("utf-8"),
                   "review_queue.csv", "text/csv")

event = ui.show_df(shown, hide_index=True, on_select="rerun", selection_mode="single-row", column_config={
    "Risk_Index": st.column_config.ProgressColumn("Risk index", min_value=0, max_value=100, format="%.0f"),
    "Confidence": st.column_config.NumberColumn("Model confidence", format="%.2f"),
    "Risk_Category": "Concern level", "Support_Priority": "Support priority",
    "Suggested_Follow_Up_Days": "Follow-up (days)", "Attention_Areas": "Areas needing attention",
    "Weakest_Subject": "Weakest subject", "Needs_Manual_Review": "Manual review"})

try:
    rows = list(event.selection.rows)
except Exception:  # noqa: BLE001 - selection not supported / nothing selected
    rows = []

if rows and rows[0] < len(shown):
    chosen = shown.iloc[rows[0]][C.ID_COL]
    if st.button(f"Open detailed report for {chosen}  →", type="primary"):
        st.session_state["focus_student"] = chosen
        st.switch_page("views/student_report.py")
else:
    st.caption("Tip: click the box at the left of a row, then open that student's detailed report.")
