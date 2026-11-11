"""Detailed, student-specific report: indicators, subject-wise performance, attention areas, recommendations."""
import pandas as pd
import streamlit as st

import charts
import ui
from core import analysis, config as C

df = ui.require_results()
ranked = analysis.rank_for_review(df)

st.title("Student Report")
st.caption("A supportive, student-specific view for educators. It highlights where help may be useful - it does not "
           "grade, penalise or permanently label a student.")

# --------------------------------------------------------------------------- #
# Choose a student
# --------------------------------------------------------------------------- #
ids = df[C.ID_COL]
focus = st.session_state.get("focus_student")
if focus is None or not ids.eq(focus).any():
    focus = ranked[C.ID_COL].iloc[0]

with st.container(border=True):
    s1, s2 = st.columns([1, 2])
    query = s1.text_input("Search by Student ID", placeholder="e.g. STU_00042")
    if query.strip():
        options = ids[ids.str.contains(query.strip(), case=False, regex=False)].head(200).tolist()
        if not options:
            s1.warning("No student matches this search.")
            options = [focus]
    else:
        options = ranked[C.ID_COL].head(200).tolist()
        if focus not in options:
            options = [focus] + options
    cat_lookup = df.loc[ids.isin(options)].set_index(C.ID_COL)["Risk_Category"].to_dict()
    index = options.index(focus) if focus in options else 0
    choice = s2.selectbox("Select a student (highest priority first)", options, index=index,
                          format_func=lambda sid: f"{sid}  -  {cat_lookup.get(sid, '')}")
st.session_state["focus_student"] = choice

row = df.loc[ids.eq(choice)].iloc[0]
rep = analysis.student_report(row, df)
color = C.CATEGORY_COLORS[rep["category"]]

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(f"## {rep['id']} &nbsp; {ui.badge(rep['category'], color)}", unsafe_allow_html=True)
st.markdown(f"> {rep['summary']}")

if rep["needs_manual_review"]:
    msg = "The model was less certain about this record"
    if rep["data_notes"]:
        msg += f" and some values were missing ({rep['data_notes']})"
    st.warning(msg + ". Please verify the data and use your own judgement.")

h1, h2, h3 = st.columns([1, 1.4, 1], gap="large")
with h1:
    ui.plot(charts.gauge(rep["risk_index"], "Risk index (0-100)"), key="sr_gauge")
with h2:
    ui.plot(charts.probability_bar(rep["probabilities"]), key="sr_prob")
    st.caption("Model confidence: "
               f"**{rep['confidence']:.0%}**. A lower value means the student sits near the border between two groups.")
with h3:
    ui.kpi("Suggested follow-up", f"{rep['priority']}", f"check in within ~{rep['follow_up_days']} days", color)
    st.write("")
    st.download_button("⬇ Download this report", analysis.report_to_markdown(rep).encode("utf-8"),
                       f"{rep['id']}_support_report.md", "text/markdown")

# --------------------------------------------------------------------------- #
# Key indicators
# --------------------------------------------------------------------------- #
st.markdown("### Key indicators")
cols = st.columns(len(rep["indicators"]))
for col, ind in zip(cols, rep["indicators"]):
    with col:
        ui.indicator_card(ind)

# --------------------------------------------------------------------------- #
# Subject-wise performance
# --------------------------------------------------------------------------- #
st.markdown("### Subject-wise performance")
p1, p2 = st.columns([1.2, 1], gap="large")
with p1:
    ui.plot(charts.subject_bars(rep["subjects"]), key="sr_bars")
with p2:
    ui.plot(charts.radar(rep["subjects"]), key="sr_radar")

table = pd.DataFrame([{
    "Subject": s["name"], "Score": s["score"], "Cohort average": round(s["cohort_avg"], 1),
    "Difference": round(s["diff"], 1),
    "Percentile in cohort": None if s["percentile"] is None else round(s["percentile"]),
    "Points to supportive range": round(s["gap"], 1),
    "Status": f"{ui.STATUS_EMOJI[s['status']]} {s['status']}",
} for s in sorted(rep["subjects"], key=lambda s: s["score"])])
ui.show_df(table, hide_index=True, column_config={
    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
    "Difference": st.column_config.NumberColumn("vs. cohort", format="%+.1f"),
    "Percentile in cohort": st.column_config.NumberColumn("Percentile", format="%d"),
})
st.caption("Sorted from weakest to strongest. Percentile = share of the cohort scoring lower "
           "(shown when the cohort has at least 10 students).")

# --------------------------------------------------------------------------- #
# Attention areas / strengths / patterns
# --------------------------------------------------------------------------- #
st.markdown("### Areas that need attention")
a1, a2 = st.columns([3, 2], gap="large")
with a1:
    if rep["attention"]:
        for a in rep["attention"]:
            st.markdown(f"{ui.badge(a['status'], C.STATUS_COLORS[a['status']])} &nbsp; **{a['area']}** - {a['detail']}",
                        unsafe_allow_html=True)
            st.write("")
    else:
        st.success("No indicator or subject is currently below the supportive range.")
    for pattern in rep["patterns"]:
        st.info(pattern)
    st.caption("Status labels use fixed benchmarks. They can differ from the model's overall concern level, which "
               "weighs all indicators together - use both as context.")
with a2:
    st.markdown("**Strengths to build on**")
    if rep["strengths"]:
        for s in rep["strengths"]:
            st.markdown(f"- ✅ {s}")
    else:
        st.caption("No indicator is fully on track yet - recognise effort and small wins.")

# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #
st.markdown("### Suggested support (for educator review)")
for i, rec in enumerate(rep["recommendations"]):
    status_color = C.STATUS_COLORS.get(rec["severity"], C.CATEGORY_COLORS.get(rec["severity"], "#4f46e5"))
    with st.expander(f"{rec['area']}  ·  {rec['severity']}", expanded=i < 3):
        for action in rec["actions"]:
            st.markdown(f"- {action}")

st.caption("These suggestions are generated from simple benchmarks and the model's estimate. They are conversation "
           "starters - the educator decides what, if anything, to do. Nothing here changes a grade or triggers "
           "any penalty.")
