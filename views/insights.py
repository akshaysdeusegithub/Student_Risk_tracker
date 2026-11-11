"""Cohort-level findings: indicators, subjects, relationships and group fairness view."""
import pandas as pd
import streamlit as st

import charts
import ui
from core import analysis, config as C

df = ui.require_results()

st.title("Cohort Insights")
st.caption(f"Aggregate patterns for **{len(df):,}** students. Use these to plan support for groups, "
           "then open the Review Queue for individual follow-up.")

# tab_ind, tab_sub, tab_rel, tab_grp = st.tabs(
#     ["📊 Indicators", "📚 Subjects", "🔗 Relationships", "⚖️ Group view"])

tab_ind, tab_sub, tab_grp = st.tabs(
    ["📊 Indicators", "📚 Subjects", "⚖️ Group view"])

# --------------------------------------------------------------------------- #
with tab_ind:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.plot(charts.stacked_hist(df, "Assignment_Completion_Percentage",
                                    "Assignment completion by concern level", value_range=(30, 100),
                                    xlabel="Assignment completion %"), key="in_assign")
    with c2:
        ui.plot(charts.stacked_hist(df, "Previous_Semester_GPA", "Previous semester GPA by concern level",
                                    value_range=(3, 10), xlabel="GPA (out of 10)"), key="in_gpa")
    pct_cols = ["Attendance_Percentage", "Assignment_Completion_Percentage",
                "Previous_Assessment_Score", "Subject_Average"]
    means = analysis.means_by_category(df, pct_cols)
    means.columns = ["Attendance", "Assignments", "Assessment", "Subject average"]
    ui.plot(charts.grouped_bars(means, "Average indicator by concern level (%)", "Average %"), key="in_means")

    rates = analysis.attention_rates(df).rename(columns={
        "Needs attention": "% Needs attention", "Critical": "% Critical", "Total": "% Total flagged"})
    st.markdown("##### How many students are below the supportive range?")
    ui.show_df(rates.round(1), hide_index=True)

# --------------------------------------------------------------------------- #
with tab_sub:
    summary = analysis.subject_summary(df)
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        by_cat = df.groupby("Risk_Category")[C.SUBJECT_COLS].mean().reindex(C.CATEGORIES)
        by_cat.columns = [C.SUBJECTS[c] for c in by_cat.columns]
        ui.plot(charts.heatmap(by_cat, "Average subject score by concern level"), key="sub_heat")
    with c2:
        st.markdown("##### Subject overview")
        ui.show_df(summary.round(1).rename(columns={
            "Below 50": "% below 50", "Below 60": "% below 60", "Above 75": "% 75 or above"}),
            hide_index=True)
    weakest = summary.sort_values("Below 60", ascending=False).iloc[0]
    st.info(f"**{weakest['Subject']}** has the largest share of students scoring below 60 "
            f"({weakest['Below 60']:.1f}%). Consider group-level revision support for this subject.")
    counts = df["Weakest_Subject"].value_counts().rename_axis("Subject").reset_index(name="Students")
    st.markdown("##### Which subject is each student's weakest?")
    ui.show_df(counts, hide_index=True)

# --------------------------------------------------------------------------- #
# with tab_rel:
#     c1, c2 = st.columns(2, gap="large")
#     with c1:
#         ui.plot(charts.scatter_sample(df, "Attendance_Percentage", "Assignment_Completion_Percentage",
#                                       "Attendance vs. assignment completion (sample)"), key="rel_sc1")
#     with c2:
#         ui.plot(charts.scatter_sample(df, "Previous_Semester_GPA", "Previous_Assessment_Score",
#                                       "GPA vs. previous assessment (sample)"), key="rel_sc2")
#     corr_cols = C.CORE_COLS + ["Subject_Average", "Risk_Index"]
#     corr = df[corr_cols].corr().round(2)
#     corr.index = corr.columns = ["Attendance", "Assignments", "GPA", "Assessment", "Subject avg", "Risk index"]
#     ui.plot(charts.heatmap(corr, "Correlation between indicators and risk index", colorscale="RdBu"),
#             key="rel_corr")
#     st.caption("Correlation shows association, not cause. A negative link with the risk index means higher values "
#                "go with lower estimated concern.")

# --------------------------------------------------------------------------- #
with tab_grp:
    st.markdown("Comparing how concern levels are spread across groups helps you notice **possible unfairness**. "
                "These columns are **not used by the model** - they are shown for monitoring only.")
    found = [c for c in ("Stream", "Gender", "Age") if c in df.columns]
    if not found:
        st.info("The uploaded file has no Stream, Gender or Age column, so no group comparison is available.")
    for col in found:
        data = df if col != "Age" else df.assign(Age=pd.to_numeric(df["Age"], errors="coerce").astype("Int64"))
        dist = analysis.group_distribution(data.dropna(subset=[col]), col)
        sizes = data[col].astype(str).value_counts()
        if len(dist) > 12:
            st.caption(f"{col}: too many groups to display.")
            continue
        ui.plot(charts.share_bars(dist, f"Concern level by {col.lower()}"), key=f"grp_{col}")
        st.caption("Group sizes: " + ", ".join(f"{k} = {v:,}" for k, v in sizes.items() if k in dist.index))
