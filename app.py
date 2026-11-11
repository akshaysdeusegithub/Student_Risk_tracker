"""Student Risk Tracker - Streamlit entry point.

Run with:   streamlit run app.py
The pages live in ./views and are shown in a navigation bar at the top.
"""
import streamlit as st

st.set_page_config(page_title="Student Risk Tracker", page_icon="🎓", layout="wide",
                   initial_sidebar_state="collapsed")

import ui  # noqa: E402  (must come after set_page_config)

ui.inject_css()

# pages = [
#     st.Page("views/overview.py", title="Upload & Overview", icon=":material/upload_file:", default=True),
#     st.Page("views/insights.py", title="Cohort Insights", icon=":material/insights:"),
#     st.Page("views/review_queue.py", title="Review Queue", icon=":material/checklist:"),
#     st.Page("views/student_report.py", title="Student Report", icon=":material/person_search:"),
#     # st.Page("views/model_info.py", title="Model & Responsible Use", icon=":material/verified_user:"),
# ]

overview = st.Page("views/overview.py", title="Upload & Overview",
                   icon=":material/upload_file:", default=True)
pages = [overview]
if st.session_state.get("results") is not None:
    pages += [
        st.Page("views/insights.py", title="Cohort Insights", icon=":material/insights:"),
        st.Page("views/review_queue.py", title="Review Queue", icon=":material/checklist:"),
        st.Page("views/student_report.py", title="Student Report", icon=":material/person_search:"),
        # st.Page("views/model_info.py", title="Model & Responsible Use", icon=":material/verified_user:"),
    ]

try:
    nav = st.navigation(pages, position="top")  # top navigation bar (Streamlit >= 1.46)
except TypeError:  # very old Streamlit: fall back to the sidebar menu
    nav = st.navigation(pages)
nav.run()
