"""Shared Streamlit helpers: theme, cards, safe wrappers and session access."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from core import config as C, model

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1250px;}
.kpi {border-radius: 12px; padding: 14px 16px; background: rgba(128,128,128,0.08);
      border: 1px solid rgba(128,128,128,0.22);}
.kpi-label {font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em; opacity: .7;}
.kpi-value {font-size: 1.9rem; font-weight: 700; line-height: 1.25;}
.kpi-sub {font-size: 0.8rem; opacity: .7;}
.badge {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem;
        font-weight: 600; vertical-align: middle;}
.ind {border-radius: 12px; padding: 12px 14px; background: rgba(128,128,128,0.08);
      border: 1px solid rgba(128,128,128,0.22); height: 100%;}
.ind-top {display:flex; justify-content:space-between; align-items:center; gap:6px; font-size:.85rem;}
.ind-val {font-size: 1.55rem; font-weight: 700; margin: 4px 0;}
.bar {height: 8px; border-radius: 6px; background: rgba(128,128,128,0.25); overflow: hidden; margin: 6px 0;}
.bar > div {height: 100%; border-radius: 6px;}
.ind-sub {font-size: .76rem; opacity: .75;}
.note {font-size: .82rem; opacity: .75;}
</style>
"""

STATUS_EMOJI = {"On track": "🟢", "Watch": "🟡", "Needs attention": "🟠", "Critical": "🔴"}


# --------------------------------------------------------------------------- #
# Model / session
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading prediction model...")
def get_bundle() -> dict:
    return model.load_bundle()


def get_results() -> pd.DataFrame | None:
    return st.session_state.get("results")


def require_results() -> pd.DataFrame:
    """Return the predictions table or show a friendly prompt and stop the page."""
    df = get_results()
    if df is None:
        st.info("No student data loaded yet. Upload a CSV on the **Upload & Overview** page first.")
        st.page_link("views/overview.py", label="Go to Upload & Overview", icon=":material/upload_file:")
        st.stop()
    return df


def template_csv() -> bytes:
    cols = C.REQUIRED_COLS + ["Stream"]
    rows = [["STU_0001", 82.5, 75.0, 7.4, 68.0, 71, 66, 74, 69, 72, 80, "Science"],
            ["STU_0002", 58.0, 45.5, 5.1, 41.0, 52, 48, 60, 55, 50, 58, "Commerce"]]
    buf = io.StringIO()
    pd.DataFrame(rows, columns=cols).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    return (f'<span class="badge" style="background:{color}22;color:{color};'
            f'border:1px solid {color}66">{text}</span>')


def kpi(label: str, value: str, sub: str = "", color: str = "#4f46e5") -> None:
    st.markdown(f'<div class="kpi" style="border-left:6px solid {color}">'
                f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)


def indicator_card(ind: dict) -> None:
    color = C.STATUS_COLORS[ind["status"]]
    scale = 10.0 if ind["key"] == "gpa" else 100.0
    width = max(0.0, min(ind["value"] / scale * 100, 100.0))
    diff = ind["diff"]
    gap = f"{ind['gap']:.1f} below the supportive range" if ind["gap"] > 0 else "within the supportive range"
    st.markdown(
        f'<div class="ind"><div class="ind-top"><b>{ind["label"]}</b>{badge(ind["status"], color)}</div>'
        f'<div class="ind-val">{ind["value"]:.1f}<span style="font-size:.9rem;opacity:.7">{ind["unit"]}</span></div>'
        f'<div class="bar"><div style="width:{width:.0f}%;background:{color}"></div></div>'
        f'<div class="ind-sub">Cohort avg {ind["cohort_avg"]:.1f} ({diff:+.1f}) · {gap}</div></div>',
        unsafe_allow_html=True)


def plot(fig, key: str | None = None) -> None:
    """st.plotly_chart that works across Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:  # noqa: BLE001 - older Streamlit versions
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_df(df: pd.DataFrame, **kwargs):
    """st.dataframe stretched to the container, across Streamlit versions."""
    try:
        return st.dataframe(df, width="stretch", **kwargs)
    except Exception:  # noqa: BLE001 - older Streamlit versions
        return st.dataframe(df, use_container_width=True, **kwargs)


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}%"
