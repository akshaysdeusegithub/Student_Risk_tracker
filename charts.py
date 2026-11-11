"""Plotly figure builders (kept separate so the page files stay readable)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core import config as C


def _layout(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
    )
    return fig


def gauge(value: float, title: str, maximum: float = 100, suffix: str = "%") -> go.Figure:
    frac = value / maximum if maximum else 0
    color = C.CATEGORY_COLORS["Low concern"] if frac < 0.25 else (
        C.CATEGORY_COLORS["Moderate concern"] if frac < 0.5 else C.CATEGORY_COLORS["High concern"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value),
        number={"suffix": suffix, "valueformat": ".1f"},
        title={"text": title},
        gauge={"axis": {"range": [0, maximum]}, "bar": {"color": color, "thickness": 0.3},
               "steps": [{"range": [0, maximum * 0.25], "color": "rgba(46,158,107,0.18)"},
                         {"range": [maximum * 0.25, maximum * 0.5], "color": "rgba(240,160,32,0.18)"},
                         {"range": [maximum * 0.5, maximum], "color": "rgba(214,69,69,0.18)"}]}))
    return _layout(fig, height=260, legend=False)


def donut(counts: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=counts["Category"], values=counts["Students"], hole=0.62, sort=False,
        marker=dict(colors=[C.CATEGORY_COLORS[c] for c in counts["Category"]]),
        textinfo="percent", hovertemplate="%{label}: %{value:,} students<extra></extra>"))
    fig.add_annotation(text=f"<b>{int(counts['Students'].sum()):,}</b><br>students", showarrow=False,
                       x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=16))
    fig.update_layout(title="Students by concern level")
    return _layout(fig, height=300)


def attention_bar(rates: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(y=rates["Indicator"], x=rates["Needs attention"], name="Needs attention", orientation="h",
                marker_color=C.STATUS_COLORS["Needs attention"])
    fig.add_bar(y=rates["Indicator"], x=rates["Critical"], name="Critical", orientation="h",
                marker_color=C.STATUS_COLORS["Critical"])
    fig.update_layout(barmode="stack", title="Share of students needing attention, by area (%)",
                      yaxis=dict(autorange="reversed"), xaxis=dict(title="% of students"))
    return _layout(fig, height=320)


def stacked_hist(df: pd.DataFrame, column: str, title: str, bins: int = 24,
                 value_range: tuple[float, float] | None = None, xlabel: str | None = None) -> go.Figure:
    lo, hi = value_range or (float(df[column].min()), float(df[column].max()))
    if hi <= lo:
        hi = lo + 1
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure()
    for cat in C.CATEGORIES:
        counts, _ = np.histogram(df.loc[df["Risk_Category"] == cat, column], bins=edges)
        fig.add_bar(x=centers, y=counts, name=cat, marker_color=C.CATEGORY_COLORS[cat])
    fig.update_layout(barmode="stack", bargap=0.04, title=title,
                      xaxis=dict(title=xlabel or column.replace("_", " ")), yaxis=dict(title="Students"))
    return _layout(fig, height=320)


def heatmap(matrix: pd.DataFrame, title: str, fmt: str = ".1f", colorscale: str = "RdYlGn",
            reverse: bool = False) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.index),
        colorscale=colorscale, reversescale=reverse,
        text=np.round(matrix.to_numpy(dtype=float), 1), texttemplate="%{text}",
        hovertemplate="%{y} / %{x}: %{z:.1f}<extra></extra>", colorbar=dict(thickness=12)))
    fig.update_layout(title=title, yaxis=dict(autorange="reversed"))
    return _layout(fig, height=max(240, 60 * len(matrix) + 120), legend=False)


def grouped_bars(matrix: pd.DataFrame, title: str, ytitle: str) -> go.Figure:
    """matrix: index = Risk_Category, columns = metrics."""
    fig = go.Figure()
    for cat in matrix.index:
        fig.add_bar(x=list(matrix.columns), y=matrix.loc[cat].to_numpy(), name=cat,
                    marker_color=C.CATEGORY_COLORS.get(cat))
    fig.update_layout(barmode="group", title=title, yaxis=dict(title=ytitle))
    return _layout(fig, height=340)


def share_bars(dist: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for cat in C.CATEGORIES:
        fig.add_bar(y=list(dist.index), x=dist[cat].to_numpy(), name=cat, orientation="h",
                    marker_color=C.CATEGORY_COLORS[cat])
    fig.update_layout(barmode="stack", title=title, xaxis=dict(title="% of group", range=[0, 100]))
    return _layout(fig, height=max(220, 50 * len(dist) + 130))


def scatter_sample(df: pd.DataFrame, x: str, y: str, title: str, n: int = 2500) -> go.Figure:
    s = df.sample(min(n, len(df)), random_state=1)
    fig = go.Figure()
    for cat in C.CATEGORIES:
        part = s[s["Risk_Category"] == cat]
        fig.add_scatter(x=part[x], y=part[y], mode="markers", name=cat,
                        marker=dict(color=C.CATEGORY_COLORS[cat], size=6, opacity=0.6),
                        hovertext=part[C.ID_COL])
    fig.update_layout(title=title, xaxis=dict(title=x.replace("_", " ")), yaxis=dict(title=y.replace("_", " ")))
    return _layout(fig, height=380)


def subject_bars(subjects: list[dict]) -> go.Figure:
    names = [s["name"] for s in subjects]
    fig = go.Figure()
    fig.add_bar(x=names, y=[s["score"] for s in subjects], name="Student",
                marker_color=[C.STATUS_COLORS[s["status"]] for s in subjects],
                text=[f"{s['score']:.0f}" for s in subjects], textposition="outside")
    fig.add_scatter(x=names, y=[s["cohort_avg"] for s in subjects], mode="markers", name="Cohort average",
                    marker=dict(symbol="diamond", size=11, color="#4f46e5", line=dict(width=1, color="white")))
    fig.add_hline(y=C.SUBJECT_TARGET, line_dash="dot", line_color="gray",
                  annotation_text=f"Supportive range ({C.SUBJECT_TARGET}+)", annotation_position="top left")
    fig.update_layout(title="Subject scores vs. cohort average", yaxis=dict(range=[0, 110], title="Score"))
    return _layout(fig, height=360)


def radar(subjects: list[dict]) -> go.Figure:
    names = [s["name"] for s in subjects]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[s["cohort_avg"] for s in subjects] + [subjects[0]["cohort_avg"]],
                                  theta=names + [names[0]], name="Cohort average",
                                  line=dict(color="#9ca3af", dash="dot")))
    fig.add_trace(go.Scatterpolar(r=[s["score"] for s in subjects] + [subjects[0]["score"]],
                                  theta=names + [names[0]], name="Student", fill="toself",
                                  line=dict(color="#4f46e5")))
    fig.update_layout(title="Subject profile", polar=dict(radialaxis=dict(range=[0, 100], visible=True)))
    return _layout(fig, height=360)


def probability_bar(probs: dict[str, float]) -> go.Figure:
    fig = go.Figure()
    for cat in C.CATEGORIES:
        fig.add_bar(y=["Model estimate"], x=[probs.get(cat, 0) * 100], name=cat, orientation="h",
                    marker_color=C.CATEGORY_COLORS[cat], text=[f"{probs.get(cat, 0):.0%}"],
                    textposition="inside")
    fig.update_layout(barmode="stack", title="How sure is the model?", xaxis=dict(range=[0, 100], title="%"))
    return _layout(fig, height=190)


def confusion_matrix(cm: list[list[int]]) -> go.Figure:
    labels = [c.replace(" concern", "") for c in C.CATEGORIES]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale="Blues", text=cm,
                               texttemplate="%{text}", showscale=False,
                               hovertemplate="Actual %{y} / Predicted %{x}: %{z}<extra></extra>"))
    fig.update_layout(title="Confusion matrix (test set)", xaxis=dict(title="Predicted"),
                      yaxis=dict(title="Actual", autorange="reversed"))
    return _layout(fig, height=360, legend=False)


def importance_bar(items: list[dict]) -> go.Figure:
    items = sorted(items, key=lambda d: d["importance"])
    fig = go.Figure(go.Bar(x=[i["importance"] for i in items],
                           y=[i["feature"].replace("_", " ") for i in items], orientation="h",
                           marker_color="#4f46e5"))
    fig.update_layout(title="What drives the model? (permutation importance)",
                      xaxis=dict(title="Drop in macro-F1 when the feature is shuffled"))
    return _layout(fig, height=380, legend=False)


def threshold_lines(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    for key, name, color in (("precision", "Precision", "#4f46e5"), ("recall", "Recall", "#d64545"),
                             ("f1", "F1", "#2e9e6b")):
        fig.add_scatter(x=[r["threshold"] for r in rows], y=[r[key] for r in rows], mode="lines+markers",
                        name=name, line=dict(color=color))
    fig.update_layout(title="High-concern detection at different probability thresholds",
                      xaxis=dict(title="Flag a student when P(High concern) >="), yaxis=dict(range=[0, 1.02]))
    return _layout(fig, height=340)
