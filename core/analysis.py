"""Cohort-level and student-level analysis, attention areas and recommendations.

All wording is deliberately supportive: the app produces *prompts for educator
conversations*, never grades, penalties or permanent labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------- #
# Indicator levels (vectorised)
# --------------------------------------------------------------------------- #
def _levels(values: np.ndarray, cuts: tuple[float, float, float]) -> np.ndarray:
    ok, watch, attn = cuts
    return np.where(values >= ok, 0, np.where(values >= watch, 1, np.where(values >= attn, 2, 3)))


def level_of(value: float, cuts: tuple[float, float, float]) -> int:
    return int(_levels(np.array([value], dtype=float), cuts)[0])


def with_subject_average(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Subject_Average"] = out[C.SUBJECT_COLS].mean(axis=1).round(1)
    return out


def indicator_levels(df: pd.DataFrame) -> pd.DataFrame:
    """0-3 level for each indicator (0 = on track ... 3 = critical)."""
    if "Subject_Average" not in df:
        df = with_subject_average(df)
    return pd.DataFrame({k: _levels(df[s["column"]].to_numpy(float), s["cuts"])
                         for k, s in C.INDICATORS.items()}, index=df.index)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add the educator-facing columns (attention areas, priority, weakest subject ...)."""
    out = with_subject_average(df)
    lv = indicator_levels(out)
    labels = np.array([C.INDICATORS[k]["label"] for k in lv.columns], dtype=object)
    flagged = lv.to_numpy() >= 2
    out["Attention_Areas"] = ["; ".join(labels[m]) if m.any() else "None" for m in flagged]
    out["Weakest_Subject"] = out[C.SUBJECT_COLS].idxmin(axis=1).map(C.SUBJECTS)
    out["Support_Priority"] = out["Risk_Category"].map(lambda c: C.SUPPORT[c][0])
    out["Suggested_Follow_Up_Days"] = out["Risk_Category"].map(lambda c: C.SUPPORT[c][1]).astype(int)
    first = [c for c in [C.ID_COL] + C.PREDICTION_COLS if c in out.columns]
    rest = [c for c in out.columns if c not in first and c != "Data_Notes"]
    tail = ["Data_Notes"] if "Data_Notes" in out.columns else []
    return out[first + rest + tail]


# --------------------------------------------------------------------------- #
# Cohort summaries
# --------------------------------------------------------------------------- #
def category_counts(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    counts = df["Risk_Category"].value_counts().reindex(C.CATEGORIES, fill_value=0)
    return pd.DataFrame({"Category": counts.index, "Students": counts.to_numpy(),
                         "Share": (counts.to_numpy() / max(n, 1) * 100).round(1)})


def attention_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Share of students whose indicator is 'Needs attention' or 'Critical'."""
    lv = indicator_levels(df)
    rows = []
    for k, s in C.INDICATORS.items():
        rows.append({"Indicator": s["label"],
                     "Needs attention": float((lv[k] == 2).mean() * 100),
                     "Critical": float((lv[k] == 3).mean() * 100),
                     "Total": float((lv[k] >= 2).mean() * 100)})
    return pd.DataFrame(rows).sort_values("Total", ascending=False).reset_index(drop=True)


def subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, name in C.SUBJECTS.items():
        s = df[col]
        rows.append({"Subject": name, "Average": s.mean(), "Median": s.median(),
                     "Below 50": float((s < 50).mean() * 100),
                     "Below 60": float((s < 60).mean() * 100),
                     "Above 75": float((s >= 75).mean() * 100)})
    return pd.DataFrame(rows)


def means_by_category(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.groupby("Risk_Category")[columns].mean().reindex(C.CATEGORIES).dropna(how="all")


def key_findings(df: pd.DataFrame) -> list[str]:
    """Plain-language headline findings for the overview page."""
    n = len(df)
    if n == 0:
        return []
    counts = df["Risk_Category"].value_counts()
    high, mod = int(counts.get("High concern", 0)), int(counts.get("Moderate concern", 0))
    out = [f"**{high:,} students ({high / n:.1%})** are in the *High concern* group and "
           f"**{mod:,} ({mod / n:.1%})** in *Moderate concern* - together {(high + mod) / n:.1%} of the cohort "
           "may benefit from a supportive check-in."]
    rates = attention_rates(df)
    top = rates.iloc[0]
    if top["Total"] > 0:
        out.append(f"The most common area needing attention is **{top['Indicator']}** "
                   f"({top['Total']:.1f}% of students are below the supportive range).")
    if high:
        hi = df[df["Risk_Category"] == "High concern"]
        out.append(f"High-concern students average **{hi['Attendance_Percentage'].mean():.0f}% attendance** and "
                   f"**{hi['Assignment_Completion_Percentage'].mean():.0f}% assignment completion**, versus "
                   f"{df['Attendance_Percentage'].mean():.0f}% and {df['Assignment_Completion_Percentage'].mean():.0f}% "
                   "for the whole cohort.")
    subj = subject_summary(df).sort_values("Average")
    out.append(f"The lowest-scoring subject overall is **{subj.iloc[0]['Subject']}** "
               f"(average {subj.iloc[0]['Average']:.1f}); the strongest is **{subj.iloc[-1]['Subject']}** "
               f"({subj.iloc[-1]['Average']:.1f}).")
    review = int(df["Needs_Manual_Review"].sum()) if "Needs_Manual_Review" in df else 0
    if review:
        out.append(f"**{review:,} records** are flagged for manual review because the model was less certain "
                   "or some values were missing.")
    return out


# --------------------------------------------------------------------------- #
# Recommendation library
# --------------------------------------------------------------------------- #
_ACTIONS: dict[str, dict[str, list[str]]] = {
    "attendance": {
        "watch": ["Mention the attendance trend at the next routine check-in and ask if anything is getting in the way."],
        "support": [
            "Hold a short, non-judgemental check-in to understand barriers (travel, health, timetable clashes, workload).",
            "Agree a realistic short-term attendance goal and review it in two weeks.",
            "Share notes or recordings of missed classes so catching up feels manageable.",
        ]},
    "assignment": {
        "watch": ["Encourage a simple weekly planner so upcoming deadlines stay visible."],
        "support": [
            "Break pending work into small weekly milestones and review progress together.",
            "Check whether the workload or instructions are the obstacle before assuming a motivation issue.",
            "Offer a short extension or drop-in help session for the most overdue assignments.",
        ]},
    "gpa": {
        "watch": ["Review last semester's results with the student and note which courses dragged the average."],
        "support": [
            "Identify which courses lowered last semester's GPA and offer targeted academic guidance.",
            "Consider peer tutoring or faculty office hours for those courses.",
            "Discuss study habits and time management in a supportive one-to-one conversation.",
        ]},
    "assessment": {
        "watch": ["Suggest a few practice assessments to build confidence before the next test."],
        "support": [
            "Recommend focused revision and timed practice assessments on the weakest topics.",
            "Review difficult topics with the relevant subject teacher and give feedback on past answers.",
            "Check for exam anxiety or format difficulties that may be affecting performance.",
        ]},
    "subjects": {
        "watch": ["Keep an eye on subject scores at the next review."],
        "support": [
            "Review the subject-wise table below and prioritise the weakest one or two subjects.",
            "Arrange subject-specific revision, tutoring or mentoring, and track progress at the next check-in.",
        ]},
}

_SUBJECT_TIPS: dict[str, list[str]] = {
    "Math_Score": ["Short daily practice sets on core topics, starting with the concepts missed most often.",
                   "Pair with a peer tutor or a doubt-clearing session to work through worked examples."],
    "Science_Score": ["Revise concepts with diagrams and short summaries, then attempt past questions.",
                      "Link theory to simple experiments or demonstrations to make ideas stick."],
    "English_Score": ["Daily reading plus a short writing task with teacher feedback.",
                      "Build vocabulary and grammar with a small weekly target."],
    "Social_Studies_Score": ["Use timelines, maps and concept charts to organise facts.",
                             "Discuss topics in small groups or write short answer summaries."],
    "Business_Studies_Score": ["Work through short case studies and connect terms to real examples.",
                               "Practise definitions and frameworks with flash cards."],
    "Computer_Science_Score": ["Schedule regular hands-on practice (small coding exercises or lab tasks).",
                               "Pair programming or a study group can help with debugging skills."],
}

_CATEGORY_ACTIONS = {
    "High concern": "Schedule a supportive one-to-one check-in within about a week and agree on one or two concrete next steps.",
    "Moderate concern": "Plan a brief check-in within about two weeks and revisit the indicators highlighted below.",
    "Low concern": "Continue routine monitoring; acknowledge good progress and re-check at the next review.",
}


# --------------------------------------------------------------------------- #
# Student report
# --------------------------------------------------------------------------- #
def _percentile(series: pd.Series, value: float) -> float | None:
    if len(series) < 10:
        return None
    return float((series < value).mean() * 100)


def student_report(row: pd.Series, cohort: pd.DataFrame) -> dict:
    """Build a structured, educator-friendly report for one student."""
    cohort = cohort if "Subject_Average" in cohort else with_subject_average(cohort)
    if "Subject_Average" not in row.index:
        row = row.copy()
        row["Subject_Average"] = round(float(row[C.SUBJECT_COLS].astype(float).mean()), 1)

    # ---- core indicators ------------------------------------------------------------------
    indicators = []
    for key, spec in C.INDICATORS.items():
        value = float(row[spec["column"]])
        lvl = level_of(value, spec["cuts"])
        avg = float(cohort[spec["column"]].mean())
        indicators.append({
            "key": key, "label": spec["label"], "value": value, "unit": spec["unit"],
            "level": lvl, "status": C.STATUSES[lvl], "cohort_avg": avg, "diff": value - avg,
            "target": spec["target"], "gap": max(spec["target"] - value, 0.0), "weight": spec["weight"],
        })

    # ---- subjects --------------------------------------------------------------------------
    subjects = []
    for col, name in C.SUBJECTS.items():
        score = float(row[col])
        lvl = level_of(score, C.SUBJECT_CUTS)
        avg = float(cohort[col].mean())
        subjects.append({"column": col, "name": name, "score": score, "cohort_avg": avg,
                         "diff": score - avg, "percentile": _percentile(cohort[col], score),
                         "level": lvl, "status": C.STATUSES[lvl], "gap": max(C.SUBJECT_TARGET - score, 0.0)})
    by_score = sorted(subjects, key=lambda s: s["score"])
    weak = [s for s in by_score if s["level"] >= 2]
    strong = [s for s in reversed(by_score) if s["level"] == 0]

    # ---- attention areas, strengths, patterns ------------------------------------------------
    attention = []
    for ind in sorted(indicators, key=lambda i: i["weight"] * C.STATUS_POINTS[i["level"]], reverse=True):
        if ind["level"] >= 2:
            attention.append({
                "area": ind["label"], "level": ind["level"], "status": ind["status"],
                "detail": f"{ind['value']:.1f}{ind['unit']} vs. a supportive range of {ind['target']}{ind['unit']} or more "
                          f"(cohort average {ind['cohort_avg']:.1f}).",
            })
    for s in weak:
        attention.append({
            "area": f"{s['name']} (subject)", "level": s["level"], "status": s["status"],
            "detail": f"Score {s['score']:.1f}, {abs(s['diff']):.1f} points {'below' if s['diff'] < 0 else 'above'} the "
                      f"cohort average of {s['cohort_avg']:.1f}.",
        })

    strengths = []
    for ind in indicators:
        if ind["level"] == 0 and ind["key"] != "subjects":
            strengths.append(f"{ind['label']} is on track ({ind['value']:.1f}{ind['unit']}).")
    for s in strong[:3]:
        strengths.append(f"Strong in {s['name']} ({s['score']:.1f}).")

    lv = {i["key"]: i["level"] for i in indicators}
    patterns = []
    if lv["attendance"] >= 2 and lv["assignment"] >= 2:
        patterns.append("Attendance and assignment completion are both low - this often points to an engagement barrier "
                        "(workload, health, transport, personal circumstances). Explore the reason before drawing conclusions.")
    if lv["attendance"] >= 2 and lv["gpa"] == 0:
        patterns.append("Strong previous results but low attendance - a sign that something outside academics may be affecting attendance.")
    if lv["assessment"] >= 2 and lv["gpa"] <= 1:
        patterns.append("Recent assessment results are lower than the student's GPA suggests - worth checking exam preparation or anxiety.")
    spread = by_score[-1]["score"] - by_score[0]["score"]
    if spread >= C.SUBJECT_IMBALANCE_GAP:
        patterns.append(f"Uneven subject profile: {by_score[-1]['name']} ({by_score[-1]['score']:.0f}) is {spread:.0f} points "
                        f"above {by_score[0]['name']} ({by_score[0]['score']:.0f}). Peer learning or cross-subject study "
                        "strategies could help.")

    # ---- recommendations ----------------------------------------------------------------------
    category = row["Risk_Category"] if "Risk_Category" in row.index else None
    recs = []
    if category:
        recs.append({"area": "Overall", "severity": category, "actions": [_CATEGORY_ACTIONS[category]]})
    for ind in sorted(indicators, key=lambda i: i["weight"] * C.STATUS_POINTS[i["level"]], reverse=True):
        if ind["level"] == 0:
            continue
        kind = "support" if ind["level"] >= 2 else "watch"
        recs.append({"area": ind["label"], "severity": ind["status"], "actions": _ACTIONS[ind["key"]][kind]})
    for s in weak[:3]:
        recs.append({"area": s["name"], "severity": s["status"], "actions": _SUBJECT_TIPS[s["column"]]})
    if strong and weak:
        recs.append({"area": "Build on strengths", "severity": "On track",
                     "actions": [f"Use confidence in {strong[0]['name']} as a base - e.g. mentor a peer or connect "
                                 f"{strong[0]['name']} skills to topics in {weak[0]['name']}."]})
    if not weak and all(i["level"] == 0 for i in indicators):
        recs.append({"area": "Keep it up", "severity": "On track",
                     "actions": ["All indicators are in the supportive range. Recognise the progress and keep routine monitoring."]})

    # ---- summary -------------------------------------------------------------------------------
    sid = row[C.ID_COL]
    if attention:
        top = ", ".join(a["area"] for a in attention[:3])
        summary = f"{sid} is currently in the {category or 'selected'} group. Main areas to look at: {top}."
    else:
        summary = f"{sid} is currently in the {category or 'selected'} group with all indicators in the supportive range."
    if category:
        summary += f" Suggested follow-up: {C.SUPPORT[category][0].lower()} support, around {C.SUPPORT[category][1]} days."

    probs = {c: float(row[p]) for c, p in C.PROB_COLS.items()} if all(p in row.index for p in C.PROB_COLS.values()) else {}
    return {
        "id": sid, "category": category, "probabilities": probs,
        "confidence": float(row["Confidence"]) if "Confidence" in row.index else None,
        "risk_index": float(row["Risk_Index"]) if "Risk_Index" in row.index else None,
        "needs_manual_review": bool(row["Needs_Manual_Review"]) if "Needs_Manual_Review" in row.index else False,
        "data_notes": str(row["Data_Notes"]) if "Data_Notes" in row.index and isinstance(row["Data_Notes"], str) else "",
        "priority": C.SUPPORT[category][0] if category else None,
        "follow_up_days": C.SUPPORT[category][1] if category else None,
        "indicators": indicators, "subjects": subjects, "attention": attention, "strengths": strengths,
        "patterns": patterns, "recommendations": recs, "summary": summary,
        "cohort_size": int(len(cohort)),
    }


def report_to_markdown(rep: dict) -> str:
    """Printable/downloadable version of a student report."""
    L = [f"# Student support report - {rep['id']}", "",
         "> Decision-support summary for educator review. It is not a grade, a penalty or a permanent label.", "",
         f"**Concern level:** {rep['category']}  ", f"**Risk index:** {rep['risk_index']}/100  " if rep["risk_index"] is not None else "",
         f"**Model confidence:** {rep['confidence']:.0%}  " if rep["confidence"] is not None else "",
         f"**Suggested follow-up:** {rep['priority']} (about {rep['follow_up_days']} days)", "",
         "## Summary", rep["summary"], "", "## Key indicators", "",
         "| Indicator | Value | Status | Cohort avg | Target |", "|---|---|---|---|---|"]
    for i in rep["indicators"]:
        L.append(f"| {i['label']} | {i['value']:.1f}{i['unit']} | {i['status']} | {i['cohort_avg']:.1f} | {i['target']}+ |")
    L += ["", "## Subject-wise performance", "", "| Subject | Score | Cohort avg | Difference | Status |", "|---|---|---|---|---|"]
    for s in rep["subjects"]:
        L.append(f"| {s['name']} | {s['score']:.1f} | {s['cohort_avg']:.1f} | {s['diff']:+.1f} | {s['status']} |")
    L += ["", "## Areas needing attention"]
    L += [f"- **{a['area']}** ({a['status']}): {a['detail']}" for a in rep["attention"]] or ["- None identified."]
    if rep["strengths"]:
        L += ["", "## Strengths"] + [f"- {s}" for s in rep["strengths"]]
    if rep["patterns"]:
        L += ["", "## Patterns worth discussing"] + [f"- {p}" for p in rep["patterns"]]
    L += ["", "## Suggested actions (for educator review)"]
    for r in rep["recommendations"]:
        L.append(f"**{r['area']}** ({r['severity']})")
        L += [f"- {a}" for a in r["actions"]]
        L.append("")
    return "\n".join(x for x in L if x is not None)


# --------------------------------------------------------------------------- #
# Ordering and group views
# --------------------------------------------------------------------------- #
def rank_for_review(df: pd.DataFrame) -> pd.DataFrame:
    """Highest risk index first; ties broken by the weakest core indicators."""
    comp = (df["Attendance_Percentage"] + df["Assignment_Completion_Percentage"]
            + df["Previous_Assessment_Score"] + df["Previous_Semester_GPA"] * 10) / 4
    return df.assign(_comp=comp).sort_values(["Risk_Index", "_comp"], ascending=[False, True]).drop(columns="_comp")


def group_distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """% of each group falling in each risk category (for fairness monitoring)."""
    ct = pd.crosstab(df[column].astype(str), df["Risk_Category"], normalize="index") * 100
    return ct.reindex(columns=C.CATEGORIES, fill_value=0).round(1)
