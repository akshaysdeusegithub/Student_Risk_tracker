# 🎓 Student Risk Tracker

A Streamlit web app for educators: **upload a student CSV → automatic validation → ML risk prediction → overall and student-specific findings.**
Built for *Project 04 – AI Student Success and At-Risk Prediction*.

> **Responsible use.** Risk categories are *decision-support signals*, not judgements. The app never changes grades, contacts students or triggers penalties. Every flag needs human review.

---

## Quick start

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows      |   source .venv/bin/activate   # macOS / Linux

# 2. install and launch
pip install -r requirements.txt
streamlit run app.py
```

Or double-click `run.bat` (Windows) / run `./run.sh` (macOS, Linux).

> On the first launch the app may re-train its model for about 10–15 s if your scikit-learn version differs from the one used to build `models/risk_model.joblib`. This is automatic.
> Requires **Streamlit ≥ 1.46** (top navigation bar). Older versions fall back to a sidebar menu.

## Demo in 60 seconds

1. Open the app and click **▶ Try sample data** (or upload `sample_data/sample_students.csv`).
2. Read the **overall picture**: risk percentage gauge, concern-level split, key findings, areas needing attention.
3. **Cohort Insights** → indicator distributions, subject heat-map, relationships, group (fairness) view.
4. **Review Queue** → filter the students, tick a row and open the full report.
5. **Student Report** → subject-wise performance vs. cohort, areas needing attention, strengths and recommendations; download it as Markdown.
6. Try `sample_data/sample_students_with_issues.csv` to see validation in action (renamed columns, bad values, duplicates, a name column that gets removed).

## What each page does

| Page | Purpose |
|---|---|
| **Upload & Overview** (landing) | Upload CSV · automatic validation report · predictions saved to a *separate* CSV · KPI cards, overall risk gauge, donut chart, key findings, attention-area chart, top-10 priority list |
| **Cohort Insights** | Indicators by concern level, subject-wise heat-map and weakest subjects, scatter/correlation views, concern level by Stream / Gender / Age (monitoring only) |
| **Review Queue** | Filter by concern level, attention area, stream, manual-review flag; download the list; open a student's report |
| **Student Report** | Search any student · risk index and model confidence · 5 key indicators vs. cohort · **subject-wise** bar/radar/table · attention areas · patterns worth discussing · **recommendations** · Markdown download |
| **Model & Responsible Use** | Model comparison, confusion matrix, per-class precision/recall/F1, feature importance, threshold analysis, model card, data dictionary, responsible-use guidance, re-train button |

## Input CSV format

**Required columns** (names are matched flexibly – `Attendance %`, `Maths`, `GPA` etc. work):

`Student_ID, Attendance_Percentage, Assignment_Completion_Percentage, Previous_Semester_GPA, Previous_Assessment_Score, Math_Score, Science_Score, English_Score, Social_Studies_Score, Business_Studies_Score, Computer_Science_Score`

Optional: `Stream`, `Age`, `Gender` (shown in the group view only; **never used by the model**). Any extra columns are carried through to the output. Download a ready-made template from the landing page.

### Automatic validation
* Missing required columns, empty files, unreadable CSVs → clear error, nothing is predicted.
* Column-name variants matched automatically (reported to you).
* `"85%"` style values parsed; non-numeric text and out-of-range values (e.g. attendance 150, GPA 12) treated as missing.
* Blank / duplicate `Student_ID` rows removed; rows with too little data removed.
* Missing values filled with typical training values, **noted in `Data_Notes`** and flagged for manual review.
* Warns if values look like fractions (0–1) or a 4-point GPA scale.
* Columns that look like names, e-mails, phone numbers or addresses are **removed** and not stored.

## Output

Every upload creates a new file in `outputs/predictions/<name>_predicted_<timestamp>.csv` (plus `latest_predictions.csv`). The uploaded file is never modified. Added columns:

`Risk_Category, Risk_Index, Prob_High, Prob_Moderate, Prob_Low, Confidence, Needs_Manual_Review, Support_Priority, Suggested_Follow_Up_Days, Attention_Areas, Weakest_Subject, Subject_Average, Data_Notes`

Saved runs can be re-opened from the landing page.

## How the model works

* **Target** – the dataset's documented `Risk_Score` (0–100) is binned into *Low* (<30), *Moderate* (30–49) and *High* concern (≥50, identical to the dataset's `At_Risk` flag). Nothing is invented.
* **Inputs** – 10 numeric columns only (4 core indicators + 6 subject scores). `Risk_Score` / `At_Risk` are excluded (leakage); gender, age and stream are excluded (sensitive).
* **Models compared** – majority-class dummy, Logistic Regression (baseline), Gradient Boosting (final), all on the same stratified 80/20 split; selection by 5-fold CV on the *training* data only, so the test set stays untouched.
* **Metrics** – precision, recall, F1, confusion matrix, false negatives for High concern, threshold analysis. See `reports/evaluation_report.md` and the *Model & Responsible Use* page.
* **Per-student findings** – status bands (On track / Watch / Needs attention / Critical) use transparent benchmarks (attendance 85/75/60, assignments 85/70/50, GPA 8/6.5/5, assessment 75/60/40, subjects 75/60/50) defined in `core/config.py`. Subject-wise results are compared with the uploaded cohort.
* **Uncertainty** – class probabilities and confidence are shown; records with confidence < 0.60 or imputed values are flagged for manual review.

### ⚠️ Important limitation
The bundled labels come from a scoring rule over attendance, assignments, GPA and assessments. Gradient Boosting therefore scores ≈ 99.8 % macro-F1 – it has **learned that rule**, which does not prove it can forecast real student outcomes. With real, independently labelled data expect lower scores: re-train (`python train.py`), re-evaluate and review fairness before using it in practice. Logistic Regression (≈ 75 % accuracy) shows what a simpler baseline achieves.

## Project structure

```
app.py                 Streamlit entry point + top navigation bar
views/                 overview · insights · review_queue · student_report · model_info
core/                  Streamlit-free logic: config, validation, model, analysis, pipeline
charts.py, ui.py       Plotly figures and shared UI helpers
train.py               Re-train + regenerate metrics/report
data/training/         Labelled training data (de-identified)
sample_data/           Demo upload files
models/                risk_model.joblib, metrics.json
reports/               evaluation_report.md, test_evidence.txt
outputs/predictions/   Saved prediction CSVs (created at run time)
notebooks/             01_exploratory_analysis.ipynb
docs/                  data_dictionary_and_privacy.md, model_card.md
tests/                 pytest suite (validation, model, analysis, pipeline)
```

## Testing

```bash
pytest            # 23 tests: validation rules, leakage checks, model sanity, reports, CSV output
```
`reports/test_evidence.txt` holds a recorded run.

## Mapping to the project brief

| Required deliverable | Where |
|---|---|
| Dataset source / data dictionary / privacy note | `docs/data_dictionary_and_privacy.md` |
| Exploratory analysis notebook | `notebooks/01_exploratory_analysis.ipynb` |
| Reproducible ML pipeline + evaluation report | `core/model.py`, `train.py`, `reports/evaluation_report.md` |
| Dashboard with aggregate trends and review flags | Overview, Cohort Insights, Review Queue pages |
| README (intended use, limitations, demo) | this file |
| Stretch: thresholds & model card | *Model & Responsible Use* page, `docs/model_card.md` |


## Troubleshooting

* *"Missing required column(s)"* – compare your header row with the template; see the validation message for exact names.
* *Top navigation not shown* – upgrade Streamlit: `pip install -U streamlit`.
* *Model re-trains at start-up* – expected when scikit-learn versions differ; it is saved afterwards.
* *Large files* – up to 200 000 rows are accepted; 50 000 rows process in about 2 seconds.
