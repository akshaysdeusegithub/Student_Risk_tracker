# Model card – Student Risk Tracker

| | |
|---|---|
| **Intended use** | Prompt supportive educator check-ins; plan group-level support. |
| **Out of scope** | Grading, public ranking, disciplinary or admission decisions, any automatic action. |
| **Task** | 3-class classification: Low / Moderate / High concern. |
| **Inputs** | Attendance %, assignment completion %, previous GPA (10-pt), previous assessment, six subject scores. |
| **Excluded** | Names/identifiers, gender, age, stream, `Risk_Score`, `At_Risk`. |
| **Algorithm** | Median imputer → `HistGradientBoostingClassifier` (baselines: Logistic Regression, majority class). |
| **Evaluation** | Stratified 80/20 split, 5-fold CV on training data for selection; precision, recall, F1, confusion matrix, High-concern false negatives, threshold sweep. Details: `reports/evaluation_report.md`. |
| **Uncertainty** | Class probabilities + confidence; < 0.60 confidence or imputed values ⇒ manual-review flag. |
| **Known risks** | Labels are rule-derived, so test scores overstate real-world performance; may under-serve groups absent from training data; assumes 10-point GPA; benchmarks are generic. |
| **Human oversight** | Review Queue → Student Report → supportive check-in → re-check after the suggested window. |
