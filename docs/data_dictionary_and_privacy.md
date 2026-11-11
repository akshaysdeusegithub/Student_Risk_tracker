# Data dictionary and privacy note

## Dataset
* **Training data:** `data/training/students_labeled.csv` – 50,000 de-identified student records (10 model inputs plus `Student_ID`, `Stream`, `Risk_Score`, `At_Risk`).
* **Source / licence:** _add the original source and licence here before submission._
* **Target:** `Risk_Score` (0–100, documented in the dataset) binned into Low (<30) / Moderate (30–49) / High (≥50) concern. High equals the dataset's `At_Risk = 1`.

## Columns
| Column | Used by model | Meaning | Range |
|---|---|---|---|
| Student_ID | no (key only) | De-identified student code, unique | – |
| Attendance_Percentage | yes | % of classes attended | 0–100 |
| Assignment_Completion_Percentage | yes | % of assignments completed | 0–100 |
| Previous_Semester_GPA | yes | GPA on a 10-point scale | 0–10 |
| Previous_Assessment_Score | yes | Score in previous assessment | 0–100 |
| Math / Science / English / Social_Studies / Business_Studies / Computer_Science _Score | yes | Subject scores | 0–100 |
| Stream, Age, Gender | **no** | Optional; shown in the group (fairness) view only | – |
| Risk_Score, At_Risk | **no** (label) | Ground-truth label columns; excluded to prevent leakage | – |

## Output columns
`Risk_Category`, `Risk_Index` (0–100 probability-weighted, for ordering only), `Prob_High/Moderate/Low`, `Confidence`, `Needs_Manual_Review`, `Support_Priority`, `Suggested_Follow_Up_Days`, `Attention_Areas`, `Weakest_Subject`, `Subject_Average`, `Data_Notes`.

## Privacy note
* Only de-identified data should be uploaded. The validator removes columns that look like names, e-mails, phone numbers or addresses.
* Sensitive attributes (gender, age, stream) are **not** model inputs.
* Predictions are saved locally in `outputs/predictions/`. Restrict access to that folder, delete files no longer needed, and add authentication before hosting the app beyond a trusted machine.
* Model output is a support signal. No automatic action is taken on any student.
