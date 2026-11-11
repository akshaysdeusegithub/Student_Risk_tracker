# Evaluation report

_Generated 2026-11-10T08:06:02+00:00 UTC - scikit-learn 1.9.1, Python 3.13.3_

## Data and target

- Training file: `data/training/students_labeled.csv` (50,000 students)
- Train / test split: 40,000 / 10,000 (stratified 80/20, random_state=42)
- Target: documented `Risk_Score` binned into Low (<30), Moderate (30-49) and High (>=50) concern; `Risk_Score`/`At_Risk` are never used as inputs.
- Class distribution: Low concern: 22,910, Moderate concern: 9,557, High concern: 17,533
- Model inputs: Attendance_Percentage, Assignment_Completion_Percentage, Previous_Semester_GPA, Previous_Assessment_Score, Math_Score, Science_Score, English_Score, Social_Studies_Score, Business_Studies_Score, Computer_Science_Score

## Model comparison (held-out test set)

| Model | CV macro-F1 (train) | Test accuracy | Test macro-F1 | High-concern recall | High-concern precision |
|---|---|---|---|---|---|
| Majority-class baseline | 0.209 | 0.458 | 0.209 | 0.000 | 0.000 |
| Logistic Regression | 0.715 | 0.746 | 0.718 | 0.749 | 0.787 |
| Gradient Boosting (final) | 0.996 | 0.998 | 0.998 | 0.998 | 0.999 |

## Final model: Gradient Boosting

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low concern | 0.999 | 0.998 | 0.998 | 4582 |
| Moderate concern | 0.997 | 0.998 | 0.998 | 1911 |
| High concern | 0.999 | 0.998 | 0.998 | 3507 |

Confusion matrix (rows = actual, columns = predicted; order: Low concern, Moderate concern, High concern):

```
  4574       4       4
     2    1908       1
     4       2    3501
```

- High-concern students missed (false negatives): **6** of 3507
- Needs-support (Moderate+High) recall: 0.999, precision: 0.999

## Important limitation

The training labels are derived from a documented scoring rule over attendance, assignment completion, GPA and assessment scores. The near-perfect scores therefore show that the model has learned that rule - NOT that it can forecast real-world outcomes. On real, independently labelled data expect lower performance, re-train, and re-evaluate before relying on it. Outputs are decision-support signals for educators only.
