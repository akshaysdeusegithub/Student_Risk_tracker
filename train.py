"""Re-train the risk model and regenerate metrics + reports/evaluation_report.md.

    python train.py
"""
import json

from core import model

if __name__ == "__main__":
    out = model.train_and_save()
    m = out["metrics"]
    print(f"Final model: {m['final_model']}  (trained on {m['n_train']:,}, tested on {m['n_test']:,})")
    for name, r in m["models"].items():
        hc = r["per_class"]["High concern"]
        print(f"  {name:26s} macro-F1={r['macro_f1']:.3f}  High recall={hc['recall']:.3f}  precision={hc['precision']:.3f}")
    print("Saved: models/risk_model.joblib, models/metrics.json, reports/evaluation_report.md")
