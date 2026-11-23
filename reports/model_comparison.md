# Model Comparison

| Model | Validation R2 (%) | Training R2 (%) | R2 Gap (pp) | Model Size (MB) | Train Time (s) | Predict Time (ms) | Eligible | Status |
|---|---|---|---|---|---|---|---|---|
| Linear Regression | 78.39 | 78.8 | 0.41 | 0.002 | 0.0275 | 86.9455 | Yes | ELIGIBLE |
| Decision Tree | 88.66 | 94.98 | 6.31 | 0.0618 | 0.0116 | 77.3889 | No | DISQUALIFIED_GAP |
| Random Forest | 96.89 | 98.89 | 2.0 | 13.1248 | 3.1332 | 3482.3705 | Yes | ELIGIBLE |
| Gradient Boosting | 99.05 | 99.98 | 0.93 | 2.7923 | 11.205 | 88.9829 | Yes | CHAMPION |

**Champion:** Gradient Boosting

> **Selection Rationale:** Gradient Boosting is selected as champion: validation R2 = 99.05%, gap = 0.93pp (limit 5.0%), model size = 2.792 MB, train time = 11.21s, prediction latency = 88.98ms (selection cost = 0.0000).
