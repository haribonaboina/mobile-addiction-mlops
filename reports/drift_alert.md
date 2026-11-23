# ✅ Production Data Drift Report

- **Status:** `HEALTHY DISTRIBUTION`
- **Drifted Features:** `4 / 15` (26.67%)
- **Significance Threshold:** `p < 0.05`
- **Production Samples Tested:** `50`

### 🔬 Kolmogorov-Smirnov (KS) Two-Sample Tests
| Feature | Status | p-value | KS Stat | Baseline (Mean ± Std) | Production (Mean ± Std) | Mean Shift |
|---|---|---|---|---|---|---|
| `daily_usage_hours` | OK ✅ | 0.3577 | 0.1284 | 5.03 ± 1.94 | 5.16 ± 1.39 | +0.14 |
| `sleep_hours` | OK ✅ | 0.1915 | 0.1505 | 6.48 ± 1.49 | 6.52 ± 1.04 | +0.04 |
| `exercise_hours` | OK ✅ | 0.1564 | 0.1570 | 1.04 ± 0.73 | 1.18 ± 0.60 | +0.13 |
| `phone_checks_per_day` | DRIFT ⚠️ | 0.0031 | 0.2511 | 83.00 ± 37.59 | 87.04 ± 21.94 | +4.03 |
| `apps_used_daily` | OK ✅ | 0.0605 | 0.1843 | 12.66 ± 4.61 | 12.45 ± 3.78 | -0.21 |
| `time_on_social_media` | OK ✅ | 0.9086 | 0.0770 | 2.51 ± 0.99 | 2.48 ± 0.85 | -0.02 |
| `time_on_gaming` | OK ✅ | 0.0561 | 0.1863 | 1.52 ± 0.94 | 1.52 ± 0.54 | +0.00 |
| `weekend_minus_daily` | DRIFT ⚠️ | 0.0000 | 0.3466 | 0.98 ± 2.79 | 1.34 ± 0.78 | +0.36 |
| `usage_per_check` | DRIFT ⚠️ | 0.0195 | 0.2123 | 0.08 ± 0.06 | 0.06 ± 0.02 | -0.02 |
| `checks_per_hour` | DRIFT ⚠️ | 0.0167 | 0.2159 | 25.69 ± 77.96 | 17.70 ± 6.24 | -8.00 |
| `social_gaming_total` | OK ✅ | 0.5057 | 0.1139 | 4.03 ± 1.35 | 4.01 ± 1.01 | -0.02 |
| `leisure_share` | OK ✅ | 0.2205 | 0.1458 | 1.26 ± 3.71 | 0.82 ± 0.33 | -0.44 |
| `apps_x_usage` | OK ✅ | 0.5575 | 0.1094 | 63.78 ± 35.24 | 66.02 ± 31.90 | +2.25 |
| `usage_sleep_ratio` | OK ✅ | 0.2099 | 0.1475 | 0.81 ± 0.38 | 0.80 ± 0.25 | -0.01 |
| `bedtime_share` | OK ✅ | 0.5538 | 0.1097 | 0.31 ± 0.94 | 0.20 ± 0.10 | -0.11 |

---
> ✅ **Status**: No automated retraining required. Production distribution is within statistical tolerance.
