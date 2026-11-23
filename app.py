"""Streamlit serving application for the Mobile Usage Addiction Prediction MLOps Pipeline.

Features:
- Live model inference with dynamic behavioral feature engineering and risk scoring.
- Production payload and prediction logging for real-time data drift monitoring.
- Model lineage, Optuna trials, and multi-criteria champion benchmark inspector.
- Kolmogorov-Smirnov statistical data drift monitoring dashboard.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.monitor_drift import (
    INFERENCE_LOG_PATH,
    REFERENCE_DATA_PATH,
    analyze_drift,
    load_inference_logs,
    log_inference,
)
from src.feature_engineering import engineer_features, SELECTED_FEATURES

MODEL_PATH = Path(__file__).parent / "models" / "model.pkl"
SELECTION_PATH = Path(__file__).parent / "reports" / "model_selection.json"
METRICS_PATH = Path(__file__).parent / "reports" / "metrics.json"

st.set_page_config(
    page_title="Mobile Addiction MLOps Production System",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .status-badge {
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        display: inline-block;
    }
    .risk-high { background-color: #fee2e2; color: #dc2626; border: 1px solid #f87171; }
    .risk-mod { background-color: #fef3c7; color: #d97706; border: 1px solid #fcd34d; }
    .risk-low { background-color: #dcfce7; color: #16a34a; border: 1px solid #86efac; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_champion_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reports():
    selection, metrics = None, None
    if SELECTION_PATH.exists():
        with open(SELECTION_PATH, "r", encoding="utf-8") as f:
            selection = json.load(f)
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return selection, metrics


model = load_champion_model()
selection, metrics = load_reports()

# Sidebar: System Metadata
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/smartphone-tablet.png", width=64)
    st.title("📱 MLOps Monitor")
    st.caption("Production Mobile Addiction Serving")

    if selection and "champion" in selection:
        champ = selection["champion"]
        st.success(f"**Champion Model:**\n{champ['display_name']}")
        st.markdown(f"**Validation $R^2$:** `{champ['criteria']['validation_r2_pct']:.2f}%`")
        st.markdown(f"**Overfitting Gap:** `{champ['criteria']['r2_gap_pct']:.2f} pp`")
        st.markdown(f"**Model Size:** `{champ['criteria']['model_size_mb']:.3f} MB`")
    else:
        st.warning("Model selection metadata not found.")

    st.divider()
    st.caption("DVC Pipeline & Drift Monitoring Engine")
    st.caption("Target Bounded Scale: 1.0 (Low) to 10.0 (Severe)")

tab_inference, tab_metrics, tab_monitoring = st.tabs([
    "🎯 Live Inference",
    "📊 Model Lineage & Benchmark",
    "🔬 Data Drift Monitoring",
])

# ------------------ TAB 1: LIVE INFERENCE ------------------
with tab_inference:
    st.header("Predict Mobile Addiction Risk")
    st.write(
        "Enter user smartphone usage habits and behavioral metrics. "
        "The automated pipeline computes the 15 behavioral interaction features in real time."
    )

    if model is None:
        st.error("Champion model not found at `models/model.pkl`. Please execute the training pipeline first.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("⏱️ Usage Patterns")
            daily_usage = st.slider("Daily Usage (Hours)", 0.5, 14.0, 6.0, step=0.25)
            weekend_usage = st.slider("Weekend Usage (Hours)", 0.5, 16.0, 7.5, step=0.25)
            screen_before_bed = st.slider("Screen Time Before Bed (Hours)", 0.0, 4.0, 1.2, step=0.1)
            phone_checks = st.slider("Phone Checks per Day", 10, 200, 85, step=5)

        with col2:
            st.subheader("🎮 Activity Breakdown")
            social_media = st.slider("Time on Social Media (Hours)", 0.0, 8.0, 3.0, step=0.25)
            gaming = st.slider("Time on Gaming (Hours)", 0.0, 8.0, 1.5, step=0.25)
            education = st.slider("Time on Education (Hours)", 0.0, 6.0, 1.0, step=0.25)
            apps_used = st.slider("Apps Used Daily", 2, 35, 14, step=1)

        with col3:
            st.subheader("🧘 Health & Wellbeing")
            sleep = st.slider("Sleep Duration (Hours)", 3.0, 11.0, 6.5, step=0.25)
            exercise = st.slider("Daily Exercise (Hours)", 0.0, 4.0, 1.0, step=0.25)
            anxiety = st.slider("Anxiety Level (1-10)", 1, 10, 5)
            depression = st.slider("Depression Level (1-10)", 1, 10, 4)
            self_esteem = st.slider("Self Esteem (1-10)", 1, 10, 6)

        if st.button("Predict Addiction Level", type="primary", width="stretch"):
            # Raw input dictionary
            raw_input = {
                "daily_usage_hours": daily_usage,
                "weekend_usage_hours": weekend_usage,
                "screen_time_before_bed": screen_before_bed,
                "phone_checks_per_day": phone_checks,
                "time_on_social_media": social_media,
                "time_on_gaming": gaming,
                "time_on_education": education,
                "apps_used_daily": apps_used,
                "sleep_hours": sleep,
                "exercise_hours": exercise,
                "anxiety_level": anxiety,
                "depression_level": depression,
                "self_esteem": self_esteem,
            }

            # Run through feature engineering
            input_df = pd.DataFrame([raw_input])
            engineered_df = engineer_features(input_df)
            X_input = engineered_df[SELECTED_FEATURES]

            # Predict and clamp [1.0, 10.0]
            raw_pred = model.predict(X_input)[0]
            pred = float(np.clip(raw_pred, 1.0, 10.0))

            if pred >= 7.5:
                risk_label = "Severe Addiction Risk"
                risk_class = "risk-high"
            elif pred >= 4.5:
                risk_label = "Moderate Usage Level"
                risk_class = "risk-mod"
            else:
                risk_label = "Healthy / Low Risk"
                risk_class = "risk-low"

            # Log to production monitoring store
            log_inference(X_input, pred, risk_level=risk_label)

            st.divider()
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown(f"### Predicted Addiction Score: **{pred:.2f} / 10.0**")
                st.markdown(f'<span class="status-badge {risk_class}">{risk_label}</span>', unsafe_allow_html=True)
                st.progress(float((pred - 1.0) / 9.0))

            with res_col2:
                st.markdown("##### Key Behavioral Interaction Indicators")
                st.write(f"- **Apps × Daily Usage:** `{X_input['apps_x_usage'].values[0]:.1f}`")
                st.write(f"- **Usage-to-Sleep Ratio:** `{X_input['usage_sleep_ratio'].values[0]:.2f}`")
                st.write(f"- **Social + Gaming Leisure:** `{X_input['social_gaming_total'].values[0]:.1f} hrs ({X_input['leisure_share'].values[0]*100:.1f}%)`")
                st.write(f"- **Hourly Check Rate:** `{X_input['checks_per_hour'].values[0]:.1f} checks/hr`")

# ------------------ TAB 2: MODEL LINEAGE & METRICS ------------------
with tab_metrics:
    st.header("Model Lineage & Multi-Criteria Benchmark")
    st.write(
        "Candidate regressors were trained with Optuna hyperparameter optimization and cross-validated. "
        "The champion model is selected using strict overfitting limits, validation score, and low-latency tie-breaking."
    )

    comp_csv = Path(__file__).parent / "reports" / "model_comparison.csv"
    if comp_csv.exists():
        comp_df = pd.read_csv(comp_csv)
        st.dataframe(comp_df, width="stretch")
    elif selection:
        st.json(selection)
    else:
        st.info("Run `dvc repro` to generate comparison metrics.")

    if metrics:
        st.subheader("Unseen Test Set Metrics")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Test R² Score", f"{metrics.get('test_r2', 0.0):.4f}")
        m_col2.metric("Mean Absolute Error (MAE)", f"{metrics.get('test_mae', 0.0):.4f}")
        m_col3.metric("Root Mean Squared Error", f"{metrics.get('test_rmse', 0.0):.4f}")
        m_col4.metric("Within ±1.0 pt Accuracy", f"{metrics.get('test_accuracy_within_1pt', 0.0)*100:.1f}%")

# ------------------ TAB 3: DATA DRIFT MONITORING ------------------
with tab_monitoring:
    st.header("Production Data Drift & Quality Monitoring")
    st.write(
        "Production inference requests are continuously evaluated against baseline training distributions "
        "using the two-sample **Kolmogorov-Smirnov (KS)** statistical hypothesis test."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_check = st.button("Run Drift Check", type="primary")

    log_df = load_inference_logs(INFERENCE_LOG_PATH)
    if not log_df.empty:
        st.write(f"**Total production records logged:** `{len(log_df)}`")

        if run_check or len(log_df) >= 5:
            ref_path = REFERENCE_DATA_PATH if REFERENCE_DATA_PATH.exists() else Path("data/raw/phone_addiction_dataset.csv")
            if ref_path.exists():
                ref_df = pd.read_csv(ref_path)
                report = analyze_drift(ref_df, log_df)

                status_color = "red" if report["dataset_drift_detected"] else "green"
                status_msg = "⚠️ DRIFT DETECTED" if report["dataset_drift_detected"] else "✅ HEALTHY DISTRIBUTION"

                st.markdown(f"#### Distribution Status: :{status_color}[{status_msg}]")
                st.write(
                    f"Drifted features: **{report['drifted_features']} / {report['total_features']}** "
                    f"({report['drift_share_pct']}%)"
                )

                drift_rows = []
                for feat, d in report["features"].items():
                    drift_rows.append({
                        "Feature": feat,
                        "Status": "DRIFT ⚠️" if d["drift_detected"] else "HEALTHY ✅",
                        "p-value": round(d["p_value"], 4),
                        "KS Statistic": round(d["ks_statistic"], 4),
                        "Baseline Mean": round(d.get("reference_mean", 0.0), 2),
                        "Production Mean": round(d.get("current_mean", 0.0), 2),
                        "Mean Shift": round(d.get("mean_shift", 0.0), 2),
                    })
                st.dataframe(pd.DataFrame(drift_rows), width="stretch")

        with st.expander("View Recent Production Logs (Last 10 Inferences)"):
            st.dataframe(log_df.tail(10), width="stretch")
    else:
        st.info("No production inferences logged yet. Head over to the 'Live Inference' tab to submit predictions.")
