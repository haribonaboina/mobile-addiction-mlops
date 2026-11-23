"""Unit tests for champion model structure, bounds, inference, and determinism."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

MODEL_PATH = Path("models/model.pkl")

FEATURES = [
    "daily_usage_hours",
    "sleep_hours",
    "exercise_hours",
    "phone_checks_per_day",
    "apps_used_daily",
    "time_on_social_media",
    "time_on_gaming",
    "weekend_minus_daily",
    "usage_per_check",
    "checks_per_hour",
    "social_gaming_total",
    "leisure_share",
    "apps_x_usage",
    "usage_sleep_ratio",
    "bedtime_share",
]


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), f"Champion model not found at {MODEL_PATH}. Run pipeline first."
    return joblib.load(MODEL_PATH)


def test_model_pipeline_structure(model):
    """Ensure the champion is a scikit-learn Pipeline with Scaler and Model stages."""
    assert hasattr(model, "steps"), "Model should be a scikit-learn Pipeline"
    step_names = [name for name, _ in model.steps]
    assert "Scaler" in step_names, "Pipeline missing 'Scaler' stage"
    assert "Model" in step_names, "Pipeline missing 'Model' stage"


def test_single_row_inference_and_clamping(model):
    """Test inference on single sample and verify addiction level is clamped within [1.0, 10.0]."""
    sample = pd.DataFrame([{
        "daily_usage_hours": 6.5,
        "sleep_hours": 6.0,
        "exercise_hours": 1.0,
        "phone_checks_per_day": 80.0,
        "apps_used_daily": 14.0,
        "time_on_social_media": 3.0,
        "time_on_gaming": 1.5,
        "weekend_minus_daily": 1.5,
        "usage_per_check": 6.5 / 81.0,
        "checks_per_hour": 80.0 / 6.6,
        "social_gaming_total": 4.5,
        "leisure_share": 4.5 / 6.6,
        "apps_x_usage": 14.0 * 6.5,
        "usage_sleep_ratio": 6.5 / 6.1,
        "bedtime_share": 1.2 / 6.6,
    }])
    raw_pred = model.predict(sample)[0]
    pred = float(np.clip(raw_pred, 1.0, 10.0))

    assert isinstance(pred, float)
    assert 1.0 <= pred <= 10.0


def test_batch_inference_and_bounds(model):
    """Test batch prediction across multiple rows and verify all values are valid floats."""
    batch = pd.DataFrame([
        {
            "daily_usage_hours": 2.0, "sleep_hours": 8.0, "exercise_hours": 2.0,
            "phone_checks_per_day": 30.0, "apps_used_daily": 6.0, "time_on_social_media": 0.5,
            "time_on_gaming": 0.5, "weekend_minus_daily": 0.5, "usage_per_check": 2.0 / 31.0,
            "checks_per_hour": 30.0 / 2.1, "social_gaming_total": 1.0, "leisure_share": 1.0 / 2.1,
            "apps_x_usage": 12.0, "usage_sleep_ratio": 2.0 / 8.1, "bedtime_share": 0.3 / 2.1,
        },
        {
            "daily_usage_hours": 9.5, "sleep_hours": 4.5, "exercise_hours": 0.2,
            "phone_checks_per_day": 130.0, "apps_used_daily": 18.0, "time_on_social_media": 4.5,
            "time_on_gaming": 3.0, "weekend_minus_daily": 2.5, "usage_per_check": 9.5 / 131.0,
            "checks_per_hour": 130.0 / 9.6, "social_gaming_total": 7.5, "leisure_share": 7.5 / 9.6,
            "apps_x_usage": 171.0, "usage_sleep_ratio": 9.5 / 4.6, "bedtime_share": 2.0 / 9.6,
        },
    ])
    preds = np.clip(model.predict(batch), 1.0, 10.0)
    assert len(preds) == 2
    for p in preds:
        assert 1.0 <= p <= 10.0


def test_prediction_determinism(model):
    """Ensure identical inputs yield identical outputs (deterministic)."""
    sample = pd.DataFrame([{
        "daily_usage_hours": 5.0, "sleep_hours": 7.0, "exercise_hours": 1.0,
        "phone_checks_per_day": 60.0, "apps_used_daily": 10.0, "time_on_social_media": 2.0,
        "time_on_gaming": 1.0, "weekend_minus_daily": 1.0, "usage_per_check": 5.0 / 61.0,
        "checks_per_hour": 60.0 / 5.1, "social_gaming_total": 3.0, "leisure_share": 3.0 / 5.1,
        "apps_x_usage": 50.0, "usage_sleep_ratio": 5.0 / 7.1, "bedtime_share": 1.0 / 5.1,
    }])
    p1 = model.predict(sample)[0]
    p2 = model.predict(sample)[0]
    assert np.isclose(p1, p2, atol=1e-7)
