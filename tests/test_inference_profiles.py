"""Inference test across diverse behavioral profiles."""
import joblib
import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import engineer_features, SELECTED_FEATURES


def test_diverse_user_profiles():
    model = joblib.load("models/model.pkl")

    low_user = {
        "daily_usage_hours": 1.5, "weekend_usage_hours": 2.0, "screen_time_before_bed": 0.2,
        "phone_checks_per_day": 25, "time_on_social_media": 0.5, "time_on_gaming": 0.2,
        "time_on_education": 0.8, "apps_used_daily": 6, "sleep_hours": 8.0,
        "exercise_hours": 2.0, "anxiety_level": 2, "depression_level": 2, "self_esteem": 8,
    }
    high_user = {
        "daily_usage_hours": 10.5, "weekend_usage_hours": 12.0, "screen_time_before_bed": 2.5,
        "phone_checks_per_day": 140, "time_on_social_media": 5.0, "time_on_gaming": 3.5,
        "time_on_education": 0.2, "apps_used_daily": 22, "sleep_hours": 4.5,
        "exercise_hours": 0.2, "anxiety_level": 9, "depression_level": 8, "self_esteem": 3,
    }

    df_low = engineer_features(pd.DataFrame([low_user]))[SELECTED_FEATURES]
    df_high = engineer_features(pd.DataFrame([high_user]))[SELECTED_FEATURES]

    pred_low = float(np.clip(model.predict(df_low)[0], 1.0, 10.0))
    pred_high = float(np.clip(model.predict(df_high)[0], 1.0, 10.0))

    assert 1.0 <= pred_low <= 5.0, f"Low user expected <= 5.0, got {pred_low}"
    assert 7.0 <= pred_high <= 10.0, f"High user expected >= 7.0, got {pred_high}"
    assert pred_high > pred_low, "High user addiction score must exceed low user score"
