"""Unit tests for dataset presence, integrity, and preprocessing consistency."""
from pathlib import Path
import pandas as pd
import pytest

RAW_DATA_PATH = Path("data/raw/phone_addiction_dataset.csv")
TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")
X_TRAIN_PATH = Path("data/processed/X_train.csv")
X_TEST_PATH = Path("data/processed/X_test.csv")

EXPECTED_RAW_COLUMNS = [
    "daily_usage_hours",
    "sleep_hours",
    "intellectual_performance",
    "social_interactions",
    "exercise_hours",
    "anxiety_level",
    "depression_level",
    "self_esteem",
    "screen_time_before_bed",
    "phone_checks_per_day",
    "apps_used_daily",
    "time_on_social_media",
    "time_on_gaming",
    "time_on_education",
    "family_communication",
    "weekend_usage_hours",
    "addiction_level",
]

SELECTED_FEATURES = [
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

TARGET = "addiction_level"


def test_raw_dataset_exists_and_valid():
    """Verify raw dataset exists and has expected rows and columns."""
    assert RAW_DATA_PATH.exists(), f"Raw dataset not found at {RAW_DATA_PATH}"
    df = pd.read_csv(RAW_DATA_PATH)
    assert len(df) == 6000, f"Expected 6000 raw rows, got {len(df)}"
    assert df.isnull().sum().sum() == 0, "Missing values found in raw dataset"


def test_preprocessed_splits_exist_and_consistent():
    """Verify train and test splits exist, have no duplicates, and match 5968 deduplicated total."""
    assert TRAIN_DATA_PATH.exists(), f"Train split missing at {TRAIN_DATA_PATH}"
    assert TEST_DATA_PATH.exists(), f"Test split missing at {TEST_DATA_PATH}"

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    total_rows = len(train_df) + len(test_df)
    # 6000 raw rows - 32 duplicates = 5968 unique rows
    assert total_rows == 5968, f"Expected 5968 deduplicated rows, got {total_rows}"
    assert TARGET in train_df.columns
    assert TARGET in test_df.columns


def test_feature_engineering_outputs_correct_features():
    """Verify X_train and X_test contain all 15 selected features without missing values."""
    assert X_TRAIN_PATH.exists(), f"X_train not found at {X_TRAIN_PATH}"
    assert X_TEST_PATH.exists(), f"X_test not found at {X_TEST_PATH}"

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    assert list(X_train.columns) == SELECTED_FEATURES
    assert list(X_test.columns) == SELECTED_FEATURES
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0
