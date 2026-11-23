"""Feature Engineering stage for Mobile Usage Addiction Prediction pipeline.

Calculates behavioral ratios, interaction terms, and temporal usage features,
selects the optimal feature subset (15 numeric features), and separates features from target.
"""
from pathlib import Path
import pandas as pd

TRAIN_INPUT = Path("data/processed/train.csv")
TEST_INPUT = Path("data/processed/test.csv")

X_TRAIN_PATH = Path("data/processed/X_train.csv")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")

TARGET = "addiction_level"

# 15 optimal predictive features determined by correlation filter, RF importance, and RFE
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


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute domain-informed behavioral and temporal interaction features."""
    df = data.copy()

    # Temporal differential
    df["weekend_minus_daily"] = df["weekend_usage_hours"] - df["daily_usage_hours"]

    # Frequency & intensity
    df["usage_per_check"] = df["daily_usage_hours"] / (df["phone_checks_per_day"] + 1.0)
    df["checks_per_hour"] = df["phone_checks_per_day"] / (df["daily_usage_hours"] + 0.1)

    # Activity allocation & share
    df["social_gaming_total"] = df["time_on_social_media"] + df["time_on_gaming"]
    df["leisure_share"] = df["social_gaming_total"] / (df["daily_usage_hours"] + 0.1)
    df["productive_share"] = df["time_on_education"] / (df["daily_usage_hours"] + 0.1)

    # Multiplicative interaction
    df["apps_x_usage"] = df["apps_used_daily"] * df["daily_usage_hours"]

    # Sleep & biological impact
    df["sleep_deficit"] = (8.0 - df["sleep_hours"]).clip(lower=0.0)
    df["usage_sleep_ratio"] = df["daily_usage_hours"] / (df["sleep_hours"] + 0.1)
    df["bedtime_share"] = df["screen_time_before_bed"] / (df["daily_usage_hours"] + 0.1)

    # Wellbeing composite
    if {"self_esteem", "anxiety_level", "depression_level"}.issubset(df.columns):
        df["wellbeing_score"] = df["self_esteem"] - df["anxiety_level"] - df["depression_level"]

    # Non-linear intensity
    df["usage_sq"] = df["daily_usage_hours"] ** 2

    return df


def transform_split(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = pd.read_csv(input_path)

    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' missing in {input_path}")

    # Generate all engineered features
    data_engineered = engineer_features(data)

    missing_selected = [c for c in SELECTED_FEATURES if c not in data_engineered.columns]
    if missing_selected:
        raise ValueError(f"Missing engineered feature(s): {missing_selected}")

    X = data_engineered[SELECTED_FEATURES].copy()
    y = data_engineered[TARGET].copy()

    return X, y


def main():
    X_train, y_train = transform_split(TRAIN_INPUT)
    X_test, y_test = transform_split(TEST_INPUT)

    X_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_frame(name=TARGET).to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_frame(name=TARGET).to_csv(Y_TEST_PATH, index=False)

    print("Feature engineering completed successfully.")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")
    print(f"Saved split files under: {X_TRAIN_PATH.parent}")


if __name__ == "__main__":
    main()
