"""Data Preprocessing stage for Mobile Usage Addiction Prediction pipeline.

Performs deduplication, extreme outlier capping (3x IQR), and stratified/random
train-test splitting based on configuration in params.yaml.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("data/ingested/phone_addiction.csv")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
TARGET = "addiction_level"


def cap_extreme_outliers(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """Cap extreme numerical outliers using 3x IQR rule without dropping valid rows."""
    df_capped = df.copy()
    for col in num_cols:
        q1 = df_capped[col].quantile(0.25)
        q3 = df_capped[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3.0 * iqr
        upper_bound = q3 + 3.0 * iqr
        df_capped[col] = df_capped[col].clip(lower_bound, upper_bound)
    return df_capped


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Ingested dataset not found at {INPUT_PATH}. Run data_ingestion first.")

    data = pd.read_csv(INPUT_PATH)

    # 1. Deduplication
    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before - len(data)
    print(f"Removed {duplicates_removed} duplicate records ({len(data)} unique rows remaining).")

    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' is missing from ingested dataset.")

    # 2. Outlier treatment (3x IQR capping for numeric features)
    numeric_features = [c for c in data.select_dtypes(include=np.number).columns if c != TARGET]
    data = cap_extreme_outliers(data, numeric_features)

    # 3. Train-test split
    test_size = float(params.get("test_size", 0.2))
    random_state = int(params.get("random_state", 42))

    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print(f"Preprocessed train set: {train.shape}")
    print(f"Preprocessed test set : {test.shape}")
    print(f"Saved splits to: {TRAIN_PATH} and {TEST_PATH}")


if __name__ == "__main__":
    main()
