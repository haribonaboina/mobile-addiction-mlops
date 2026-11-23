"""Data Ingestion stage for Mobile Usage Addiction Prediction pipeline.

Validates the raw CSV dataset, standardizes column casing, fixes known typos,
removes non-predictive user identifiers (Name, Location), and persists clean ingested data.
"""
from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/phone_addiction_dataset.csv")
OUTPUT_PATH = Path("data/ingested/phone_addiction.csv")

EXPECTED_RAW_COLUMNS = [
    "Name",
    "Age",
    "Gender",
    "Location",
    "Daily_Usage_Hours",
    "Sleep_Hours",
    "Interllectual_Performance",
    "Social_Interactions",
    "Exercise_Hours",
    "Anxiety_Level",
    "Depression_Level",
    "Self_Esteem",
    "Screen_Time_Before_Bed",
    "Phone_Checks_Per_Day",
    "Apps_Used_Daily",
    "Time_on_Social_Media",
    "Time_on_Gaming",
    "Time_on_Education",
    "Phone_Usage_Purpose",
    "Family_Communication",
    "Weekend_Usage_Hours",
    "Addiction_Level",
]

IDENTIFIER_COLUMNS = ["name", "location"]
TARGET = "addiction_level"


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_PATH}. Please ensure phone_addiction_dataset.csv is present."
        )

    data = pd.read_csv(RAW_PATH)
    if data.empty:
        raise ValueError("The raw input dataset is empty.")

    # Validate raw schema
    missing_raw = sorted(set(EXPECTED_RAW_COLUMNS) - set(data.columns))
    if missing_raw:
        # Check case-insensitive match
        lower_cols = [c.lower() for c in data.columns]
        missing_lower = [c for c in EXPECTED_RAW_COLUMNS if c.lower() not in lower_cols]
        if missing_lower:
            raise ValueError(f"Missing expected raw columns: {missing_lower}")

    # Standardize column names to lowercase snake_case
    data.columns = [c.strip().lower() for c in data.columns]

    # Fix known domain typo from dataset schema
    if "interllectual_performance" in data.columns:
        data = data.rename(columns={"interllectual_performance": "intellectual_performance"})

    # Strip whitespace in text fields
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].astype(str).str.strip()

    # Drop non-predictive high-cardinality identifier columns
    cols_to_drop = [c for c in IDENTIFIER_COLUMNS if c in data.columns]
    if cols_to_drop:
        data = data.drop(columns=cols_to_drop)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Data ingestion completed: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Ingested columns: {list(data.columns)}")
    print(f"Saved ingested data to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
