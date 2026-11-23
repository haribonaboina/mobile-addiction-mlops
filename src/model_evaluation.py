"""Model evaluation stage for Mobile Usage Addiction Prediction pipeline.

Evaluates the production champion model and all candidates on unseen test data.
Clamps predictions within the target scale [1.0, 10.0], computes R2, MAE, RMSE,
Pearson correlation, and +/-1 pt accuracy rate, and logs artifacts to MLflow.
"""
from pathlib import Path
import json
import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml

X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")
SELECTION_PATH = Path("reports/model_selection.json")
METRICS_PATH = Path("reports/metrics.json")
DETAILS_PATH = Path("reports/evaluation_details.json")
COMPARISON_CSV_PATH = Path("reports/model_comparison.csv")
TARGET = "addiction_level"

MODEL_NAMES = [
    "LinearRegression",
    "DecisionTree",
    "RandomForest",
    "GradientBoosting",
]


def evaluate_regressor(model, X, y):
    raw_preds = model.predict(X)
    # Target scale is right-censored and bounded [1.0, 10.0]
    preds = np.clip(raw_preds, 1.0, 10.0)

    mae = float(mean_absolute_error(y, preds))
    mse = float(mean_squared_error(y, preds))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y, preds))
    pearson_corr = float(stats.pearsonr(y, preds)[0]) if len(y) > 1 else 1.0
    within_1pt = float((np.abs(y - preds) <= 1.0).mean())

    residuals = (y - preds).tolist()
    return {
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson_corr": round(pearson_corr, 4),
        "accuracy_within_1pt": round(within_1pt, 4),
        "mean_residual": round(float(np.mean(residuals)), 4),
        "std_residual": round(float(np.std(residuals)), 4),
    }


def main():
    if not SELECTION_PATH.exists():
        raise FileNotFoundError(f"Selection report not found: {SELECTION_PATH}. Run model_building first.")

    with open("params.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    with open(SELECTION_PATH, "r", encoding="utf-8") as file:
        selection = json.load(file)

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)[TARGET].values

    champion_info = selection["champion"]
    champion_name = champion_info["model_name"]
    champion_path = Path("models/model.pkl")

    champion = joblib.load(champion_path)
    criteria = champion_info["criteria"]
    policy = selection["selection_policy"]

    # Evaluate all candidate models on unseen test split
    all_results = {}
    for model_name in MODEL_NAMES:
        candidate_path = Path("models") / f"{model_name}.pkl"
        if candidate_path.exists():
            cand_model = joblib.load(candidate_path)
            all_results[model_name] = evaluate_regressor(cand_model, X_test, y_test)

    champion_metrics = evaluate_regressor(champion, X_test, y_test)
    test_r2_pct = champion_metrics["r2"] * 100.0

    # Flat numeric-only structure for clean DVC metrics tracking
    metrics = {
        "test_r2": champion_metrics["r2"],
        "test_mae": champion_metrics["mae"],
        "test_rmse": champion_metrics["rmse"],
        "test_pearson_corr": champion_metrics["pearson_corr"],
        "test_accuracy_within_1pt": champion_metrics["accuracy_within_1pt"],
        "champion_cv_r2": champion_info["best_cv_r2"],
        "champion_training_r2_pct": criteria["training_r2_pct"],
        "champion_validation_r2_pct": criteria["validation_r2_pct"],
        "champion_r2_gap_pct": criteria["r2_gap_pct"],
        "champion_model_size_mb": criteria["model_size_mb"],
        "champion_training_time_seconds": criteria["training_time_seconds"],
        "champion_prediction_time_ms": criteria["prediction_time_ms"],
        "test_vs_validation_gap_pct": round(criteria["validation_r2_pct"] - test_r2_pct, 2),
    }

    details = {
        "champion_model": champion_name,
        "selection_basis": (
            f"Multi-criteria optimization: Overfitting gap strictly below {policy['max_r2_gap_pct']}%, "
            f"highest validation R2, and weighted tie-breaker across size and latency."
        ),
        "selection_explanation": champion_info["explanation"],
        "selection_policy": policy,
        "champion_metrics": champion_metrics,
        "all_models_test_metrics": all_results,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    with open(DETAILS_PATH, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=4)

    # MLflow tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.get("experiment_name", "Mobile_Addiction_MLOps_Pipeline"))

    with mlflow.start_run(run_name="champion_evaluation"):
        mlflow.log_param("champion_model", champion_name)
        mlflow.log_metrics({
            "test_r2": champion_metrics["r2"],
            "test_mae": champion_metrics["mae"],
            "test_rmse": champion_metrics["rmse"],
            "test_within_1pt": champion_metrics["accuracy_within_1pt"],
            "champion_validation_r2_pct": criteria["validation_r2_pct"],
            "champion_r2_gap_pct": criteria["r2_gap_pct"],
            "champion_model_size_mb": criteria["model_size_mb"],
            "champion_train_time_sec": criteria["training_time_seconds"],
            "champion_pred_time_ms": criteria["prediction_time_ms"],
        })
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(DETAILS_PATH))
        mlflow.log_artifact(str(SELECTION_PATH))
        if COMPARISON_CSV_PATH.exists():
            mlflow.log_artifact(str(COMPARISON_CSV_PATH))

    print("\n" + "=" * 60)
    print(f"CHAMPION TEST EVALUATION: {champion_name}")
    print(f"Test R2    : {champion_metrics['r2']:.4f}")
    print(f"Test MAE   : {champion_metrics['mae']:.4f}")
    print(f"Test RMSE  : {champion_metrics['rmse']:.4f}")
    print(f"Within ±1pt: {champion_metrics['accuracy_within_1pt']*100:.1f}%")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
