"""Model building with automated multi-criteria champion selection for regression.

Step 1: Train candidate regressors (LinearRegression, DecisionTree, RandomForest, GradientBoosting)
        with Optuna hyperparameter optimization & 5-fold Cross-Validation.
Step 2: Measure six production criteria per model:
        model size (MB), training time (s), prediction latency (ms),
        training R2 (%), validation R2 (%), and overfitting gap (pp).
Step 3: Apply automated multi-criteria champion selection policy from params.yaml:
        - hard filter: overfitting gap strictly below max_r2_gap_pct
        - validation R2 within tolerance of the highest eligible score
        - tie-breaking by weighted model size, train time, and inference latency.
Step 4: Save all candidate models in models/<Name>.pkl and the champion in models/model.pkl.
Step 5: Export reports/model_selection.json, reports/model_comparison.csv, reports/model_comparison.md.
"""
from pathlib import Path
import json
import os
import shutil
import time
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import yaml

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

X_TRAIN_PATH = Path("data/processed/X_train.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
SELECTION_PATH = REPORTS_DIR / "model_selection.json"
COMPARISON_CSV_PATH = REPORTS_DIR / "model_comparison.csv"
COMPARISON_MD_PATH = REPORTS_DIR / "model_comparison.md"
TARGET = "addiction_level"

MODEL_NAMES = [
    "LinearRegression",
    "DecisionTree",
    "RandomForest",
    "GradientBoosting",
]

DISPLAY_NAMES = {
    "LinearRegression": "Linear Regression",
    "DecisionTree": "Decision Tree",
    "RandomForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
}


def suggest_params(model_name: str, trial: optuna.Trial) -> dict:
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax"])

    if model_name == "LinearRegression":
        params = {}
    elif model_name == "DecisionTree":
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        }
    elif model_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.8, 1.0]),
        }
    elif model_name == "GradientBoosting":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return {"scaler_type": scaler_type, **params}


def build_pipeline(model_name: str, params: dict, random_state: int = 42) -> Pipeline:
    scaler = (
        StandardScaler()
        if params.get("scaler_type", "standard") == "standard"
        else MinMaxScaler()
    )
    model_params = {k: v for k, v in params.items() if k != "scaler_type"}

    if model_name == "LinearRegression":
        model = LinearRegression(**model_params)
    elif model_name == "DecisionTree":
        model = DecisionTreeRegressor(random_state=random_state, **model_params)
    elif model_name == "RandomForest":
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params)
    elif model_name == "GradientBoosting":
        model = GradientBoostingRegressor(random_state=random_state, **model_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return Pipeline([("Scaler", scaler), ("Model", model)])


def objective(trial, model_name, X, y, cv, gap_limit, use_penalty, penalty_weight, random_state):
    params = suggest_params(model_name, trial)
    pipeline = build_pipeline(model_name, params, random_state)

    cv_res = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="r2",
        return_train_score=True,
        n_jobs=-1,
    )

    val_r2 = float(np.mean(cv_res["test_score"]))
    train_r2 = float(np.mean(cv_res["train_score"]))
    gap_pp = (train_r2 - val_r2) * 100.0

    score = val_r2
    if use_penalty and gap_pp >= gap_limit:
        excess = (gap_pp - gap_limit) / 100.0
        score -= penalty_weight * excess

    trial.set_user_attr("train_r2", train_r2)
    trial.set_user_attr("val_r2", val_r2)
    trial.set_user_attr("gap_pp", gap_pp)
    return score


def measure_criteria(pipeline, X, y, repeats_train, repeats_pred, n_pred_samples, model_path):
    train_times = []
    for _ in range(repeats_train):
        pipe_clone = joblib.load(model_path)
        t0 = time.perf_counter()
        pipe_clone.fit(X, y)
        train_times.append(time.perf_counter() - t0)
    train_time_sec = float(np.median(train_times))

    pipeline.fit(X, y)
    sample_rows = X.iloc[: min(n_pred_samples, len(X))].copy()
    pred_times = []
    for _ in range(repeats_pred):
        t0 = time.perf_counter()
        for i in range(len(sample_rows)):
            _ = pipeline.predict(sample_rows.iloc[[i]])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        pred_times.append(elapsed_ms)
    pred_time_ms = float(np.median(pred_times))

    model_size_mb = float(os.path.getsize(model_path) / (1024 * 1024))
    return {
        "model_size_mb": round(model_size_mb, 4),
        "training_time_seconds": round(train_time_sec, 4),
        "prediction_time_ms": round(pred_time_ms, 4),
    }


def _normalise(values: dict) -> dict:
    vals = list(values.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def select_champion(results: dict, cfg: dict):
    gap_limit = float(cfg.get("max_r2_gap_pct", 5.0))
    tolerance = float(cfg.get("r2_tolerance_pct", 1.0))
    weights = cfg["weights"]

    for r in results.values():
        r["eligible"] = bool(r["r2_gap_pct"] < gap_limit)

    eligible = {n: r for n, r in results.items() if r["eligible"]}
    fallback_used = False

    if not eligible:
        mode = cfg.get("fallback_if_none_eligible", "smallest_gap")
        if mode == "error":
            raise RuntimeError(f"No model has an R2 gap strictly below {gap_limit}%.")
        warnings.warn("No eligible model; falling back to smallest R2 gap.")
        fallback_used = True
        eligible = {n: r for n, r in results.items()}

    best_val = max(r["validation_r2_pct"] for r in eligible.values())
    if fallback_used:
        smallest_gap = min(r["r2_gap_pct"] for r in eligible.values())
        finalists = {
            n: r for n, r in eligible.items()
            if r["r2_gap_pct"] - smallest_gap < 1e-9
        }
    else:
        finalists = {
            n: r for n, r in eligible.items()
            if best_val - r["validation_r2_pct"] <= tolerance + 1e-9
        }

    norm_size = _normalise({n: r["model_size_mb"] for n, r in finalists.items()})
    norm_train = _normalise({n: r["training_time_seconds"] for n, r in finalists.items()})
    norm_pred = _normalise({n: r["prediction_time_ms"] for n, r in finalists.items()})

    for n, r in finalists.items():
        r["selection_cost"] = float(
            weights["model_size_mb"] * norm_size[n]
            + weights["training_time_seconds"] * norm_train[n]
            + weights["prediction_time_ms"] * norm_pred[n]
        )

    ranked = sorted(
        finalists,
        key=lambda n: (
            round(finalists[n]["selection_cost"], 12),
            -finalists[n]["validation_r2_pct"],
            n,
        ),
    )
    champion = ranked[0]

    for name, r in results.items():
        if name == champion:
            r["status"] = "CHAMPION"
        elif name in finalists:
            r["status"] = "FINALIST"
        elif r["eligible"]:
            r["status"] = "ELIGIBLE"
        else:
            r["status"] = "DISQUALIFIED_GAP"

    c = results[champion]
    explanation = (
        f"{DISPLAY_NAMES[champion]} is selected as champion: validation R2 = "
        f"{c['validation_r2_pct']:.2f}%, gap = {c['r2_gap_pct']:.2f}pp (limit {gap_limit}%), "
        f"model size = {c['model_size_mb']:.3f} MB, train time = {c['training_time_seconds']:.2f}s, "
        f"prediction latency = {c['prediction_time_ms']:.2f}ms (selection cost = {c['selection_cost']:.4f})."
    )
    return champion, explanation, fallback_used


def write_reports(results: dict, champion: str, explanation: str, policy: dict):
    rows = []
    for name in MODEL_NAMES:
        r = results[name]
        rows.append({
            "Model": DISPLAY_NAMES[name],
            "Validation R2 (%)": r["validation_r2_pct"],
            "Training R2 (%)": r["training_r2_pct"],
            "R2 Gap (pp)": r["r2_gap_pct"],
            "Model Size (MB)": r["model_size_mb"],
            "Train Time (s)": r["training_time_seconds"],
            "Predict Time (ms)": r["prediction_time_ms"],
            "Eligible": "Yes" if r["eligible"] else "No",
            "Status": r["status"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(COMPARISON_CSV_PATH, index=False)

    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    md = "\n".join(
        ["# Model Comparison", "", header, sep, *body, "",
         f"**Champion:** {DISPLAY_NAMES[champion]}", "",
         f"> **Selection Rationale:** {explanation}", ""]
    )
    COMPARISON_MD_PATH.write_text(md, encoding="utf-8")
    return df


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    sel_cfg = config["selection"]
    meas_cfg = config["measurement"]
    gap_limit = float(sel_cfg.get("max_r2_gap_pct", 5.0))
    use_penalty = bool(sel_cfg.get("gap_penalty_in_search", True))
    penalty_weight = float(sel_cfg.get("gap_penalty_weight", 1.0))

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)[TARGET].values

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    random_state = int(config.get("random_state", 42))
    cv_folds = int(config.get("cv_folds", 5))
    n_trials = int(config.get("n_trials", 15))

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = {}

    print(f"Training and evaluating {len(MODEL_NAMES)} candidate models...")

    for model_name in MODEL_NAMES:
        print(f"\n---> Optimizing {DISPLAY_NAMES[model_name]} ({n_trials} trials)...")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )

        # For LinearRegression, only 2 scaler choices; reduce trials
        trials_count = 2 if model_name == "LinearRegression" else n_trials

        study.optimize(
            lambda trial: objective(
                trial,
                model_name,
                X_train,
                y_train,
                cv,
                gap_limit,
                use_penalty,
                penalty_weight,
                random_state,
            ),
            n_trials=trials_count,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_pipeline = build_pipeline(model_name, best_params, random_state)
        best_pipeline.fit(X_train, y_train)

        # Measure train and CV validation metrics
        cv_res = cross_validate(
            best_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="r2",
            return_train_score=True,
            n_jobs=-1,
        )
        train_r2 = float(np.mean(cv_res["train_score"]))
        val_r2 = float(np.mean(cv_res["test_score"]))
        gap_pp = (train_r2 - val_r2) * 100.0

        model_path = MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(best_pipeline, model_path)

        criteria = measure_criteria(
            best_pipeline,
            X_train,
            y_train,
            repeats_train=meas_cfg.get("training_time_repeats", 3),
            repeats_pred=meas_cfg.get("prediction_repeats", 5),
            n_pred_samples=meas_cfg.get("prediction_n_samples", 50),
            model_path=model_path,
        )

        results[model_name] = {
            "best_params": best_params,
            "training_r2_pct": round(train_r2 * 100.0, 2),
            "validation_r2_pct": round(val_r2 * 100.0, 2),
            "r2_gap_pct": round(gap_pp, 2),
            "best_cv_r2": round(val_r2, 4),
            **criteria,
        }
        print(f"  Validation R2: {val_r2*100:.2f}%, Train R2: {train_r2*100:.2f}%, Gap: {gap_pp:.2f}pp")

    # Multi-criteria champion selection
    champion, explanation, fallback = select_champion(results, sel_cfg)
    write_reports(results, champion, explanation, sel_cfg)

    # Save champion to models/model.pkl
    champion_path = MODELS_DIR / "model.pkl"
    shutil.copyfile(MODELS_DIR / f"{champion}.pkl", champion_path)

    selection_meta = {
        "champion": {
            "model_name": champion,
            "display_name": DISPLAY_NAMES[champion],
            "best_cv_r2": results[champion]["best_cv_r2"],
            "criteria": {
                "validation_r2_pct": results[champion]["validation_r2_pct"],
                "training_r2_pct": results[champion]["training_r2_pct"],
                "r2_gap_pct": results[champion]["r2_gap_pct"],
                "model_size_mb": results[champion]["model_size_mb"],
                "training_time_seconds": results[champion]["training_time_seconds"],
                "prediction_time_ms": results[champion]["prediction_time_ms"],
            },
            "best_params": results[champion]["best_params"],
            "explanation": explanation,
        },
        "selection_policy": {
            "max_r2_gap_pct": gap_limit,
            "r2_tolerance_pct": float(sel_cfg.get("r2_tolerance_pct", 1.0)),
            "weights": sel_cfg["weights"],
            "fallback_used": fallback,
        },
        **{n: results[n] for n in MODEL_NAMES},
    }

    with open(SELECTION_PATH, "w", encoding="utf-8") as f:
        json.dump(selection_meta, f, indent=4)

    print("\n" + "=" * 60)
    print(f"CHAMPION MODEL SELECTED: {DISPLAY_NAMES[champion]}")
    print(f"Rationale: {explanation}")
    print(f"Champion artifact saved to: {champion_path}")
    print(f"Model comparison saved to: {COMPARISON_CSV_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
