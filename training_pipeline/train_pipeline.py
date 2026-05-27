"""
train_pipeline.py
=================
MLOps Training Pipeline — Malaria Outbreak Prediction

This script:
1. Loads engineered train/validation/test data from the feature store
2. Trains the selected Logistic Regression model
3. Evaluates it on validation and test sets
4. Logs the run to MLflow
5. Saves the trained model, scaler, and metadata locally

Usage:
    python train_pipeline.py
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_STORE = Path(os.getenv("FEATURE_STORE", "data/feature_store"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:mlruns/")

EXPERIMENT_NAME = "Malaria_Outbreak_Prediction"
RUN_NAME = "LogisticRegression_Balanced_Final"
MODEL_NAME = "MalariaOutbreakPredictor"

THRESHOLD = 0.50
TRAIN_YEARS = "2000-2016"
VAL_YEARS = "2017-2019"
TEST_YEARS = "2020-2021"

RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_csv(filename: str) -> pd.DataFrame | pd.Series:
    path = FEATURE_STORE / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    return df.squeeze() if df.shape[1] == 1 else df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load scaled train/validation/test splits for Logistic Regression."""
    log.info("Loading feature store data...")

    X_train = load_csv("X_train_scaled.csv")
    X_val = load_csv("X_val_scaled.csv")
    X_test = load_csv("X_test_scaled.csv")

    y_train = load_csv("y_train.csv")
    y_val = load_csv("y_val.csv")
    y_test = load_csv("y_test.csv")

    log.info(f"Train shape: {X_train.shape}")
    log.info(f"Val shape:   {X_val.shape}")
    log.info(f"Test shape:  {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_fitted_scaler():
    scaler_path = FEATURE_STORE / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing fitted scaler: {scaler_path}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return scaler


def compute_metrics(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    prefix: str,
) -> dict[str, float]:
    """Compute classification metrics using probability thresholding."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        f"{prefix}_auc_roc": round(roc_auc_score(y, y_prob), 6),
        f"{prefix}_f1": round(f1_score(y, y_pred, zero_division=0), 6),
        f"{prefix}_precision": round(precision_score(y, y_pred, zero_division=0), 6),
        f"{prefix}_recall": round(recall_score(y, y_pred, zero_division=0), 6),
        f"{prefix}_accuracy": round(accuracy_score(y, y_pred), 6),
        f"{prefix}_avg_precision": round(average_precision_score(y, y_prob), 6),
    }


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[LogisticRegression, dict]:
    """Train the final selected Logistic Regression model."""
    params = {
        "C": 1.0,
        "class_weight": "balanced",
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    }

    log.info("Training Logistic Regression...")
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    return model, params


def save_pickle(obj: object, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_metadata(
    *,
    params: dict,
    val_metrics: dict,
    test_metrics: dict,
    feature_cols: list[str],
    model_path: Path,
    scaler_path: Path,
) -> dict:
    metadata = {
        "best_model": "LogisticRegression",
        "model_name": MODEL_NAME,
        "run_name": RUN_NAME,
        "stage": "Staging",
        "threshold": THRESHOLD,
        "train_years": TRAIN_YEARS,
        "val_years": VAL_YEARS,
        "test_years": TEST_YEARS,
        "feature_cols": feature_cols,
        "model_params": params,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_file": str(model_path),
        "scaler_file": str(scaler_path),
    }

    metadata_path = MODEL_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Saved metadata to {metadata_path}")
    return metadata


def log_to_mlflow(
    *,
    model: LogisticRegression,
    params: dict,
    val_metrics: dict,
    test_metrics: dict,
    X_train: pd.DataFrame,
) -> None:
    """Log params, metrics, and model artifact to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    signature = infer_signature(X_train, model.predict_proba(X_train))

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_param("train_years", TRAIN_YEARS)
        mlflow.log_param("val_years", VAL_YEARS)
        mlflow.log_param("test_years", TEST_YEARS)
        mlflow.log_param("pipeline_stage", "training")

        for key, value in params.items():
            mlflow.log_param(key, value)

        mlflow.log_metrics({**val_metrics, **test_metrics})

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

    log.info("Logged run to MLflow successfully.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=" * 60)
    log.info("MALARIA TRAINING PIPELINE STARTED")
    log.info("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    feature_cols = X_train.columns.tolist()

    model, params = train_model(X_train, y_train)

    val_metrics = compute_metrics(model, X_val, y_val, THRESHOLD, prefix="val")
    test_metrics = compute_metrics(model, X_test, y_test, THRESHOLD, prefix="test")

    log.info(f"Validation metrics: {val_metrics}")
    log.info(f"Test metrics:       {test_metrics}")

    model_path = MODEL_DIR / "logistic_regression.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"

    save_pickle(model, model_path)
    log.info(f"Saved model to {model_path}")

    scaler = load_fitted_scaler()
    save_pickle(scaler, scaler_path)
    log.info(f"Saved scaler to {scaler_path}")

    log_to_mlflow(
        model=model,
        params=params,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        X_train=X_train,
    )

    metadata = save_metadata(
        params=params,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        feature_cols=feature_cols,
        model_path=model_path,
        scaler_path=scaler_path,
    )

    log.info("=" * 60)
    log.info("MALARIA TRAINING PIPELINE COMPLETE")
    log.info(f"Best model: {metadata['best_model']}")
    log.info(f"Threshold:  {metadata['threshold']}")
    log.info("=" * 60)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()