"""
train_pipeline.py
=================
MLOps Training Pipeline — Malaria Outbreak Prediction
Trains Logistic Regression + XGBoost, compares them,
promotes the winner to MLflow Staging.

Usage:
    python train_pipeline.py

Outputs:
    models/logistic_regression.pkl
    models/xgboost_model.pkl
    models/scaler.pkl
    models/model_metadata.json
    MLflow runs logged + best model promoted to Staging
"""

import os, json, pickle, logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score,
                              precision_score, recall_score,
                              average_precision_score)
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient

# ── Paths ─────────────────────────────────────────────────────────
FEATURE_STORE = r"C:\Users\Likhita Kolli\OneDrive - Hochschule Luzern\SEM_2\AI\Data\Processed\feature_store"
MODEL_DIR     = r"C:\Users\Likhita Kolli\OneDrive - Hochschule Luzern\SEM_2\AI\models"
MLFLOW_URI    = r"file:///C:/Users/Likhita Kolli/OneDrive - Hochschule Luzern/SEM_2/AI/mlruns"
EXPERIMENT    = "Malaria_Outbreak_Prediction"
MODEL_NAME    = "MalariaOutbreakPredictor"
THRESHOLD     = 0.40   # tuned in LR notebook; applied to both models

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

os.makedirs(MODEL_DIR, exist_ok=True)


# ── Step 1: Load data ─────────────────────────────────────────────
def load_data():
    log.info("Loading feature store...")
    def load(f): return pd.read_csv(os.path.join(FEATURE_STORE, f))

    X_train       = load("X_train_scaled.csv")
    X_train_smote = load("X_train_smote.csv")
    X_val         = load("X_val_scaled.csv")
    X_test        = load("X_test_scaled.csv")
    y_train       = load("y_train.csv").squeeze()
    y_train_smote = load("y_train_smote.csv").squeeze()
    y_val         = load("y_val.csv").squeeze()
    y_test        = load("y_test.csv").squeeze()

    log.info(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return (X_train, X_train_smote, X_val, X_test,
            y_train, y_train_smote, y_val, y_test)


# ── Step 2: Evaluate helper ───────────────────────────────────────
def compute_metrics(model, X, y, threshold=THRESHOLD, prefix="val"):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        f"{prefix}_auc_roc"      : round(roc_auc_score(y, y_prob), 4),
        f"{prefix}_f1"           : round(f1_score(y, y_pred, zero_division=0), 4),
        f"{prefix}_precision"    : round(precision_score(y, y_pred, zero_division=0), 4),
        f"{prefix}_recall"       : round(recall_score(y, y_pred, zero_division=0), 4),
        f"{prefix}_avg_precision": round(average_precision_score(y, y_prob), 4),
    }


# ── Step 3: Train Logistic Regression ────────────────────────────
def train_logistic_regression(X_train_smote, y_train_smote,
                               X_val, y_val, X_test, y_test):
    log.info("Training Logistic Regression...")

    params = dict(C=1.0, penalty="l2", solver="lbfgs",
                  max_iter=1000, random_state=42)
    model = LogisticRegression(**params)
    model.fit(X_train_smote, y_train_smote)

    val_metrics  = compute_metrics(model, X_val,  y_val,  prefix="val")
    test_metrics = compute_metrics(model, X_test, y_test, prefix="test")

    log.info(f"LR  Val  AUC={val_metrics['val_auc_roc']}  "
             f"F1={val_metrics['val_f1']}")
    log.info(f"LR  Test AUC={test_metrics['test_auc_roc']}  "
             f"F1={test_metrics['test_f1']}")

    # Save locally
    path = os.path.join(MODEL_DIR, "logistic_regression.pkl")
    with open(path, "wb") as f: pickle.dump(model, f)

    return model, params, val_metrics, test_metrics


# ── Step 4: Train XGBoost ─────────────────────────────────────────
def train_xgboost(X_train_smote, y_train_smote,
                  X_val, y_val, X_test, y_test):
    log.info("Training XGBoost...")

    # XGBoost uses unscaled data — reload raw SMOTE split
    fs = FEATURE_STORE
    X_tr_raw = pd.read_csv(os.path.join(fs, "X_train_smote.csv"))
    y_tr_raw = pd.read_csv(os.path.join(fs, "y_train_smote.csv")).squeeze()
    X_v_raw  = pd.read_csv(os.path.join(fs, "X_val.csv"))
    X_te_raw = pd.read_csv(os.path.join(fs, "X_test.csv"))

    params = dict(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 5,
        gamma             = 0.1,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = 42,
        n_jobs            = -1,
    )
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_tr_raw, y_tr_raw,
        eval_set=[(X_v_raw, y_val)],
        verbose=False,
    )

    val_metrics  = compute_metrics(model, X_v_raw,  y_val,  prefix="val")
    test_metrics = compute_metrics(model, X_te_raw, y_test, prefix="test")

    log.info(f"XGB Val  AUC={val_metrics['val_auc_roc']}  "
             f"F1={val_metrics['val_f1']}")
    log.info(f"XGB Test AUC={test_metrics['test_auc_roc']}  "
             f"F1={test_metrics['test_f1']}")

    path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    with open(path, "wb") as f: pickle.dump(model, f)

    return model, params, val_metrics, test_metrics, X_tr_raw


# ── Step 5: Log to MLflow & promote winner to Staging ─────────────
def log_and_promote(models_info, X_train_smote, X_tr_xgb):
    """
    models_info: list of dicts with keys:
        name, model, params, val_metrics, test_metrics, X_train_ref
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient(tracking_uri=MLFLOW_URI)

    run_ids   = {}
    val_aucs  = {}

    for info in models_info:
        name         = info["name"]
        model        = info["model"]
        params       = info["params"]
        val_metrics  = info["val_metrics"]
        test_metrics = info["test_metrics"]
        X_ref        = info["X_train_ref"]

        log.info(f"Logging MLflow run: {name}")

        with mlflow.start_run(run_name=name) as run:
            # ── Log params
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("threshold",   THRESHOLD)
            mlflow.log_param("train_years", "2000-2016")
            mlflow.log_param("val_years",   "2017-2019")
            mlflow.log_param("test_years",  "2020-2022")
            mlflow.log_param("pipeline_stage", "training")

            # ── Log val + test metrics
            mlflow.log_metrics({**val_metrics, **test_metrics})

            # ── Log model with signature
            sig = infer_signature(X_ref, model.predict_proba(X_ref))

            if "XGBoost" in name:
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="model",
                    signature=sig,
                    registered_model_name=MODEL_NAME,
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=sig,
                    registered_model_name=MODEL_NAME,
                )

            run_ids[name]  = run.info.run_id
            val_aucs[name] = val_metrics["val_auc_roc"]
            log.info(f"  Run ID: {run.info.run_id}")

    # ── Pick winner by val AUC ────────────────────────────────────
    best_name = max(val_aucs, key=val_aucs.get)
    log.info(f"Winner: {best_name}  (AUC={val_aucs[best_name]})")

    # ── Transition best model version to Staging ──────────────────
    # Get latest version of the registered model
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    # Find the version that matches the best run
    best_run_id = run_ids[best_name]
    best_version = None
    for v in versions:
        if v.run_id == best_run_id:
            best_version = v.version
            break

    if best_version:
        client.transition_model_version_stage(
            name    = MODEL_NAME,
            version = best_version,
            stage   = "Staging",
            archive_existing_versions = True,  # archive others
        )
        log.info(f"Model v{best_version} promoted to STAGING ✅")
    else:
        log.warning("Could not find version to promote.")

    return best_name, run_ids, val_aucs, best_version


# ── Step 6: Save scaler + metadata ───────────────────────────────
def save_metadata(best_name, val_aucs, test_metrics_map,
                  best_version, feature_cols):
    # Rebuild and save scaler
    scaler_params = pd.read_csv(
        os.path.join(FEATURE_STORE, "scaler_params.csv"))
    scaler = StandardScaler()
    scaler.mean_  = scaler_params["mean"].values
    scaler.scale_ = scaler_params["std"].values
    scaler.n_features_in_ = len(scaler_params)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "best_model"      : best_name,
        "model_name"      : MODEL_NAME,
        "model_version"   : str(best_version),
        "stage"           : "Staging",
        "threshold"       : THRESHOLD,
        "feature_cols"    : feature_cols,
        "val_aucs"        : val_aucs,
        "test_metrics"    : test_metrics_map,
        "train_years"     : "2000-2016",
        "val_years"       : "2017-2019",
        "test_years"      : "2020-2022",
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Scaler + metadata saved ✅")
    return metadata


# ── MAIN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 55)
    log.info("TRAINING PIPELINE STARTED")
    log.info("=" * 55)

    (X_train, X_train_smote, X_val, X_test,
     y_train, y_train_smote, y_val, y_test) = load_data()

    feature_cols = X_train.columns.tolist()

    lr_model,  lr_params,  lr_val,  lr_test  = train_logistic_regression(
        X_train_smote, y_train_smote, X_val, y_val, X_test, y_test)

    xgb_model, xgb_params, xgb_val, xgb_test, X_tr_xgb = train_xgboost(
        X_train_smote, y_train_smote, X_val, y_val, X_test, y_test)

    models_info = [
        {"name": "LogisticRegression", "model": lr_model,
         "params": lr_params, "val_metrics": lr_val,
         "test_metrics": lr_test, "X_train_ref": X_train_smote},
        {"name": "XGBoost", "model": xgb_model,
         "params": xgb_params, "val_metrics": xgb_val,
         "test_metrics": xgb_test, "X_train_ref": X_tr_xgb},
    ]

    best_name, run_ids, val_aucs, best_version = log_and_promote(
        models_info, X_train_smote, X_tr_xgb)

    test_metrics_map = {
        "LogisticRegression": lr_test,
        "XGBoost"           : xgb_test,
    }

    metadata = save_metadata(best_name, val_aucs,
                             test_metrics_map, best_version, feature_cols)

    log.info("=" * 55)
    log.info("TRAINING PIPELINE COMPLETE")
    log.info(f"Best model  : {best_name}")
    log.info(f"Val AUC     : {val_aucs[best_name]}")
    log.info(f"MLflow stage: Staging (v{best_version})")
    log.info(f"Models saved: {MODEL_DIR}")
    log.info("=" * 55)
    print(json.dumps(metadata, indent=2))
