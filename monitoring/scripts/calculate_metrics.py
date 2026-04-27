"""
calculate_metrics.py
--------------------
Calculate monitoring metrics for the latest malaria batch and store them in PostgreSQL.

What it computes:
- data drift (KS test against reference data)
- batch accuracy / precision / recall / f1
- prediction distribution
- mean predicted probability
- mean latency

The metrics table stores one row per processed batch, which is ideal for Grafana.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import psycopg2
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = PROJECT_ROOT  / "monitoring" / "data" / "reference.csv"
BATCH_GLOB = str(PROJECT_ROOT  / "monitoring" / "data" / "current_batches" / "*.csv")

FEATURES = [
    "malaria_incidence",
    "precipitation_mm",
    "pop_density",
    "gdp_per_capita",
    "temp_annual_mean_c",
    "temp_growing_season_mean_c",
]


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "test"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "example"),
    )


def main() -> None:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError("reference.csv not found. Run prepare_reference.py first.")

    batch_files = glob.glob(BATCH_GLOB)
    if not batch_files:
        raise FileNotFoundError("No batch files found. Run generate_batch.py first.")

    latest_file = max(batch_files, key=os.path.getmtime)
    batch_id = Path(latest_file).stem

    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(latest_file)

    drifted_features = 0
    drift_details = {}

    for feature in FEATURES:
        _, p_value = ks_2samp(reference[feature], current[feature])
        drift_flag = p_value < 0.05
        drift_details[feature] = {
            "p_value": float(p_value),
            "drifted": bool(drift_flag),
        }
        if drift_flag:
            drifted_features += 1

    share_drifted = drifted_features / len(FEATURES)

    y_true = current["true_label"]
    y_pred = current["outbreak_prediction"]

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    pred_share = current["outbreak_prediction"].value_counts(normalize=True).to_dict()
    pred_low_risk = pred_share.get(0, 0.0)
    pred_high_risk = pred_share.get(1, 0.0)

    avg_probability = float(current["outbreak_probability"].mean())
    avg_latency = float(current["inference_latency_ms"].mean())
    drift_mode = str(current["drift_mode"].mode().iloc[0]) if "drift_mode" in current.columns else "unknown"

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS monitoring_metrics (
                    timestamp TIMESTAMP,
                    batch_id TEXT PRIMARY KEY,
                    batch_size INT,
                    drift_mode TEXT,
                    num_drifted_features INT,
                    share_drifted_features FLOAT,
                    accuracy FLOAT,
                    precision FLOAT,
                    recall FLOAT,
                    f1_macro FLOAT,
                    avg_outbreak_probability FLOAT,
                    pred_low_risk_share FLOAT,
                    pred_high_risk_share FLOAT,
                    avg_latency_ms FLOAT,
                    avg_malaria_incidence FLOAT,
                    avg_precipitation_mm FLOAT,
                    avg_temp_annual_mean_c FLOAT
                );
                """
            )

            cur.execute(
                """
                INSERT INTO monitoring_metrics (
                    timestamp,
                    batch_id,
                    batch_size,
                    drift_mode,
                    num_drifted_features,
                    share_drifted_features,
                    accuracy,
                    precision,
                    recall,
                    f1_macro,
                    avg_outbreak_probability,
                    pred_low_risk_share,
                    pred_high_risk_share,
                    avg_latency_ms,
                    avg_malaria_incidence,
                    avg_precipitation_mm,
                    avg_temp_annual_mean_c
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (batch_id) DO NOTHING
                """,
                (
                    datetime.now(UTC),
                    batch_id,
                    int(len(current)),
                    drift_mode,
                    int(drifted_features),
                    float(share_drifted),
                    float(accuracy),
                    float(precision),
                    float(recall),
                    float(f1_macro),
                    avg_probability,
                    float(pred_low_risk),
                    float(pred_high_risk),
                    avg_latency,
                    float(current["malaria_incidence"].mean()),
                    float(current["precipitation_mm"].mean()),
                    float(current["temp_annual_mean_c"].mean()),
                ),
            )

            for _, row in current.iterrows():
                cur.execute(
                    """
                    INSERT INTO prediction_logs (
                        batch_id,
                        country,
                        year,
                        temperature_mean,
                        precipitation_mm,
                        gdp_per_capita,
                        population_density,
                        malaria_lag1,
                        malaria_lag2,
                        malaria_lag3,
                        outbreak_probability,
                        outbreak_prediction,
                        model_name,
                        model_version,
                        request_timestamp,
                        inference_latency_ms
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (batch_id, country, year) DO NOTHING
                    """,
                    (
                       batch_id,
                       row["country_name"],
                       int(row["year"]),
                       float(row["temp_annual_mean_c"]),
                       float(row["precipitation_mm"]),
                       float(row["gdp_per_capita"]),
                       float(row["pop_density"]),
                       row.get("malaria_lag1"),
                       row.get("malaria_lag2"),
                       row.get("malaria_lag3"),
                       float(row["outbreak_probability"]),
                       int(row["outbreak_prediction"]),
                       "MalariaOutbreakPredictor",
                       "v1",
                       datetime.now(UTC),
                       float(row["inference_latency_ms"]),
                    ),
                )
        conn.commit()
    finally:
        conn.close()

    print("metrics saved to database")
    print(f"batch_id={batch_id}")
    print(f"drifted_features={drifted_features}/{len(FEATURES)}")
    print(f"accuracy={accuracy:.4f} | f1_macro={f1_macro:.4f}")
    print("drift details:")
    for feature, detail in drift_details.items():
        print(f"  - {feature}: p={detail['p_value']:.6f}, drifted={detail['drifted']}")


if __name__ == "__main__":
    main()
