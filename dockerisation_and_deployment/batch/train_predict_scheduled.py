"""
train_predict_scheduled.py

Simulates a scheduled batch job:
1. Retrains the model
2. Runs a sample prediction
3. Saves the result
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path

from prefect import flow, task, get_run_logger

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR / "dockerisation_and_deployment" / "webservices"))

from predict import PredictionPipeline 

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@task(name="run_training")
def run_training():
    logger = get_run_logger()
    logger.info("Starting yearly malaria model retraining.")

    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "training_pipeline" / "train_pipeline.py")],
        capture_output=True,
        text=True
    )

    if result.stdout:
        logger.info(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            logger.error(result.stderr)
        raise RuntimeError("Training failed.")

    logger.info("Training completed successfully.")


@task(name="run_prediction")
def run_prediction():
    logger = get_run_logger()
    logger.info("Starting yearly batch prediction.")

    pipeline = PredictionPipeline()

    sample = {
        "country_name": "Ghana",
        "country_code": "GHA",
        "year": 2023,
        "malaria_incidence": 180.5,
        "precipitation_mm": 1200.0,
        "pop_density": 130.0,
        "gdp_per_capita": 2200.0,
        "temp_annual_mean_c": 26.5,
        "temp_growing_season_mean_c": 27.1,
        "history": [
            {
                "country_name": "Ghana",
                "country_code": "GHA",
                "year": 2020,
                "malaria_incidence": 170.0,
                "precipitation_mm": 1100.0,
                "pop_density": 125.0,
                "gdp_per_capita": 2100.0,
                "temp_annual_mean_c": 26.1,
                "temp_growing_season_mean_c": 26.8
            },
            {
                "country_name": "Ghana",
                "country_code": "GHA",
                "year": 2021,
                "malaria_incidence": 175.0,
                "precipitation_mm": 1150.0,
                "pop_density": 127.0,
                "gdp_per_capita": 2150.0,
                "temp_annual_mean_c": 26.3,
                "temp_growing_season_mean_c": 26.9
            },
            {
                "country_name": "Ghana",
                "country_code": "GHA",
                "year": 2022,
                "malaria_incidence": 178.0,
                "precipitation_mm": 1180.0,
                "pop_density": 129.0,
                "gdp_per_capita": 2180.0,
                "temp_annual_mean_c": 26.4,
                "temp_growing_season_mean_c": 27.0
            }
        ]
    }

    prediction_result = pipeline.predict_single(sample)

    run_time = datetime.now(UTC)
    timestamp = run_time.strftime("%Y%m%d_%H%M%S")

    output_payload = {
        "generated_at": run_time.isoformat(),
        "input": sample,
        "prediction": prediction_result
    }

    output_file = OUTPUT_DIR / f"scheduled_prediction_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(output_payload, f, indent=2)

    logger.info(f"Prediction completed successfully. Output saved to {output_file}")
    logger.info(json.dumps(output_payload, indent=2))

    return output_payload


@flow(name="malaria_yearly_train_and_batch_predict")
def malaria_yearly_train_and_batch_predict():
    logger = get_run_logger()
    logger.info("Yearly malaria pipeline started.")
    run_training()
    prediction = run_prediction()
    logger.info("Yearly malaria pipeline finished successfully.")
    return prediction


if __name__ == "__main__":
    malaria_yearly_train_and_batch_predict()