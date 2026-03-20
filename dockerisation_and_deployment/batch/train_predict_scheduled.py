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

# ── Setup paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]

# Allow import from webservices
sys.path.append(str(BASE_DIR / "dockerisation_and_deployment" / "webservices"))

from predict import PredictionPipeline  # noqa: E402

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 1: Run training ─────────────────────────────────────────────────────
def run_training():
    print("\n🚀 Running training pipeline...\n")

    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "training_pipeline" / "train_pipeline.py")],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Training failed")

    print("\n✅ Training complete.\n")


# ── Step 2: Run prediction ───────────────────────────────────────────────────
def run_prediction():
    print("🔮 Running prediction...\n")

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

    result = pipeline.predict_single(sample)
    result["generated_at"] = datetime.now(UTC).isoformat()

    output_file = OUTPUT_DIR / "scheduled_prediction.json"

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print("✅ Prediction complete.\n")
    print("📁 Saved to:", output_file)
    print("\nResult:")
    print(json.dumps(result, indent=2))


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_training()
    run_prediction()