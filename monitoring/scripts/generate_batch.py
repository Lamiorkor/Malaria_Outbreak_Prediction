"""
generate_batch.py
-----------------
Generate one new production-like batch for malaria monitoring.

What the script does:
1. Loads historical malaria data
2. Samples rows from years after the reference window
3. Generates model predictions
4. Optionally injects synthetic drift for the live presentation
5. Saves one batch CSV in data/current_batches/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_DIR = PROJECT_ROOT / "dockerisation_and_deployment" / "webservices"
if str(WEBAPP_DIR) not in sys.path:
    sys.path.append(str(WEBAPP_DIR))

from predict import PredictionPipeline  # noqa: E402

DEFAULT_RAW_PATHS = [
    PROJECT_ROOT / "data" / "malaria_final_dataset.csv",
]

CONFIG_PATH = PROJECT_ROOT  / "monitoring" / "data" / "monitoring_config.json"
BATCH_DIR = PROJECT_ROOT / "monitoring" / "data" / "current_batches"


def load_raw_dataset(raw_path: str | None) -> pd.DataFrame:
    if raw_path:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Raw dataset not found: {path}")
        return pd.read_csv(path)

    for path in DEFAULT_RAW_PATHS:
        if path.exists():
            return pd.read_csv(path)

    raise FileNotFoundError(
        "Could not find malaria_final_dataset.csv. "
        "Pass --raw-path explicitly or place the file in monitoring/data/raw or project root."
    )


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("monitoring_config.json not found. Run prepare_reference.py first.")
    return json.loads(CONFIG_PATH.read_text())


def add_future_label(df: pd.DataFrame, outbreak_threshold: float) -> pd.DataFrame:
    df = df.sort_values(["Country Name", "Year"]).reset_index(drop=True)
    df["Next_Year_Incidence"] = df.groupby("Country Name")["Malaria_Incidence"].shift(-1)
    df["true_label"] = (df["Next_Year_Incidence"] >= outbreak_threshold).astype("Int64")
    return df


def has_three_year_history(country_df: pd.DataFrame, current_year: int) -> bool:
    years = set(country_df["Year"].tolist())
    return all((current_year - i) in years for i in [1, 2, 3])


def build_history(country_df: pd.DataFrame, current_year: int) -> List[Dict]:
    hist = (
        country_df[country_df["Year"].isin([current_year - 3, current_year - 2, current_year - 1])]
        .sort_values("Year")
        .copy()
    )
    return [
        {
            "country_name": row["Country Name"],
            "country_code": row["Country Code"],
            "year": int(row["Year"]),
            "malaria_incidence": float(row["Malaria_Incidence"]),
            "precipitation_mm": float(row["Precipitation_mm"]),
            "pop_density": float(row["Pop_Density"]),
            "gdp_per_capita": float(row["GDP_per_Capita"]),
            "temp_annual_mean_c": float(row["Temp_Annual_Mean_C"]),
            "temp_growing_season_mean_c": float(row["Temp_GrowingSeason_Mean_C"]),
        }
        for _, row in hist.iterrows()
    ]


def build_current_record(row: pd.Series) -> Dict:
    return {
        "country_name": row["Country Name"],
        "country_code": row["Country Code"],
        "year": int(row["Year"]),
        "malaria_incidence": float(row["Malaria_Incidence"]),
        "precipitation_mm": float(row["Precipitation_mm"]),
        "pop_density": float(row["Pop_Density"]),
        "gdp_per_capita": float(row["GDP_per_Capita"]),
        "temp_annual_mean_c": float(row["Temp_Annual_Mean_C"]),
        "temp_growing_season_mean_c": float(row["Temp_GrowingSeason_Mean_C"]),
    }


def inject_synthetic_drift(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "none":
        return df

    shifted = df.copy()

    if mode == "climate":
        shifted["Temp_Annual_Mean_C"] += 1.2
        shifted["Temp_GrowingSeason_Mean_C"] += 1.0
        shifted["Precipitation_mm"] *= 1.15
    elif mode == "economic":
        shifted["GDP_per_Capita"] *= 0.88
        shifted["Pop_Density"] *= 1.03
    elif mode == "mixed":
        shifted["Temp_Annual_Mean_C"] += 0.8
        shifted["Temp_GrowingSeason_Mean_C"] += 0.7
        shifted["Precipitation_mm"] *= 1.10
        shifted["GDP_per_Capita"] *= 0.92
        shifted["Malaria_Incidence"] *= 1.08
    elif mode == "extreme":
        shifted["Temp_Annual_Mean_C"] += 2.5
        shifted["Precipitation_mm"] *= 1.3
        shifted["GDP_per_Capita"] *= 0.7
        shifted["Malaria_Incidence"] *= 1.2
    else:
        raise ValueError(f"Unsupported drift mode: {mode}")

    return shifted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=str, default=None, help="Path to malaria_final_dataset.csv")
    parser.add_argument("--batch-size", type=int, default=30, help="Number of rows to sample")
    parser.add_argument(
        "--drift-mode",
        type=str,
        choices=["none", "climate", "economic", "mixed", "extreme"],
        default="mixed",
        help="Synthetic drift scenario for the demo",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    config = load_config()
    raw = load_raw_dataset(args.raw_path)
    raw = add_future_label(raw, outbreak_threshold=config["outbreak_incidence_threshold"])

    production_pool = raw[
        raw["Year"].gt(config["reference_end_year"])
        & raw["Next_Year_Incidence"].notna()
    ].copy()

    production_pool = inject_synthetic_drift(production_pool, args.drift_mode)

    eligible_rows = []

    raw_sorted = raw.sort_values(["Country Name", "Year"]).reset_index(drop=True)

    for _, row in production_pool.iterrows():
        country_name = row["Country Name"]
        current_year = int(row["Year"])

        full_country_df = raw_sorted[
            raw_sorted["Country Name"] == country_name
        ].copy()

        if has_three_year_history(full_country_df, current_year):
            eligible_rows.append(row)

    eligible = pd.DataFrame(eligible_rows)
    if eligible.empty:
        raise RuntimeError("No eligible rows found for batch generation.")

    sampled = eligible.sample(
        n=min(args.batch_size, len(eligible)),
        replace=False,
        random_state=args.random_state,
    ).sort_values(["Country Name", "Year"])

    pipeline = PredictionPipeline()
    batch_results = []

    # Use raw data sorted by country/year for proper history lookup
    raw_sorted = raw.sort_values(["Country Name", "Year"]).reset_index(drop=True)

    for _, row in sampled.iterrows():
        country_name = row["Country Name"]
        current_year = int(row["Year"])
        country_hist_df = raw_sorted[raw_sorted["Country Name"] == country_name].copy()

        history = build_history(country_hist_df, current_year)
        record = build_current_record(row)
        record["history"] = history

        start = time.perf_counter()
        prediction = pipeline.predict_single(record.copy())
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        batch_results.append(
            {
                "country_name": record["country_name"],
                "country_code": record["country_code"],
                "year": record["year"],
                "malaria_incidence": record["malaria_incidence"],
                "precipitation_mm": record["precipitation_mm"],
                "pop_density": record["pop_density"],
                "gdp_per_capita": record["gdp_per_capita"],
                "malaria_lag1": history[-1]["malaria_incidence"] if len(history) >= 1 else None,
                "malaria_lag2": history[-2]["malaria_incidence"] if len(history) >= 2 else None,
                "malaria_lag3": history[-3]["malaria_incidence"] if len(history) >= 3 else None,
                "temp_annual_mean_c": record["temp_annual_mean_c"],
                "temp_growing_season_mean_c": record["temp_growing_season_mean_c"],
                "true_label": int(row["true_label"]),
                "outbreak_probability": float(prediction["outbreak_probability"]),
                "outbreak_prediction": int(prediction["outbreak_alert"]),
                "risk_level": prediction["risk_level"],
                "inference_latency_ms": latency_ms,
                "drift_mode": args.drift_mode,
            }
        )

    batch_df = pd.DataFrame(batch_results)
    batch_id = str(uuid.uuid4())[:8]
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BATCH_DIR / f"{batch_id}.csv"
    batch_df.to_csv(output_path, index=False)

    print(f"batch saved: {output_path}")
    print(f"rows: {len(batch_df)} | drift_mode: {args.drift_mode}")


if __name__ == "__main__":
    main()
