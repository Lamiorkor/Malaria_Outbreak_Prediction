"""
prepare_reference.py
--------------------
Create a reference dataset for malaria monitoring using the validation period.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEBAPP_DIR = PROJECT_ROOT / "dockerisation_and_deployment" / "webservices"

if str(WEBAPP_DIR) not in sys.path:
    sys.path.append(str(WEBAPP_DIR))

from predict import PredictionPipeline  # noqa: E402

DEFAULT_RAW_PATHS = [
    PROJECT_ROOT / "data" / "malaria_final_dataset.csv",
]

REFERENCE_OUTPUT = PROJECT_ROOT / "monitoring" / "data" / "reference.csv"
CONFIG_OUTPUT = PROJECT_ROOT / "monitoring" / "data" / "monitoring_config.json"


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
        "Pass --raw-path explicitly or place it in data/malaria_final_dataset.csv."
    )


def add_future_label(
    df: pd.DataFrame,
    label_quantile: float,
    train_end_year: int,
) -> tuple[pd.DataFrame, float]:
    df = df.sort_values(["Country Name", "Year"]).reset_index(drop=True)

    df["Next_Year_Incidence"] = (
        df.groupby("Country Name")["Malaria_Incidence"].shift(-1)
    )

    threshold = float(
        df.loc[df["Year"] <= train_end_year, "Malaria_Incidence"]
        .dropna()
        .quantile(label_quantile)
    )

    df["true_label"] = (
        df["Next_Year_Incidence"] >= threshold
    ).astype("Int64")

    return df, threshold


def has_three_year_history(full_country_df: pd.DataFrame, current_year: int) -> bool:
    years = set(full_country_df["Year"].tolist())
    return all((current_year - i) in years for i in [1, 2, 3])


def build_history(full_country_df: pd.DataFrame, current_year: int) -> List[Dict]:
    hist = (
        full_country_df[
            full_country_df["Year"].isin(
                [current_year - 3, current_year - 2, current_year - 1]
            )
        ]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=str, default=None)
    parser.add_argument("--train-end-year", type=int, default=2016)
    parser.add_argument("--reference-start-year", type=int, default=2017)
    parser.add_argument("--reference-end-year", type=int, default=2019)
    parser.add_argument("--label-quantile", type=float, default=0.75)
    args = parser.parse_args()

    raw = load_raw_dataset(args.raw_path)

    raw, outbreak_threshold = add_future_label(
        raw,
        label_quantile=args.label_quantile,
        train_end_year=args.train_end_year,
    )

    # Use validation years as monitoring reference, but build history from full raw data.
    candidates = raw[
        raw["Year"].between(args.reference_start_year, args.reference_end_year)
        & raw["Next_Year_Incidence"].notna()
        & raw["true_label"].notna()
    ].copy()

    pipeline = PredictionPipeline()
    results = []

    raw_sorted = raw.sort_values(["Country Name", "Year"]).reset_index(drop=True)

    for _, row in candidates.iterrows():
        country_name = row["Country Name"]
        current_year = int(row["Year"])

        full_country_df = raw_sorted[
            raw_sorted["Country Name"] == country_name
        ].copy()

        if not has_three_year_history(full_country_df, current_year):
            continue

        history = build_history(full_country_df, current_year)
        record = build_current_record(row)
        record["history"] = history

        start = time.perf_counter()
        prediction = pipeline.predict_single(record.copy())
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        results.append(
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
            }
        )

    reference = (
        pd.DataFrame(results)
        .sort_values(["country_name", "year"])
        .reset_index(drop=True)
    )

    REFERENCE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(REFERENCE_OUTPUT, index=False)

    CONFIG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_OUTPUT.write_text(
        json.dumps(
            {
                "train_end_year": args.train_end_year,
                "reference_start_year": args.reference_start_year,
                "reference_end_year": args.reference_end_year,
                "label_quantile": args.label_quantile,
                "outbreak_incidence_threshold": outbreak_threshold,
                "reference_rows": len(reference),
            },
            indent=2,
        )
    )

    print(f"reference.csv created with {len(reference)} rows")
    print(f"monitoring config saved to {CONFIG_OUTPUT}")
    print(f"Outbreak threshold: {outbreak_threshold:.4f}")


if __name__ == "__main__":
    main()