"""
predict.py
===================
MLOps Prediction Pipeline — Malaria Outbreak Prediction

Loads the model currently in MLflow Staging and runs
batch predictions on new country-year data.

Usage:
    python predict.py --input new_data.csv --output predictions.csv

    # Or import as a module in FastAPI:
    from predict import PredictionPipeline
    pipeline = PredictionPipeline()
    result   = pipeline.predict_single({...})
"""

import json
import logging
import pickle
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "models")))

class PredictionPipeline:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.threshold = None
        self.feature_cols = None
        self._load_local()

    def _load_local(self):
        with open(MODEL_DIR / "logistic_regression.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(MODEL_DIR / "model_metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.threshold = self.metadata["threshold"]
        self.feature_cols = self.metadata["feature_cols"]
        log.info("Prediction pipeline loaded successfully.")

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["Country Name", "Year"]).reset_index(drop=True)
        grp = df.groupby("Country Name")

        df["Malaria_Lag1"] = grp["Malaria_Incidence"].shift(1)
        df["Malaria_Lag2"] = grp["Malaria_Incidence"].shift(2)
        df["Malaria_Lag3"] = grp["Malaria_Incidence"].shift(3)
        df["Temp_Lag1"] = grp["Temp_Annual_Mean_C"].shift(1)
        df["Precip_Lag1"] = grp["Precipitation_mm"].shift(1)
        df["GDP_Lag1"] = grp["GDP_per_Capita"].shift(1)

        def roll3(x):
            return x.rolling(window=3, min_periods=2).mean()

        df["Malaria_Roll3"] = grp["Malaria_Incidence"].transform(roll3)
        df["Temp_Roll3"] = grp["Temp_Annual_Mean_C"].transform(roll3)
        df["Precip_Roll3"] = grp["Precipitation_mm"].transform(roll3)

        df["Malaria_YoY_Change"] = (
            grp["Malaria_Incidence"]
            .transform(lambda x: x.pct_change())
            .replace([np.inf, -np.inf], np.nan)
            .clip(-5, 5)
        )

        df["Temp_Precip_Interaction"] = df["Temp_Annual_Mean_C"] * df["Precipitation_mm"]
        df["Temp_Squared"] = df["Temp_Annual_Mean_C"] ** 2
        df["Climate_Risk_Index"] = (
            df["Temp_Annual_Mean_C"] * df["Precipitation_mm"]
        ) / (df["GDP_per_Capita"] + 1)
        df["Vulnerability_Index"] = df["Pop_Density"] / (df["GDP_per_Capita"] + 1)
        df["Log_GDP"] = np.log1p(df["GDP_per_Capita"])
        df["Log_Malaria_Incidence"] = np.log1p(df["Malaria_Incidence"])

        return df

    def _scale(self, X: pd.DataFrame) -> pd.DataFrame:
        scaled = self.scaler.transform(X)
        return pd.DataFrame(scaled, columns=X.columns, index=X.index)

    def predict_single(self, record: dict) -> dict:
        history = record.pop("history", [])
        all_rows = history + [record]

        def normalise(r):
            mapping = {
                "country_name": "Country Name",
                "country_code": "Country Code",
                "year": "Year",
                "malaria_incidence": "Malaria_Incidence",
                "precipitation_mm": "Precipitation_mm",
                "pop_density": "Pop_Density",
                "gdp_per_capita": "GDP_per_Capita",
                "temp_annual_mean_c": "Temp_Annual_Mean_C",
                "temp_growing_season_mean_c": "Temp_GrowingSeason_Mean_C",
            }
            return {mapping.get(k, k): v for k, v in r.items()}

        df_in = pd.DataFrame([normalise(r) for r in all_rows])

        defaults = {
            "Country Name": "Unknown",
            "Country Code": "UNK",
            "Year": 0,
            "Malaria_Incidence": 0.0,
            "Precipitation_mm": 0.0,
            "Pop_Density": 0.0,
            "GDP_per_Capita": 0.0,
            "Temp_Annual_Mean_C": 0.0,
            "Temp_GrowingSeason_Mean_C": 0.0,
        }

        for col, default in defaults.items():
            if col not in df_in.columns:
                df_in[col] = default
            else:
                df_in[col] = df_in[col].fillna(default)

        df_eng = self.engineer_features(df_in)

        for col in self.feature_cols:
            if col not in df_eng.columns:
                df_eng[col] = 0.0

        df_eng[self.feature_cols] = df_eng[self.feature_cols].fillna(0.0)

        X = df_eng[self.feature_cols].tail(1)
        X_scaled = self._scale(X)

        prob = float(self.model.predict_proba(X_scaled)[:, 1][0])
        alert = prob >= self.threshold

        risk_level = "HIGH" if prob >= 0.70 else "MEDIUM" if prob >= 0.50 else "LOW"

        return {
            "country": str(df_in.iloc[-1]["Country Name"]),
            "year": int(df_in.iloc[-1]["Year"]),
            "outbreak_probability": round(prob, 4),
            "outbreak_alert": bool(alert),
            "risk_level": risk_level,
            "threshold_used": self.threshold,
            "model_name": self.metadata["model_name"],
        }