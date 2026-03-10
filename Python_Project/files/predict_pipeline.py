"""
predict_pipeline.py
===================
MLOps Prediction Pipeline — Malaria Outbreak Prediction

Loads the model currently in MLflow Staging and runs
batch predictions on new country-year data.

Usage:
    python predict_pipeline.py --input new_data.csv --output predictions.csv

    # Or import as a module in FastAPI:
    from predict_pipeline import PredictionPipeline
    pipeline = PredictionPipeline()
    result   = pipeline.predict_single({...})
"""

import os, json, pickle, argparse, logging
import pandas as pd
import numpy as np
import mlflow
from mlflow import MlflowClient

# ── Paths ─────────────────────────────────────────────────────────
MODEL_DIR  = r"C:\Users\Likhita Kolli\OneDrive - Hochschule Luzern\SEM_2\AI\models"
MLFLOW_URI = r"file:///C:/Users/Likhita Kolli/OneDrive - Hochschule Luzern/SEM_2/AI/mlruns"
MODEL_NAME = "MalariaOutbreakPredictor"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
class PredictionPipeline:
    """
    Encapsulates the full prediction logic:
      1. Load Staging model from MLflow Model Registry
      2. Load scaler + metadata
      3. Engineer features for new input
      4. Scale input
      5. Return probability + risk label

    This class is imported directly by FastAPI.
    """

    def __init__(self, stage: str = "Staging"):
        self.stage    = stage
        self.model    = None
        self.scaler   = None
        self.metadata = None
        self._load()

    # ── Load model from MLflow Staging ───────────────────────────
    def _load(self):
        import mlflow as _mlflow
        import mlflow.sklearn
        log.info(f"Loading model from MLflow ({self.stage})...")
        _mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient(tracking_uri=MLFLOW_URI)

        # Get the version currently in Staging
        versions = client.get_latest_versions(
            MODEL_NAME, stages=[self.stage] if self.stage != "None" else [])

        if not versions:
            log.warning(f"No model in {self.stage}. "
                        "Falling back to local .pkl file.")
            self._load_local()
            return

        version    = versions[0]
        model_uri  = f"models:/{MODEL_NAME}/{self.stage}"

        log.info(f"  Model URI  : {model_uri}")
        log.info(f"  Version    : {version.version}")
        log.info(f"  Run ID     : {version.run_id}")

        # Load model — works for both sklearn and xgboost
        try:
            import mlflow.sklearn as _mlflow_sklearn
            self.model = _mlflow_sklearn.load_model(model_uri)
            log.info("  Loaded as sklearn model")
        except Exception:
            import mlflow.xgboost as _mlflow_xgb
            self.model = _mlflow_xgb.load_model(model_uri)
            log.info("  Loaded as XGBoost model")

        self._load_scaler_and_meta()
        log.info("Pipeline ready ✅")

    def _load_local(self):
        """Fallback: load from local .pkl files."""
        with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
            meta = json.load(f)
        best = meta["best_model"]
        pkl  = "logistic_regression.pkl" if "Logistic" in best \
               else "xgboost_model.pkl"
        with open(os.path.join(MODEL_DIR, pkl), "rb") as f:
            self.model = pickle.load(f)
        self._load_scaler_and_meta()

    def _load_scaler_and_meta(self):
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
            self.metadata = json.load(f)
        self.threshold    = self.metadata["threshold"]
        self.feature_cols = self.metadata["feature_cols"]

    # ── Feature engineering (mirrors feature_engineering notebook) ─
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same transformations as the Feature Engineering notebook.
        Input df must have columns:
            Country Name, Year, Malaria_Incidence, Precipitation_mm,
            Pop_Density, GDP_per_Capita, Temp_Annual_Mean_C,
            Temp_GrowingSeason_Mean_C
        """
        df = df.sort_values(["Country Name", "Year"]).reset_index(drop=True)
        grp = df.groupby("Country Name")

        # Lag features
        df["Malaria_Lag1"] = grp["Malaria_Incidence"].shift(1)
        df["Malaria_Lag2"] = grp["Malaria_Incidence"].shift(2)
        df["Malaria_Lag3"] = grp["Malaria_Incidence"].shift(3)
        df["Temp_Lag1"]    = grp["Temp_Annual_Mean_C"].shift(1)
        df["Precip_Lag1"]  = grp["Precipitation_mm"].shift(1)
        df["GDP_Lag1"]     = grp["GDP_per_Capita"].shift(1)

        # Rolling averages
        def roll3(x):
            return x.rolling(window=3, min_periods=2).mean()

        df["Malaria_Roll3"] = grp["Malaria_Incidence"].transform(roll3)
        df["Temp_Roll3"]    = grp["Temp_Annual_Mean_C"].transform(roll3)
        df["Precip_Roll3"]  = grp["Precipitation_mm"].transform(roll3)

        # YoY change
        df["Malaria_YoY_Change"] = (
            grp["Malaria_Incidence"]
            .transform(lambda x: x.pct_change())
            .replace([np.inf, -np.inf], np.nan)
            .clip(-5, 5)
        )

        # Interaction features
        df["Temp_Precip_Interaction"] = (
            df["Temp_Annual_Mean_C"] * df["Precipitation_mm"])
        df["Temp_Squared"]            = df["Temp_Annual_Mean_C"] ** 2
        df["Climate_Risk_Index"]      = (
            df["Temp_Annual_Mean_C"] * df["Precipitation_mm"]
        ) / (df["GDP_per_Capita"] + 1)
        df["Vulnerability_Index"]     = (
            df["Pop_Density"] / (df["GDP_per_Capita"] + 1))
        df["Log_GDP"]                 = np.log1p(df["GDP_per_Capita"])
        df["Log_Malaria_Incidence"]   = np.log1p(df["Malaria_Incidence"])

        return df

    # ── Scale input ───────────────────────────────────────────────
    def _scale(self, X: pd.DataFrame) -> pd.DataFrame:
        scaled = self.scaler.transform(X)
        return pd.DataFrame(scaled, columns=X.columns, index=X.index)

    # ── Predict: single record (for FastAPI) ──────────────────────
    def predict_single(self, record: dict) -> dict:
        """
        Predict outbreak probability for one country-year record.
        Accepts optional _history key with list of prior year dicts.
        If no history provided, uses median values from training data
        to fill lag features so prediction still works.
        """
        # Build full history DataFrame
        history = record.pop("_history", [])
        all_rows = history + [record]

        # Normalise key names (API uses snake_case, pipeline uses Title Case)
        def normalise(r):
            mapping = {
                "country_name"            : "Country Name",
                "country_code"            : "Country Code",
                "malaria_incidence"       : "Malaria_Incidence",
                "precipitation_mm"        : "Precipitation_mm",
                "pop_density"             : "Pop_Density",
                "gdp_per_capita"          : "GDP_per_Capita",
                "temp_annual_mean_c"      : "Temp_Annual_Mean_C",
                "temp_growing_season_mean": "Temp_GrowingSeason_Mean_C",
                "year"                    : "Year",
            }
            return {mapping.get(k, k): v for k, v in r.items()}

        all_rows = [normalise(r) for r in all_rows]
        df_in    = pd.DataFrame(all_rows)

        # Ensure required columns exist
        required = ["Country Name", "Year", "Malaria_Incidence",
                    "Precipitation_mm", "Pop_Density", "GDP_per_Capita",
                    "Temp_Annual_Mean_C", "Temp_GrowingSeason_Mean_C"]
        for col in required:
            if col not in df_in.columns:
                df_in[col] = 0.0

        df_eng = self.engineer_features(df_in)

        # If lag features are NaN (no history), fill with column median
        for col in self.feature_cols:
            if col in df_eng.columns:
                df_eng[col] = df_eng[col].fillna(df_eng[col].median())

        # Add any missing feature columns with 0
        for col in self.feature_cols:
            if col not in df_eng.columns:
                df_eng[col] = 0.0

        # Fill all remaining NaN with 0 (robust fallback)
        df_eng[self.feature_cols] = df_eng[self.feature_cols].fillna(0.0)

        X        = df_eng[self.feature_cols].tail(1)
        X_scaled = self._scale(X)

        prob  = float(self.model.predict_proba(X_scaled)[:, 1][0])
        alert = prob >= self.threshold

        risk_level = (
            "HIGH"   if prob >= 0.70 else
            "MEDIUM" if prob >= 0.40 else
            "LOW"
        )

        country = (df_in.iloc[-1].get("Country Name")
                   or record.get("country_name", "Unknown"))
        year    = int(df_in.iloc[-1].get("Year", 0))

        return {
            "country"             : country,
            "year"                : year,
            "outbreak_probability": round(prob, 4),
            "outbreak_alert"      : alert,
            "risk_level"          : risk_level,
            "threshold_used"      : self.threshold,
            "model_stage"         : self.stage,
            "model_name"          : MODEL_NAME,
        }

    # ── Predict: batch CSV (CLI use) ──────────────────────────────
    def predict_batch(self, input_path: str, output_path: str):
        """
        Run predictions on a CSV file of new country-year records.
        Saves results to output_path.
        """
        log.info(f"Loading input: {input_path}")
        df = pd.read_csv(input_path)

        log.info("Engineering features...")
        df_eng = self.engineer_features(df)
        df_eng = df_eng.dropna(subset=self.feature_cols)

        X = df_eng[self.feature_cols]

        # Use scaler only if model is LR (scale-sensitive)
        best = self.metadata.get("best_model", "")
        if "Logistic" in best or "LR" in best:
            X = self._scale(X)

        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        df_eng["outbreak_probability"] = probs.round(4)
        df_eng["outbreak_alert"]       = preds
        df_eng["risk_level"]           = np.where(
            probs >= 0.70, "HIGH",
            np.where(probs >= 0.40, "MEDIUM", "LOW")
        )

        out_cols = ["Country Name", "Year", "Malaria_Incidence",
                    "outbreak_probability", "outbreak_alert", "risk_level"]
        df_out = df_eng[out_cols]
        df_out.to_csv(output_path, index=False)

        log.info(f"Predictions saved to: {output_path}")
        log.info(f"Records processed   : {len(df_out)}")
        log.info(f"Outbreaks predicted : {preds.sum()}")
        return df_out


# ── CLI entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Malaria Outbreak Predictor")
    parser.add_argument("--input",  required=True,
                        help="Path to input CSV with raw features")
    parser.add_argument("--output", default="predictions.csv",
                        help="Path for output predictions CSV")
    parser.add_argument("--stage",  default="Staging",
                        choices=["Staging", "Production", "None"],
                        help="MLflow model stage to load")
    args = parser.parse_args()

    pipeline = PredictionPipeline(stage=args.stage)
    results  = pipeline.predict_batch(args.input, args.output)
    print(results.head(10).to_string(index=False))
