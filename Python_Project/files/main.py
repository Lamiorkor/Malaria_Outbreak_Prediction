"""
main.py
=======
FastAPI Application — Malaria Outbreak Prediction
Loads the Staging model from MLflow Model Registry.

Usage:
    pip install fastapi uvicorn mlflow xgboost scikit-learn pandas numpy
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /              → health check
    GET  /model/info    → current model stage, version, metrics
    POST /predict       → predict outbreak for one country-year
    POST /predict/batch → predict for multiple records
    POST /model/promote → promote Staging → Production (admin)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional
import pandas as pd
import numpy as np
import json, os, logging
from datetime import datetime

# Import the prediction pipeline
import sys
sys.path.append(r"C:\Users\Likhita Kolli\OneDrive - Hochschule Luzern\SEM_2\AI\Python_Project\files")
from predict_pipeline import PredictionPipeline

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    global pipeline
    import logging as _log
    logger = _log.getLogger(__name__)
    logger.info("Loading pipeline...")
    try:
        pipeline = PredictionPipeline(stage="Staging")
        logger.info("Pipeline loaded ✅")
    except Exception as e:
        logger.error(f"Failed: {e}")
        pipeline = PredictionPipeline.__new__(PredictionPipeline)
        pipeline.stage = "local"
        pipeline._load_local()
    yield

app = FastAPI(lifespan=lifespan,
    title        = "🦟 Malaria Outbreak Predictor API",
    description  = """
Predicts the probability of a malaria outbreak in the following year
based on climate and socioeconomic data.

**Model:** Logistic Regression / XGBoost (best selected automatically)  
**MLflow Stage:** Staging  
**Data:** World Bank + NOAA climate indicators  
    """,
    version      = "1.0.0",
    docs_url     = "/docs",    # Swagger UI  → http://localhost:8000/docs
    redoc_url    = "/redoc",   # ReDoc UI    → http://localhost:8000/redoc
)

# Allow all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Load pipeline once at startup ─────────────────────────────────
# FastAPI loads this when the server starts — not on every request
pipeline: Optional[PredictionPipeline] = None



# ═══════════════════════════════════════════════════════════════════
# Pydantic Schemas
# 📖 https://docs.pydantic.dev/latest/
# ═══════════════════════════════════════════════════════════════════

class CountryRecord(BaseModel):
    """Input schema for a single country-year prediction."""
    country_name             : str   = Field(...)
    country_code             : str   = Field(...)
    year                     : int   = Field(..., ge=2000, le=2030)
    malaria_incidence        : float = Field(..., ge=0,   description="Cases per 1,000 population at risk")
    precipitation_mm         : float = Field(..., ge=0)
    pop_density              : float = Field(..., gt=0,   description="People per sq km")
    gdp_per_capita           : float = Field(..., ge=0,   description="USD")
    temp_annual_mean_c       : float = Field(...)
    temp_growing_season_mean : float = Field(...)

    # Previous years needed for lag features (optional — provide for accuracy)
    history                  : Optional[List[dict]] = Field(
        None,
        description="List of prior year records for the same country "
                    "(used to compute lag features). Include at least 3 years."
    )

    @field_validator("year")
    @classmethod
    def year_must_be_reasonable(cls, v):
        if v < 2000 or v > 2030:
            raise ValueError("Year must be between 2000 and 2030")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "country_name"            : "Ghana",
                "country_code"            : "GHA",
                "year"                    : 2023,
                "malaria_incidence"       : 180.5,
                "precipitation_mm"        : 1200.0,
                "pop_density"             : 130.0,
                "gdp_per_capita"          : 2200.0,
                "temp_annual_mean_c"      : 26.5,
                "temp_growing_season_mean": 27.1,
            }
        }
    )


class PredictionResponse(BaseModel):
    """Output schema returned by /predict."""
    country              : str
    year                 : int
    outbreak_probability : float = Field(..., description="P(outbreak next year) in [0,1]")
    outbreak_alert       : bool  = Field(..., description="True if prob >= threshold")
    risk_level           : str   = Field(..., description="LOW / MEDIUM / HIGH")
    threshold_used       : float
    model_stage          : str
    model_name           : str
    predicted_at         : str


class BatchRequest(BaseModel):
    """Input schema for batch predictions."""
    records: List[CountryRecord]


class BatchResponse(BaseModel):
    """Output schema for batch predictions."""
    total_records    : int
    outbreaks_flagged: int
    predictions      : List[PredictionResponse]


class ModelInfo(BaseModel):
    """Schema for /model/info response."""
    model_name   : str
    model_stage  : str
    best_model   : str
    threshold    : float
    val_aucs     : dict
    test_metrics : dict


# ═══════════════════════════════════════════════════════════════════
# Helper: build record dict from request
# ═══════════════════════════════════════════════════════════════════
def build_record(req: CountryRecord) -> list:
    """
    Convert a CountryRecord (API input) into a list of raw dicts
    that PredictionPipeline.engineer_features() can process.
    Includes history rows if provided.
    """
    current = {
        "Country Name"             : req.country_name,
        "Country Code"             : req.country_code,
        "Year"                     : req.year,
        "Malaria_Incidence"        : req.malaria_incidence,
        "Precipitation_mm"         : req.precipitation_mm,
        "Pop_Density"              : req.pop_density,
        "GDP_per_Capita"           : req.gdp_per_capita,
        "Temp_Annual_Mean_C"       : req.temp_annual_mean_c,
        "Temp_GrowingSeason_Mean_C": req.temp_growing_season_mean,
    }

    if req.history:
        rows = req.history + [current]
    else:
        rows = [current]

    return rows


# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

# ── GET / — Health check ──────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    """Health check — confirms API is running."""
    return {
        "status" : "healthy",
        "app"    : "Malaria Outbreak Predictor",
        "version": "1.0.0",
        "docs"   : "http://localhost:8000/docs",
        "time"   : datetime.utcnow().isoformat(),
    }


# ── GET /model/info — Model metadata ─────────────────────────────
@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def model_info():
    """
    Returns information about the currently loaded model.
    Shows which model (LR or XGBoost) is in Staging,
    its validation and test metrics, and the decision threshold.
    """
    if pipeline is None:
        raise HTTPException(status_code=503,
                            detail="Pipeline not loaded yet.")
    meta = pipeline.metadata
    return ModelInfo(
        model_name   = meta["model_name"],
        model_stage  = pipeline.stage,
        best_model   = meta.get("best_model", "unknown"),
        threshold    = meta["threshold"],
        val_aucs     = meta.get("val_aucs", {}),
        test_metrics = meta.get("test_metrics", {}),
    )


# ── POST /predict — Single prediction ────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: CountryRecord):
    """
    Predict malaria outbreak probability for a single country-year.

    **Input:** country name, year, climate + socioeconomic indicators  
    **Output:** outbreak probability [0,1], alert flag, risk level (LOW/MEDIUM/HIGH)

    Provide at least 3 years of `history` for best accuracy
    (enables lag feature computation).
    """
    if pipeline is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Try again in a moment.")

    try:
        rows   = build_record(request)
        # predict_single expects the most recent row; pass full history
        record = rows[-1]
        record["_history"] = rows[:-1]  # pipeline uses this if available

        result = pipeline.predict_single(record)

        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])

        result["predicted_at"] = datetime.utcnow().isoformat()
        log.info(f"Prediction: {result['country']} {result['year']} "
                 f"→ P={result['outbreak_probability']} "
                 f"ALERT={result['outbreak_alert']}")
        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500,
                            detail=f"Prediction failed: {str(e)}")


# ── POST /predict/batch — Batch predictions ───────────────────────
@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(request: BatchRequest):
    """
    Predict outbreak probability for multiple country-year records.
    Returns all predictions plus a summary count.

    Useful for dashboard views or bulk risk assessment.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    predictions = []
    for rec in request.records:
        try:
            rows   = build_record(rec)
            record = rows[-1]
            result = pipeline.predict_single(record)
            if "error" not in result:
                result["predicted_at"] = datetime.utcnow().isoformat()
                predictions.append(PredictionResponse(**result))
        except Exception as e:
            log.warning(f"Skipping {rec.country_name} {rec.year}: {e}")

    outbreaks = sum(1 for p in predictions if p.outbreak_alert)

    return BatchResponse(
        total_records     = len(predictions),
        outbreaks_flagged = outbreaks,
        predictions       = predictions,
    )


# ── POST /model/promote — Promote Staging → Production ────────────
@app.post("/model/promote", tags=["Model"])
async def promote_model(secret_key: str = "malaria2024"):
    """
    Promote the current Staging model to Production.

    **Admin use only.** Requires the secret_key parameter.
    In production, replace this with proper authentication.

    This transitions the MLflow model version from Staging → Production,
    making it the official serving model.
    """
    if secret_key != "malaria2024":
        raise HTTPException(status_code=403, detail="Invalid secret key.")

    try:
        import mlflow
        from mlflow import MlflowClient

        MLFLOW_URI = r"file:///C:/Users/Likhita Kolli/OneDrive - Hochschule Luzern/SEM_2/AI/mlruns"
        MODEL_NAME = "MalariaOutbreakPredictor"

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient(tracking_uri=MLFLOW_URI)

        staging_versions = client.get_latest_versions(
            MODEL_NAME, stages=["Staging"])

        if not staging_versions:
            raise HTTPException(status_code=404,
                                detail="No model version found in Staging.")

        version = staging_versions[0].version
        client.transition_model_version_stage(
            name    = MODEL_NAME,
            version = version,
            stage   = "Production",
            archive_existing_versions=True,
        )

        log.info(f"Model v{version} promoted to Production ✅")
        return {
            "status" : "promoted",
            "model"  : MODEL_NAME,
            "version": version,
            "stage"  : "Production",
            "time"   : datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
