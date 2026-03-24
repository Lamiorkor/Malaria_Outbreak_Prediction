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

from contextlib import asynccontextmanager
from datetime import datetime, UTC
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .predict import PredictionPipeline
except ImportError:
    from predict import PredictionPipeline

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = PredictionPipeline()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Malaria Outbreak Predictor API",
    version="1.0.0",
)


class CountryRecord(BaseModel):
    country_name: str
    country_code: str
    year: int = Field(..., ge=2000, le=2035)
    malaria_incidence: float = Field(..., ge=0)
    precipitation_mm: float = Field(..., ge=0)
    pop_density: float = Field(..., gt=0)
    gdp_per_capita: float = Field(..., ge=0)
    temp_annual_mean_c: float
    temp_growing_season_mean_c: float
    history: Optional[List[dict]] = None


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "app": "Malaria Outbreak Predictor API",
        "time": datetime.now(UTC).isoformat()
    }


@app.get("/model/info")
async def model_info():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    return pipeline.metadata


@app.post("/predict")
async def predict(record: CountryRecord):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        result = pipeline.predict_single(record.model_dump())
        result["predicted_at"] = datetime.now(UTC).isoformat()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))