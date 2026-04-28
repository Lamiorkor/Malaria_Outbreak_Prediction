from contextlib import asynccontextmanager
from datetime import datetime, UTC
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import os
import time
import psycopg2

try:
    from .predict import PredictionPipeline
except ImportError:
    from predict import PredictionPipeline

pipeline = None


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5433"),
        dbname=os.getenv("DB_NAME", "test"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "example"),
    )


def log_prediction_to_db(record: dict, result: dict, latency_ms: float, predicted_at: datetime):
    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        history = record.get("history") or []

        malaria_lag1 = history[-1]["malaria_incidence"] if len(history) >= 1 else None
        malaria_lag2 = history[-2]["malaria_incidence"] if len(history) >= 2 else None
        malaria_lag3 = history[-3]["malaria_incidence"] if len(history) >= 3 else None

        cur.execute(
            """
            INSERT INTO prediction_logs (
                country,
                year,
                precipitation_mm,
                population_density,
                gdp_per_capita,
                temperature_mean,
                malaria_lag1,
                malaria_lag2,
                malaria_lag3,
                outbreak_probability,
                outbreak_prediction,
                model_name,
                model_version,
                inference_latency_ms,
                request_timestamp
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record.get("country_name"),
                record.get("year"),
                record.get("precipitation_mm"),
                record.get("pop_density"),
                record.get("gdp_per_capita"),
                record.get("temp_annual_mean_c"),

                malaria_lag1,
                malaria_lag2,
                malaria_lag3,

                result.get("outbreak_probability"),
                int(result.get("outbreak_alert")),
                result.get("model_name"),
                "v1",
                latency_ms,
                predicted_at,  # use passed timestamp instead of new one
            ),
        )

        conn.commit()

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


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
        record_dict = record.model_dump()

        start_time = time.time()
        result = pipeline.predict_single(record_dict.copy())
        latency_ms = round((time.time() - start_time) * 1000, 2)
        predicted_at = datetime.now(UTC)

        log_prediction_to_db(record_dict, result, latency_ms, predicted_at)

        result["predicted_at"] = predicted_at.isoformat()
        result["inference_latency_ms"] = latency_ms
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
