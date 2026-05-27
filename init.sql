-- ──────────────────────────────────────────────────────────────────────────
-- init.sql — runs automatically the first time the Postgres container starts.
--
-- Creates the prediction_logs table used by main.py's log_prediction_to_db().
-- Columns match the INSERT statement in
--   dockerisation_and_deployment/webservices/main.py
-- ──────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS prediction_logs (
    id                   SERIAL PRIMARY KEY,
    country              TEXT,
    year                 INTEGER,
    precipitation_mm     DOUBLE PRECISION,
    population_density   DOUBLE PRECISION,
    gdp_per_capita       DOUBLE PRECISION,
    temperature_mean     DOUBLE PRECISION,
    malaria_lag1         DOUBLE PRECISION,
    malaria_lag2         DOUBLE PRECISION,
    malaria_lag3         DOUBLE PRECISION,
    outbreak_probability DOUBLE PRECISION,
    outbreak_prediction  INTEGER,
    model_name           TEXT,
    model_version        TEXT,
    inference_latency_ms DOUBLE PRECISION,
    request_timestamp    TIMESTAMPTZ
);
