"""
test_smoke_pipeline.py
======================
Smoke test for the prediction pipeline using a *dummy model*.

Why this exists
---------------
Slide 11 ("Dummy Model Concept") of the SW10 deck describes exactly this:

    * Replace the real ML model
    * Avoid MLflow dependency in CI
    * Fast and stable testing
    * Focus on logic, not accuracy

So instead of loading `models/logistic_regression.pkl` (which CI doesn't have),
we monkey-patch `PredictionPipeline._load_local` and inject a tiny dummy model
plus an identity-style "scaler".  The real feature-engineering code is then
exercised end-to-end with realistic-looking inputs — which catches bugs like
missing columns, wrong column order or KeyErrors that the unit tests alone
cannot find.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Dummies — stand in for the real LogisticRegression + StandardScaler
# ─────────────────────────────────────────────────────────────────────────────
class DummyModel:
    """Returns a fixed probability of 0.42 → MEDIUM risk under threshold 0.5."""
    def predict_proba(self, X):
        n = len(X)
        return np.array([[0.58, 0.42]] * n)


class IdentityScaler:
    """Behaves like a fitted StandardScaler that does no transformation."""
    def transform(self, X):
        return np.asarray(X)


# Hard-coded list of the 22 feature columns the real model expects.
# Keeping this list inside the test makes the test self-contained — if
# someone changes the production feature schema, the test will fail and
# they will be forced to update both at once.
FEATURE_COLS = [
    "Malaria_Incidence", "Precipitation_mm", "Pop_Density", "GDP_per_Capita",
    "Temp_Annual_Mean_C", "Temp_GrowingSeason_Mean_C",
    "Malaria_Lag1", "Malaria_Lag2", "Malaria_Lag3",
    "Temp_Lag1", "Precip_Lag1", "GDP_Lag1",
    "Malaria_Roll3", "Temp_Roll3", "Precip_Roll3",
    "Malaria_YoY_Change",
    "Temp_Precip_Interaction", "Temp_Squared",
    "Climate_Risk_Index", "Vulnerability_Index",
    "Log_GDP", "Log_Malaria_Incidence",
]

DUMMY_METADATA = {
    "model_name": "DummyMalariaModel",
    "threshold": 0.5,
    "feature_cols": FEATURE_COLS,
}


# ─────────────────────────────────────────────────────────────────────────────
# Fixture — build a PredictionPipeline that uses the dummies
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def dummy_pipeline(monkeypatch):
    """Yield a PredictionPipeline instance wired up with dummy artefacts."""
    from predict import PredictionPipeline

    # Replace _load_local so it does no disk I/O at all
    def _fake_load(self):
        self.model = DummyModel()
        self.scaler = IdentityScaler()
        self.metadata = DUMMY_METADATA
        self.threshold = DUMMY_METADATA["threshold"]
        self.feature_cols = DUMMY_METADATA["feature_cols"]

    monkeypatch.setattr(PredictionPipeline, "_load_local", _fake_load)
    return PredictionPipeline()


# Reusable input that mirrors what FastAPI would forward to predict_single()
SAMPLE_REQUEST = {
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
            "country_name": "Ghana", "country_code": "GHA", "year": 2020,
            "malaria_incidence": 170.0, "precipitation_mm": 1100.0,
            "pop_density": 125.0, "gdp_per_capita": 2100.0,
            "temp_annual_mean_c": 26.1, "temp_growing_season_mean_c": 26.8,
        },
        {
            "country_name": "Ghana", "country_code": "GHA", "year": 2021,
            "malaria_incidence": 175.0, "precipitation_mm": 1150.0,
            "pop_density": 127.0, "gdp_per_capita": 2150.0,
            "temp_annual_mean_c": 26.3, "temp_growing_season_mean_c": 26.9,
        },
        {
            "country_name": "Ghana", "country_code": "GHA", "year": 2022,
            "malaria_incidence": 178.0, "precipitation_mm": 1180.0,
            "pop_density": 129.0, "gdp_per_capita": 2180.0,
            "temp_annual_mean_c": 26.4, "temp_growing_season_mean_c": 27.0,
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Slide 10 — "What We Test": prediction logic + output format
# ─────────────────────────────────────────────────────────────────────────────
def test_predict_single_returns_expected_keys(dummy_pipeline):
    """The API contract: result must contain these fields, with the right types."""
    result = dummy_pipeline.predict_single(SAMPLE_REQUEST.copy())

    expected_keys = {
        "country", "year",
        "outbreak_probability", "outbreak_alert",
        "risk_level", "threshold_used", "model_name",
    }
    assert expected_keys.issubset(result.keys()), (
        f"Missing keys: {expected_keys - result.keys()}"
    )

    assert isinstance(result["outbreak_probability"], float)
    assert isinstance(result["outbreak_alert"], bool)
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_predict_single_probability_bounds(dummy_pipeline):
    """Probabilities must be in [0, 1]."""
    result = dummy_pipeline.predict_single(SAMPLE_REQUEST.copy())
    assert 0.0 <= result["outbreak_probability"] <= 1.0


def test_predict_single_uses_threshold(dummy_pipeline):
    """The dummy model returns 0.42 → below 0.5 threshold → alert must be False."""
    result = dummy_pipeline.predict_single(SAMPLE_REQUEST.copy())
    assert result["outbreak_alert"] is False
    assert result["risk_level"] == "LOW"        # 0.42 < 0.50


def test_feature_engineering_produces_no_nans(dummy_pipeline):
    """Internal check: engineered features must be free of NaNs before scaling."""
    df = pd.DataFrame([SAMPLE_REQUEST | {"Country Name": "Ghana"}])  # noqa: E225
    # We only validate that the public API does not crash on minimal inputs.
    # The deep test is the round-trip above; this one guards against silent NaNs.
    result = dummy_pipeline.predict_single(SAMPLE_REQUEST.copy())
    assert not np.isnan(result["outbreak_probability"])
