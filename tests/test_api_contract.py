"""
test_api_contract.py
====================
Contract tests for the FastAPI request schema (Pydantic).

These tests validate the *interface* of the prediction API — they make sure
the public contract (field names, types, bounds) does not change silently.
They do NOT load the real model, hit the database, or talk to MLflow,
keeping them aligned with Slide 11 (Dummy Model Concept).

Pattern: we import only the `CountryRecord` schema from `main.py` so that we
do not trigger the FastAPI app's lifespan (which would try to load the model
artefacts that are absent in the CI runner).
"""

import importlib
import pytest
from pydantic import ValidationError


# ─────────────────────────────────────────────────────────────────────────────
# Helper: import only the schema class without starting the FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
def _load_schema():
    main = importlib.import_module("main")
    return main.CountryRecord


CountryRecord = _load_schema()


# A minimal valid payload reused across tests
VALID_PAYLOAD = {
    "country_name": "Ghana",
    "country_code": "GHA",
    "year": 2023,
    "malaria_incidence": 180.5,
    "precipitation_mm": 1200.0,
    "pop_density": 130.0,
    "gdp_per_capita": 2200.0,
    "temp_annual_mean_c": 26.5,
    "temp_growing_season_mean_c": 27.1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────
def test_valid_payload_is_accepted():
    record = CountryRecord(**VALID_PAYLOAD)
    assert record.country_name == "Ghana"
    assert record.year == 2023


def test_history_is_optional():
    record = CountryRecord(**VALID_PAYLOAD)
    assert record.history is None


def test_history_can_be_attached():
    payload = {**VALID_PAYLOAD, "history": [VALID_PAYLOAD]}
    record = CountryRecord(**payload)
    assert isinstance(record.history, list)
    assert len(record.history) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Bounds / type validation 
# ─────────────────────────────────────────────────────────────────────────────
def test_year_below_lower_bound_rejected():
    payload = {**VALID_PAYLOAD, "year": 1999}      # schema requires year ≥ 2000
    with pytest.raises(ValidationError):
        CountryRecord(**payload)


def test_year_above_upper_bound_rejected():
    payload = {**VALID_PAYLOAD, "year": 2050}      # schema requires year ≤ 2035
    with pytest.raises(ValidationError):
        CountryRecord(**payload)


def test_negative_precipitation_rejected():
    payload = {**VALID_PAYLOAD, "precipitation_mm": -10.0}
    with pytest.raises(ValidationError):
        CountryRecord(**payload)


def test_zero_population_density_rejected():
    """Pop_Density must be strictly positive (gt=0) — division-by-zero protection."""
    payload = {**VALID_PAYLOAD, "pop_density": 0.0}
    with pytest.raises(ValidationError):
        CountryRecord(**payload)


def test_missing_required_field_rejected():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "country_name"}
    with pytest.raises(ValidationError):
        CountryRecord(**payload)


def test_wrong_type_rejected():
    payload = {**VALID_PAYLOAD, "year": "twenty-twenty-three"}
    with pytest.raises(ValidationError):
        CountryRecord(**payload)
