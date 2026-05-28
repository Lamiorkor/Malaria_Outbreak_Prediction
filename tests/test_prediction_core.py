"""
test_prediction_core.py
=======================
Unit tests for the malaria prediction-core validation logic.


"""

import pytest

from prediction_core import predict, validate_input, EXPECTED_FEATURE_COUNT


# ────────────────────────────────────────────────────────────
# Dummy models that replace the real LogisticRegression in CI 
# ────────────────────────────────────────────────────────────
class DummyModel:
    """Stand-in for sklearn LogisticRegression; always predicts 0 (no outbreak)."""
    def predict(self, X):
        return [0]


class ShapeCheckingDummyModel:
    """Verifies that exactly one record with the right number of features arrives."""
    def predict(self, X):
        assert len(X) == 1, "predict() must be called with exactly one row"
        assert len(X[0]) == EXPECTED_FEATURE_COUNT, (
            f"row must have {EXPECTED_FEATURE_COUNT} features"
        )
        return [1]





# 1. Prediction logic
def test_valid_prediction_returns_model_output():
    """A correctly-shaped input must reach the model and return its output."""
    sample = [1.0] * EXPECTED_FEATURE_COUNT
    assert predict(sample, DummyModel()) == [0]


def test_model_receives_correct_shape():
    """The validator must pass through exactly one row of N features."""
    sample = [0.5] * EXPECTED_FEATURE_COUNT
    assert predict(sample, ShapeCheckingDummyModel()) == [1]


# 2. Input validation (shape + types)
def test_rejects_wrong_feature_count():
    """One feature short → ValueError, not a model crash downstream."""
    bad_sample = [1.0] * (EXPECTED_FEATURE_COUNT - 1)
    with pytest.raises(ValueError, match="exactly"):
        validate_input(bad_sample)


def test_rejects_extra_features():
    """Too many features must also fail validation."""
    bad_sample = [1.0] * (EXPECTED_FEATURE_COUNT + 3)
    with pytest.raises(ValueError):
        validate_input(bad_sample)


def test_rejects_non_numeric_value():
    """A single non-numeric entry must be caught before reaching the model."""
    sample = [1.0] * EXPECTED_FEATURE_COUNT
    sample[5] = "not a number"
    with pytest.raises(ValueError, match="numeric"):
        validate_input(sample)


def test_rejects_non_sequence_input():
    """Strings, dicts, scalars must all be rejected."""
    for bad in ["a string", 42, {"feature": 1.0}, None]:
        with pytest.raises(ValueError):
            validate_input(bad)


# 3. Error handling — invalid input must NOT call the model
def test_invalid_input_never_calls_model():
    """If validation fails, the dummy model.predict must not be reached."""
    class ExplodingModel:
        def predict(self, X):
            raise AssertionError("predict() must not be called on invalid input")

    with pytest.raises(ValueError):
        predict([1.0] * 3, ExplodingModel())   # wrong length on purpose
