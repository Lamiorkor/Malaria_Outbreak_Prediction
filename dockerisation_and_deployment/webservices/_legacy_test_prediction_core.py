import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent))

from prediction_core import predict, validate_input, EXPECTED_FEATURE_COUNT


class DummyModel:
    def predict(self, X):
        return [0]


class ShapeCheckingDummyModel:
    def predict(self, X):
        assert len(X) == 1
        assert len(X[0]) == EXPECTED_FEATURE_COUNT
        return [1]


def test_valid_prediction():
    sample = [1.0] * EXPECTED_FEATURE_COUNT
    model = DummyModel()

    result = predict(sample, model)

    assert result == [0]


def test_invalid_feature_length():
    sample = [1.0] * (EXPECTED_FEATURE_COUNT - 1)

    with pytest.raises(ValueError):
        validate_input(sample)


def test_non_numeric_input():
    sample = [1.0] * EXPECTED_FEATURE_COUNT
    sample[3] = "wrong"

    with pytest.raises(ValueError):
        validate_input(sample)


def test_model_input_shape():
    sample = [1.0] * EXPECTED_FEATURE_COUNT
    model = ShapeCheckingDummyModel()

    result = predict(sample, model)

    assert result == [1]


def test_input_must_be_list_or_tuple():
    sample = "not a valid input"

    with pytest.raises(ValueError):
        validate_input(sample)