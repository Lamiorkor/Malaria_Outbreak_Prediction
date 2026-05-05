"""
Core prediction logic for the malaria outbreak prediction service.

This file is intentionally lightweight so it can be tested in CI/CD
without loading the full FastAPI app or real model artifacts.
"""

from numbers import Number


EXPECTED_FEATURE_COUNT = 16


def validate_input(features):
    """
    Validate one malaria prediction input row.

    Expected input:
    - a list or tuple
    - exactly 16 numeric features
    """

    if not isinstance(features, (list, tuple)):
        raise ValueError("Input features must be provided as a list or tuple.")

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Malaria input must contain exactly {EXPECTED_FEATURE_COUNT} features."
        )

    for x in features:
        if not isinstance(x, Number):
            raise ValueError("All features must be numeric.")


def predict(features, model):
    """
    Validate input and return model prediction.

    The model must have a .predict() method.
    """

    validate_input(features)
    return model.predict([features])