"""
Unit tests for model inference, calibration bounds, and risk discrimination.
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import FraudPreprocessor


@pytest.fixture
def preprocessor_and_model():
    prep_path = ROOT / "models" / "preprocessor.pkl"
    model_path = ROOT / "models" / "xgboost_fraud_model.pkl"

    assert prep_path.exists(), "Preprocessor pickle not found. Run train.py first."
    assert model_path.exists(), "XGBoost model pickle not found. Run train.py first."

    prep = FraudPreprocessor.load(prep_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return prep, model


def test_model_inference_probability_bounds(preprocessor_and_model):
    prep, model = preprocessor_and_model
    # Normal synthetic legitimate transaction
    legit_df = pd.DataFrame([{f"V{i}": 0.0 for i in range(1, 29)} | {"Amount": 25.0}])
    transformed = prep.transform(legit_df)
    prob = model.predict_proba(transformed)[0, 1]

    assert 0.0 <= prob <= 1.0
    assert prob < 0.35, f"Expected low risk score for normal transaction, got {prob}"


def test_model_discriminates_extreme_fraud(preprocessor_and_model):
    prep, model = preprocessor_and_model
    # Signature of known severe fraud from test set (V14, V10, V17 strongly negative)
    fraud_df = pd.DataFrame([{
        "Amount": 1.0,
        "V1": -4.39, "V2": 1.35, "V3": -2.59, "V4": 2.67,
        "V5": -1.12, "V6": -1.60, "V7": -2.47, "V8": 1.21,
        "V9": -0.80, "V10": -4.61, "V11": 4.45, "V12": -7.38,
        "V13": 0.38, "V14": -8.57, "V15": -0.09, "V16": -6.15,
        "V17": -12.91, "V18": -4.38, "V19": 1.12, "V20": -0.17,
        "V21": 0.57, "V22": 0.17, "V23": -0.43, "V24": -0.05,
        "V25": 0.25, "V26": -0.65, "V27": -0.82, "V28": -0.73
    }])
    transformed = prep.transform(fraud_df)
    prob = model.predict_proba(transformed)[0, 1]

    assert prob > 0.70, f"Expected high fraud probability for severe attack, got {prob}"
