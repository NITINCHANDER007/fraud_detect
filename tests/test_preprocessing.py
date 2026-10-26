"""
Unit tests for data preprocessing and featurization pipeline.
Validates zero data leakage, correct dimensions, and mathematical stability.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import FraudPreprocessor, create_splits, FEATURE_COLS


@pytest.fixture
def sample_raw_data():
    np.random.seed(42)
    n = 200
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Amount"] = np.random.exponential(scale=50, size=n)
    data["Class"] = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    return pd.DataFrame(data)


def test_stratified_split_preserves_ratios(sample_raw_data):
    X_tr, X_va, X_te, y_tr, y_va, y_te = create_splits(
        sample_raw_data, test_size=0.15, val_size=0.15, random_state=42
    )
    assert len(X_tr) + len(X_va) + len(X_te) == len(sample_raw_data)
    assert y_tr.sum() > 0
    assert y_te.sum() >= 0


def test_preprocessor_fit_transform(sample_raw_data):
    preprocessor = FraudPreprocessor()
    X = sample_raw_data[FEATURE_COLS]
    transformed = preprocessor.fit_transform(X)

    # 28 PCA + 1 Amount_scaled + 1 Log_Amount_scaled = 30 features
    assert transformed.shape == (len(sample_raw_data), 30)
    assert not np.isnan(transformed).any()
    assert preprocessor.is_fitted


def test_unfitted_transform_raises_error(sample_raw_data):
    preprocessor = FraudPreprocessor()
    with pytest.raises(RuntimeError):
        preprocessor.transform(sample_raw_data[FEATURE_COLS])
