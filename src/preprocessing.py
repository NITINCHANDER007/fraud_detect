"""
preprocessing.py - Strict featurization ordering and data pipeline.
Prevents data leakage by fitting transformations solely on the training split.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Tuple, Dict, Any
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


class FraudPreprocessor:
    """
    Production-grade preprocessor for transaction risk scoring.
    Transforms raw transaction features into model-ready tensors while strictly
    guaranteeing no data leakage across training, validation, and inference boundaries.
    """

    def __init__(self):
        self.amount_scaler = RobustScaler()
        self.log_amount_scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names_in_ = FEATURE_COLS
        self.feature_names_out_ = [f"V{i}" for i in range(1, 29)] + [
            "Amount_scaled",
            "Log_Amount_scaled",
        ]

    def fit(self, X: pd.DataFrame) -> "FraudPreprocessor":
        """Fit scalers strictly on training data."""
        if "Amount" not in X.columns:
            raise ValueError("Input dataframe must contain 'Amount' column.")

        amounts = X[["Amount"]].values
        log_amounts = np.log1p(np.maximum(amounts, 0))

        self.amount_scaler.fit(amounts)
        self.log_amount_scaler.fit(log_amounts)
        self.is_fitted = True
        logger.info("FraudPreprocessor fitted successfully on training data.")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features into scaled numpy array ready for model inference."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")

        # Ensure V1..V28 and Amount exist
        for col in self.feature_names_in_:
            if col not in X.columns:
                raise ValueError(f"Missing required feature column: {col}")

        # V1 through V28 are already zero-mean PCA components
        pca_features = X[[f"V{i}" for i in range(1, 29)]].values

        amounts = X[["Amount"]].values
        log_amounts = np.log1p(np.maximum(amounts, 0))

        amount_scaled = self.amount_scaler.transform(amounts)
        log_amount_scaled = self.log_amount_scaler.transform(log_amounts)

        # Combine all features: [V1..V28, Amount_scaled, Log_Amount_scaled] -> 30 features
        transformed = np.hstack([pca_features, amount_scaled, log_amount_scaled])
        return transformed

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, filepath: Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath: Path) -> "FraudPreprocessor":
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Preprocessor loaded from {filepath}")
        return obj


def create_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Performs stratified train / validation / test split to preserve rare fraud class ratio.
    Default: 70% Train, 15% Validation, 15% Test.
    """
    logger.info("Splitting dataset into Train (70%), Validation (15%), Test (15%)...")
    X = df[FEATURE_COLS].copy()
    y = df["Class"].copy()

    # Step 1: Split into Train (70%) and Temp (30%)
    temp_size = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=random_state, stratify=y
    )

    # Step 2: Split Temp into Validation (50% of 30% = 15%) and Test (50% of 30% = 15%)
    val_ratio = val_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_ratio), random_state=random_state, stratify=y_temp
    )

    logger.info(
        f"Split complete:\n"
        f"  Train:      {len(X_train):,} samples (Fraud: {y_train.sum()} = {y_train.mean()*100:.3f}%)\n"
        f"  Validation: {len(X_val):,} samples (Fraud: {y_val.sum()} = {y_val.mean()*100:.3f}%)\n"
        f"  Test:       {len(X_test):,} samples (Fraud: {y_test.sum()} = {y_test.mean()*100:.3f}%)"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    from src.data_loader import load_dataset

    df = load_dataset()
    X_tr, X_va, X_te, y_tr, y_va, y_te = create_splits(df)
    preprocessor = FraudPreprocessor()
    X_tr_scaled = preprocessor.fit_transform(X_tr)
    X_va_scaled = preprocessor.transform(X_va)
    X_te_scaled = preprocessor.transform(X_te)
    print(f"Transformed shapes: Train={X_tr_scaled.shape}, Val={X_va_scaled.shape}, Test={X_te_scaled.shape}")
