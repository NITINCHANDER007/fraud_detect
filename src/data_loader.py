"""
data_loader.py - Data ingestion and caching module for transaction fraud detection.
Fetches the public Credit Card Fraud Detection dataset from OpenML and caches locally.
"""

from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_FILE = DATA_DIR / "creditcard.csv"


def load_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Loads the credit card fraud dataset.
    If cached locally at data/raw/creditcard.csv, loads from disk.
    Otherwise fetches from OpenML (data_id=43627) and saves to disk.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists() and not force_download:
        logger.info(f"Loading cached dataset from {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)
    else:
        logger.info("Fetching Credit Card Fraud dataset from OpenML (data_id=43627)...")
        bunch = fetch_openml(data_id=43627, as_frame=True, parser="auto")
        df = bunch.frame.copy()

        # Ensure Class is integer (0 = Legitimate, 1 = Fraud)
        df["Class"] = df["Class"].astype(int)

        logger.info(f"Caching dataset to {DATA_FILE} ({df.shape[0]} rows, {df.shape[1]} cols)...")
        df.to_csv(DATA_FILE, index=False)

    # Validate schema
    expected_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")

    logger.info(
        f"Dataset loaded: {df.shape[0]:,} transactions | "
        f"Legitimate: {(df['Class'] == 0).sum():,} | "
        f"Fraud: {(df['Class'] == 1).sum():,} "
        f"({(df['Class'] == 1).mean() * 100:.3f}%)"
    )
    return df


if __name__ == "__main__":
    df = load_dataset()
    print("Dataset head:")
    print(df.head(3))
