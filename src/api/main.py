"""
main.py - Production-ready FastAPI service for Real-Time Transaction Risk Scoring.
Delivers sub-millisecond scoring latencies, multi-tier risk decisions, and interactive dashboard.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import pickle
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging

from src.preprocessing import FraudPreprocessor
from src.api.schemas import (
    TransactionInput,
    RiskScoreResponse,
    RiskFactor,
    BatchTransactionInput,
    BatchRiskScoreResponse,
    ModelInfoResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# In-memory model state
app_state: Dict[str, Any] = {
    "model": None,
    "baseline_model": None,
    "preprocessor": None,
    "metadata": None,
    "optimal_threshold": 0.7466,
    "low_risk_ceiling": 0.20,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts into memory on startup and warm up the inference engine."""
    logger.info("Initializing Transaction Risk Scoring API Engine...")

    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    model_path = MODELS_DIR / "xgboost_fraud_model.pkl"
    baseline_path = MODELS_DIR / "baseline_logistic_regression.pkl"
    metadata_path = MODELS_DIR / "model_metadata.json"

    if not preprocessor_path.exists() or not model_path.exists():
        logger.warning("Models not found. Training model pipeline automatically...")
        from src.train import train_and_track
        train_and_track()

    # Load preprocessor
    app_state["preprocessor"] = FraudPreprocessor.load(preprocessor_path)

    # Load XGBoost model
    with open(model_path, "rb") as f:
        app_state["model"] = pickle.load(f)
    logger.info("Production XGBoost model loaded.")

    # Load Baseline Logistic Regression
    if baseline_path.exists():
        with open(baseline_path, "rb") as f:
            app_state["baseline_model"] = pickle.load(f)

    # Load Metadata
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            app_state["metadata"] = json.load(f)
            app_state["optimal_threshold"] = app_state["metadata"].get("optimal_threshold", 0.7466)
            thresholds = app_state["metadata"].get("risk_thresholds", {})
            app_state["low_risk_ceiling"] = thresholds.get("low_risk_ceiling", 0.20)

    # Inference engine warmup
    dummy_df = pd.DataFrame([{f"V{i}": 0.0 for i in range(1, 29)} | {"Amount": 50.0}])
    dummy_tensor = app_state["preprocessor"].transform(dummy_df)
    _ = app_state["model"].predict_proba(dummy_tensor)
    logger.info("Inference engine warm-up complete. Service ready to accept traffic.")

    yield
    logger.info("Shutting down Transaction Risk Scoring API Engine.")


app = FastAPI(
    title="Transaction Risk Scoring API",
    description="Real-Time Payment Fraud Detection and Decision Engine powered by XGBoost and MLflow.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _compute_risk_factors(tx_dict: Dict[str, float], transformed_row: np.ndarray) -> List[RiskFactor]:
    """Identify top features contributing towards high risk score."""
    model = app_state["model"]
    importances = model.feature_importances_
    feature_names = app_state["preprocessor"].feature_names_out_

    # Calculate contribution as importance * absolute z-score/deviation
    deviations = np.abs(transformed_row)
    contributions = importances * deviations

    top_indices = np.argsort(contributions)[::-1][:3]
    factors = []
    for idx in top_indices:
        feat_name = feature_names[idx]
        raw_key = feat_name.replace("_scaled", "")
        raw_val = tx_dict.get(raw_key, float(transformed_row[idx]))
        factors.append(
            RiskFactor(
                feature=feat_name,
                contribution=float(round(contributions[idx], 4)),
                feature_value=float(round(raw_val, 4)),
            )
        )
    return factors


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health probe for Kubernetes and load balancers."""
    is_ready = app_state["model"] is not None and app_state["preprocessor"] is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "model_version": app_state["metadata"]["model_version"] if app_state["metadata"] else "unknown",
        "engine": "XGBoost",
        "timestamp": time.time(),
    }


@app.get("/v1/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
def get_model_info():
    """Returns production model metadata, benchmark performance metrics, and threshold settings."""
    if not app_state["metadata"]:
        raise HTTPException(status_code=500, detail="Model metadata is not loaded.")
    return app_state["metadata"]


@app.post("/v1/score", response_model=RiskScoreResponse, tags=["Scoring"])
def score_transaction(transaction: TransactionInput):
    """
    Real-time fraud scoring endpoint.
    Calculates probability of fraud, risk tier, recommended authorization action, and latency.
    """
    if not app_state["model"] or not app_state["preprocessor"]:
        raise HTTPException(status_code=503, detail="Risk scoring model is not initialized.")

    start_time = time.perf_counter()
    tx_dict = transaction.model_dump()
    tx_id = tx_dict.get("transaction_id") or "tx_unknown"

    # Prepare DataFrame
    row_data = {f"V{i}": tx_dict[f"V{i}"] for i in range(1, 29)}
    row_data["Amount"] = float(tx_dict["Amount"])
    df_row = pd.DataFrame([row_data])

    # Transform through leak-free preprocessor
    transformed = app_state["preprocessor"].transform(df_row)

    # Score probability
    prob = float(app_state["model"].predict_proba(transformed)[0, 1])
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Decision Matrix
    threshold = app_state["optimal_threshold"]
    low_ceiling = app_state["low_risk_ceiling"]

    if prob < low_ceiling:
        risk_tier = "LOW"
        action = "APPROVE"
        is_fraud = False
    elif prob < threshold:
        risk_tier = "MEDIUM"
        action = "MANUAL_REVIEW"
        is_fraud = False
    else:
        risk_tier = "HIGH"
        action = "DECLINE"
        is_fraud = True

    # Identify top risk factors
    top_factors = _compute_risk_factors(tx_dict, transformed[0])

    return RiskScoreResponse(
        transaction_id=tx_id,
        risk_score=round(prob, 4),
        is_fraud=is_fraud,
        risk_tier=risk_tier,
        recommended_action=action,
        decision_threshold=round(threshold, 4),
        latency_ms=round(latency_ms, 3),
        model_version=app_state["metadata"]["model_version"] if app_state["metadata"] else "v1.0.0",
        top_risk_factors=top_factors,
    )


@app.post("/v1/batch-score", response_model=BatchRiskScoreResponse, tags=["Scoring"])
def batch_score_transactions(batch: BatchTransactionInput):
    """
    Vectorized high-throughput batch scoring endpoint.
    Scores thousands of transactions concurrently with sub-millisecond per-transaction throughput.
    """
    if not app_state["model"] or not app_state["preprocessor"]:
        raise HTTPException(status_code=503, detail="Risk scoring model is not initialized.")

    start_time = time.perf_counter()
    transactions = batch.transactions
    total_tx = len(transactions)

    # Vectorized conversion
    records = []
    ids = []
    for tx in transactions:
        t_dict = tx.model_dump()
        ids.append(t_dict.get("transaction_id") or "tx_unknown")
        row = {f"V{i}": t_dict[f"V{i}"] for i in range(1, 29)}
        row["Amount"] = float(t_dict["Amount"])
        records.append(row)

    df_batch = pd.DataFrame(records)
    transformed_batch = app_state["preprocessor"].transform(df_batch)
    probs = app_state["model"].predict_proba(transformed_batch)[:, 1]

    threshold = app_state["optimal_threshold"]
    low_ceiling = app_state["low_risk_ceiling"]
    model_ver = app_state["metadata"]["model_version"] if app_state["metadata"] else "v1.0.0"

    results = []
    flagged_count = 0
    review_count = 0
    approved_count = 0

    for i in range(total_tx):
        p = float(probs[i])
        if p < low_ceiling:
            tier = "LOW"
            act = "APPROVE"
            is_f = False
            approved_count += 1
        elif p < threshold:
            tier = "MEDIUM"
            act = "MANUAL_REVIEW"
            is_f = False
            review_count += 1
        else:
            tier = "HIGH"
            act = "DECLINE"
            is_f = True
            flagged_count += 1

        results.append(
            RiskScoreResponse(
                transaction_id=ids[i],
                risk_score=round(p, 4),
                is_fraud=is_f,
                risk_tier=tier,
                recommended_action=act,
                decision_threshold=round(threshold, 4),
                latency_ms=0.0,
                model_version=model_ver,
                top_risk_factors=[],
            )
        )

    total_latency = (time.perf_counter() - start_time) * 1000.0
    avg_latency = total_latency / total_tx if total_tx > 0 else 0.0

    return BatchRiskScoreResponse(
        total_transactions=total_tx,
        flagged_fraud_count=flagged_count,
        review_required_count=review_count,
        approved_count=approved_count,
        total_latency_ms=round(total_latency, 2),
        average_latency_per_tx_ms=round(avg_latency, 3),
        results=results,
    )


@app.get("/v1/sample-transactions", tags=["Testing"])
def get_sample_transactions():
    """Returns curated representative transaction payloads for live testing in UI or curl."""
    samples = [
        {
            "scenario": "Legitimate Low-Value Daily Purchase (Coffee/Groceries)",
            "expected_outcome": "APPROVE (Risk < 0.05)",
            "payload": {
                "transaction_id": "tx_legit_001",
                "Amount": 14.50,
                "V1": -0.428, "V2": 0.812, "V3": 1.250, "V4": -0.320,
                "V5": 0.150, "V6": -0.210, "V7": 0.380, "V8": 0.050,
                "V9": -0.120, "V10": -0.250, "V11": -0.410, "V12": -0.050,
                "V13": -0.220, "V14": 0.150, "V15": 0.890, "V16": 0.210,
                "V17": -0.150, "V18": 0.020, "V19": -0.110, "V20": -0.050,
                "V21": -0.180, "V22": -0.450, "V23": 0.020, "V24": 0.120,
                "V25": -0.150, "V26": 0.080, "V27": 0.020, "V28": 0.010
            }
        },
        {
            "scenario": "Legitimate High-Value Electronics Purchase",
            "expected_outcome": "APPROVE (Risk < 0.15)",
            "payload": {
                "transaction_id": "tx_legit_high_value",
                "Amount": 849.00,
                "V1": 1.152, "V2": -0.214, "V3": 0.582, "V4": 0.410,
                "V5": -0.620, "V6": -0.180, "V7": -0.350, "V8": 0.110,
                "V9": 0.650, "V10": -0.150, "V11": 0.220, "V12": 0.450,
                "V13": 0.120, "V14": -0.210, "V15": 0.450, "V16": 0.180,
                "V17": -0.050, "V18": -0.120, "V19": 0.080, "V20": -0.080,
                "V21": -0.120, "V22": -0.280, "V23": 0.050, "V24": 0.210,
                "V25": 0.250, "V26": 0.350, "V27": -0.020, "V28": 0.010
            }
        },
        {
            "scenario": "Confirmed Fraud: Automated Card-Testing Bot Pattern",
            "expected_outcome": "DECLINE (Risk > 0.85)",
            "payload": {
                "transaction_id": "tx_fraud_bot_01",
                "Amount": 1.00,
                "V1": -4.3979, "V2": 1.3583, "V3": -2.5928, "V4": 2.6797,
                "V5": -1.1281, "V6": -1.6065, "V7": -2.4793, "V8": 1.2159,
                "V9": -0.7996, "V10": -4.6140, "V11": 4.4557, "V12": -7.3831,
                "V13": 0.3807, "V14": -8.5779, "V15": -0.0890, "V16": -6.1578,
                "V17": -12.9168, "V18": -4.3862, "V19": 1.1265, "V20": -0.1716,
                "V21": 0.5735, "V22": 0.1769, "V23": -0.4362, "V24": -0.0535,
                "V25": 0.2524, "V26": -0.6574, "V27": -0.8271, "V28": -0.7303
            }
        },
        {
            "scenario": "Confirmed Fraud: High-Dollar Account Takeover Attack",
            "expected_outcome": "DECLINE (Risk > 0.90)",
            "payload": {
                "transaction_id": "tx_fraud_ato_02",
                "Amount": 1289.50,
                "V1": -2.3122, "V2": 1.9519, "V3": -1.6098, "V4": 3.9979,
                "V5": -0.5221, "V6": -1.4265, "V7": -2.5373, "V8": 1.3916,
                "V9": -2.7700, "V10": -2.7722, "V11": 3.2020, "V12": -2.8999,
                "V13": -0.5952, "V14": -4.2892, "V15": 0.3897, "V16": -1.1407,
                "V17": -2.8300, "V18": -0.0168, "V19": 0.4169, "V20": 0.1269,
                "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3201,
                "V25": 0.0445, "V26": 0.1778, "V27": 0.2611, "V28": -0.1432
            }
        }
    ]
    return samples


# Serve static web dashboard
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_dashboard():
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Transaction Risk Scoring API. Visit /docs for Swagger specifications."}
