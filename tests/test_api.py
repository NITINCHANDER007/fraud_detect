"""
Integration tests for FastAPI endpoints: /health, /v1/model-info, /v1/score, /v1/batch-score.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "XGBoost" in data["engine"]


def test_model_info_endpoint(client):
    response = client.get("/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "optimal_threshold" in data
    assert "benchmarks" in data
    assert data["optimal_threshold"] > 0


def test_score_legitimate_transaction(client):
    payload = {
        "transaction_id": "tx_test_legit",
        "Amount": 14.50,
        "V1": -0.428, "V2": 0.812, "V3": 1.250, "V4": -0.320,
        "V5": 0.150, "V6": -0.210, "V7": 0.380, "V8": 0.050,
        "V9": -0.120, "V10": -0.250, "V11": -0.410, "V12": -0.050,
        "V13": -0.220, "V14": 0.150, "V15": 0.890, "V16": 0.210,
        "V17": -0.150, "V18": 0.020, "V19": -0.110, "V20": -0.050,
        "V21": -0.180, "V22": -0.450, "V23": 0.020, "V24": 0.120,
        "V25": -0.150, "V26": 0.080, "V27": 0.020, "V28": 0.010
    }
    response = client.post("/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "tx_test_legit"
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["recommended_action"] in ["APPROVE", "MANUAL_REVIEW", "DECLINE"]
    assert data["latency_ms"] >= 0.0


def test_score_fraudulent_transaction(client):
    payload = {
        "transaction_id": "tx_test_fraud",
        "Amount": 1.00,
        "V1": -4.3979, "V2": 1.3583, "V3": -2.5928, "V4": 2.6797,
        "V5": -1.1281, "V6": -1.6065, "V7": -2.4793, "V8": 1.2159,
        "V9": -0.7996, "V10": -4.6140, "V11": 4.4557, "V12": -7.3831,
        "V13": 0.3807, "V14": -8.5779, "V15": -0.0890, "V16": -6.1578,
        "V17": -12.9168, "V18": -4.3862, "V19": 1.1265, "V20": -0.1716,
        "V21": 0.5735, "V22": 0.1769, "V23": -0.4362, "V24": -0.0535,
        "V25": 0.2524, "V26": -0.6574, "V27": -0.8271, "V28": -0.7303
    }
    response = client.post("/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_score"] > 0.70
    assert data["risk_tier"] == "HIGH"
    assert data["recommended_action"] == "DECLINE"
    assert data["is_fraud"] is True


def test_batch_scoring(client):
    sample = {f"V{i}": 0.0 for i in range(1, 29)} | {"Amount": 50.0}
    batch_payload = {
        "transactions": [
            dict(sample, transaction_id=f"batch_{i}") for i in range(10)
        ]
    }
    response = client.post("/v1/batch-score", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 10
    assert len(data["results"]) == 10
    assert data["average_latency_per_tx_ms"] >= 0.0


def test_sample_transactions_endpoint(client):
    response = client.get("/v1/sample-transactions")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) >= 4
