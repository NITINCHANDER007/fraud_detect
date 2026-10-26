"""
schemas.py - Pydantic request and response schemas for real-time transaction risk scoring.
Includes strict validation, documentation, and representative transaction examples.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
import uuid


class TransactionInput(BaseModel):
    """
    Standard transaction payload representing an incoming payment card authorization request.
    Features V1 through V28 represent confidential PCA attributes (velocity, geolocation, token).
    """
    transaction_id: Optional[str] = Field(
        default_factory=lambda: f"tx_{uuid.uuid4().hex[:10]}",
        description="Unique transaction reference ID"
    )
    Amount: float = Field(..., ge=0.0, description="Transaction amount in currency units (e.g. USD)")
    V1: float = Field(..., description="PCA component V1")
    V2: float = Field(..., description="PCA component V2")
    V3: float = Field(..., description="PCA component V3")
    V4: float = Field(..., description="PCA component V4")
    V5: float = Field(..., description="PCA component V5")
    V6: float = Field(..., description="PCA component V6")
    V7: float = Field(..., description="PCA component V7")
    V8: float = Field(..., description="PCA component V8")
    V9: float = Field(..., description="PCA component V9")
    V10: float = Field(..., description="PCA component V10")
    V11: float = Field(..., description="PCA component V11")
    V12: float = Field(..., description="PCA component V12")
    V13: float = Field(..., description="PCA component V13")
    V14: float = Field(..., description="PCA component V14")
    V15: float = Field(..., description="PCA component V15")
    V16: float = Field(..., description="PCA component V16")
    V17: float = Field(..., description="PCA component V17")
    V18: float = Field(..., description="PCA component V18")
    V19: float = Field(..., description="PCA component V19")
    V20: float = Field(..., description="PCA component V20")
    V21: float = Field(..., description="PCA component V21")
    V22: float = Field(..., description="PCA component V22")
    V23: float = Field(..., description="PCA component V23")
    V24: float = Field(..., description="PCA component V24")
    V25: float = Field(..., description="PCA component V25")
    V26: float = Field(..., description="PCA component V26")
    V27: float = Field(..., description="PCA component V27")
    V28: float = Field(..., description="PCA component V28")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "tx_demo_sample",
                "Amount": 149.62,
                "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
                "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
                "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
                "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
                "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
                "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
                "V25": 0.1285, "V26": -0.1891, "V27": 0.1336, "V28": -0.0211
            }
        }
    )


class RiskFactor(BaseModel):
    feature: str = Field(..., description="Feature identifier")
    contribution: float = Field(..., description="Relative contribution or importance")
    feature_value: float = Field(..., description="Raw value of feature in transaction")


class RiskScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float = Field(..., description="Calibrated posterior probability of fraud (0.0 to 1.0)")
    is_fraud: bool = Field(..., description="Binary classification at current operating threshold")
    risk_tier: str = Field(..., description="LOW, MEDIUM, or HIGH risk tier")
    recommended_action: str = Field(..., description="APPROVE, MANUAL_REVIEW (Step-Up 2FA), or DECLINE")
    decision_threshold: float = Field(..., description="Threshold applied for binary classification")
    latency_ms: float = Field(..., description="Inference execution latency in milliseconds")
    model_version: str = Field(..., description="Deployed model artifact identifier")
    top_risk_factors: List[RiskFactor] = Field(default_factory=list, description="Top discriminative features")


class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput] = Field(..., min_length=1, max_length=5000)


class BatchRiskScoreResponse(BaseModel):
    total_transactions: int
    flagged_fraud_count: int
    review_required_count: int
    approved_count: int
    total_latency_ms: float
    average_latency_per_tx_ms: float
    results: List[RiskScoreResponse]


class ModelInfoResponse(BaseModel):
    model_version: str
    training_timestamp: str
    dataset: Dict[str, Any]
    optimal_threshold: float
    risk_thresholds: Dict[str, float]
    features: List[str]
    top_features: List[Dict[str, Any]]
    benchmarks: Dict[str, Any]
