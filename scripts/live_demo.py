"""
live_demo.py - Runs real-time transaction scoring against the running FastAPI service.
"""

import urllib.request
import json

print("=" * 85)
print("AEGISGUARD RISK SCORING ENGINE - LIVE TRANSACTION EVALUATION DEMO")
print("=" * 85)

# Fetch sample presets
with urllib.request.urlopen("http://localhost:8000/v1/sample-transactions") as r:
    samples = json.loads(r.read().decode())

for s in samples:
    scenario = s["scenario"]
    expected = s["expected_outcome"]
    payload = s["payload"]

    req = urllib.request.Request(
        "http://localhost:8000/v1/score",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))

    print(f"\n[SCENARIO]   {scenario}")
    print(f"  Expected:  {expected}")
    print(f"  Amount:    ${payload['Amount']:,.2f}")
    print(f"  Risk Score:{res['risk_score'] * 100:.2f}% (Probability = {res['risk_score']:.4f})")
    print(f"  Decision:  Tier: {res['risk_tier']:<6} | Action: {res['recommended_action']}")
    print(f"  Latency:   {res['latency_ms']:.2f} ms")
    if res.get("top_risk_factors"):
        factors = ", ".join([f"{f['feature']} (val: {f['feature_value']})" for f in res["top_risk_factors"][:2]])
        print(f"  Top Attribution: {factors}")
    print("-" * 85)

print("\nAll live transactions scored successfully!\n")
