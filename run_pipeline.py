"""
run_pipeline.py - Master execution script for the Transaction Risk Scoring System.
Orchestrates: Ingestion -> Preprocessing -> Model Training -> MLflow Tracking -> Testing -> Server Launch.
"""

import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON_EXE = ROOT / ".venv" / "Scripts" / "python.exe"
UVICORN_EXE = ROOT / ".venv" / "Scripts" / "uvicorn.exe"


def run_cmd(cmd, desc):
    print(f"\n>>> [STEP] {desc}...")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"FAILED: {desc} exited with status {result.returncode}")
        sys.exit(result.returncode)
    print(f"COMPLETED: {desc} succeeded.\n")


def main():
    parser = argparse.ArgumentParser(description="Master Execution Pipeline")
    parser.add_argument("--train-only", action="store_true", help="Only run training and MLflow tracking")
    parser.add_argument("--test-only", action="store_true", help="Only run test suite")
    parser.add_argument("--serve-only", action="store_true", help="Only launch FastAPI server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve API on")
    args = parser.parse_args()

    if args.serve_only:
        print(f"\n>>> Starting FastAPI Server on http://localhost:{args.port} ...")
        subprocess.run([str(UVICORN_EXE), "src.api.main:app", "--host", "0.0.0.0", "--port", str(args.port), "--reload"], cwd=str(ROOT))
        return

    if args.test_only:
        run_cmd([str(PYTHON_EXE), "-m", "pytest", "-v"], "Running Automated Test Suite")
        return

    # Full Pipeline
    print("=" * 80)
    print("TRANSACTION RISK SCORING SYSTEM: END-TO-END PIPELINE")
    print("=" * 80)

    # Step 1: Ingest & Cache Data
    run_cmd([str(PYTHON_EXE), "src/data_loader.py"], "Ingesting & Caching Transaction Data")

    # Step 2: EDA & Feature Plots
    run_cmd([str(PYTHON_EXE), "src/eda.py"], "Generating Exploratory Data Visualizations")

    # Step 3: Train & Benchmark with MLflow
    run_cmd([str(PYTHON_EXE), "src/train.py"], "Training Baseline vs. XGBoost & MLflow Logging")

    # Step 4: Run Tests
    run_cmd([str(PYTHON_EXE), "-m", "pytest", "-v"], "Executing Pytest Test Suite")

    if not args.train_only:
        print(f"\n>>> All systems validated. Launching FastAPI Risk Engine on http://localhost:{args.port} ...")
        print(f">>> Interactive Dashboard: http://localhost:{args.port}")
        print(f">>> OpenAPI Swagger Docs:  http://localhost:{args.port}/docs\n")
        subprocess.run([str(UVICORN_EXE), "src.api.main:app", "--host", "0.0.0.0", "--port", str(args.port)], cwd=str(ROOT))


if __name__ == "__main__":
    main()
