"""
Quick helper script to exercise the /predict endpoint with a deterministic feature vector.

This avoids having to recreate the entire feature-engineering pipeline. We:
- Load the trained model to retrieve the exact feature names.
- Populate a dictionary with zeros for every feature.
- Fill in realistic values for the most important numeric/dummy columns.
- Call /predict and optionally use Ollama to explain the result.

Run:
    python demo_predict.py --explain
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import requests


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "loan_classifier_final.pkl"
API_BASE = "http://127.0.0.1:8000"

OLLAMA_BASE = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3"


def build_demo_features() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifacts not found at {MODEL_PATH}")

    artifacts = joblib.load(MODEL_PATH)
    feature_names = artifacts["features"]

    # Start with zeros for every feature
    features: Dict[str, Any] = {name: 0.0 for name in feature_names}

    # Fill numeric features with plausible values
    features.update(
        {
            "loan_amnt": 15000,
            "int_rate": 13.5,
            "installment": 505.0,
            "annual_inc": 80000,
            "dti": 18.0,
            "revol_bal": 12000,
            "revol_util": 45.0,
            "total_acc": 24,
            "open_acc": 12,
            "pub_rec": 0,
            "delinq_2yrs": 0,
            "inq_last_6mths": 1,
            "emp_length": 6,
            "credit_history_years": 12.0,
            "income_to_debt_ratio": 4.0,
            "available_credit_ratio": 0.65,
            "payment_to_income_ratio": 0.15,
            "loan_to_income_ratio": 0.19,
            "revol_util_squared": 0.45**2,
            "dti_squared": 0.18**2,
            "inquiry_ratio": 0.05,
            "open_acc_ratio": 0.55,
            "has_delinq": 0,
        }
    )

    # Term dummy (only have 60 months dummy; 36 months => 0)
    features["term_ 60 months"] = 0

    # Grade/subgrade
    features["grade_B"] = 1
    features["sub_grade_B3"] = 1

    # Home ownership
    features["home_ownership_MORTGAGE"] = 1

    # Verification status
    features["verification_status_Verified"] = 1

    # Purpose
    features["purpose_debt_consolidation"] = 1

    # State
    features["addr_state_CA"] = 1

    return features


def call_predict(features: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{API_BASE}/predict", json={"features": features}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def explain_with_ollama(features: Dict[str, Any], prediction: Dict[str, Any]) -> str:
    prompt = (
        "You are a risk officer. Explain the loan default prediction using the JSON context.\n\n"
        f"{json.dumps({'features': features, 'prediction': prediction}, indent=2)}\n\n"
        "Respond in 3-5 sentences, citing the probability and approval decision."
    )
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo /predict endpoint.")
    parser.add_argument("--explain", action="store_true", help="Use Ollama to verbalize the prediction.")
    args = parser.parse_args()

    features = build_demo_features()
    prediction = call_predict(features)

    print("=== /predict raw response ===")
    print(json.dumps(prediction, indent=2))

    if args.explain:
        print("\n=== LLM explanation ===")
        try:
            explanation = explain_with_ollama(features, prediction)
            print(explanation)
        except Exception as exc:
            print(f"Failed to call Ollama: {exc}")


if __name__ == "__main__":
    main()


