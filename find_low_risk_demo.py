"""
Helper to search for a low-risk (approved) loan scenario using the trained model.

It:
- Reuses the feature schema from `loan_classifier_final.pkl`
- Constructs increasingly conservative synthetic borrower profiles
- Calls the FastAPI `/predict` endpoint for each
- Prints the first scenario where `approved == True`, if any

Run:
    python find_low_risk_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import requests


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_classifier_final.pkl"
API_BASE = "http://127.0.0.1:8000"


def load_feature_names() -> List[str]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifacts not found at {MODEL_PATH}")
    artifacts = joblib.load(MODEL_PATH)
    return list(artifacts["features"])


def make_base_features(feature_names: List[str]) -> Dict[str, Any]:
    """Start with zeros for all features."""
    return {name: 0.0 for name in feature_names}


def build_scenario(
    feature_names: List[str],
    *,
    loan_amnt: float,
    int_rate: float,
    annual_inc: float,
    dti: float,
    grade: str,
    term_60: bool,
    home_own: str,
    purpose: str,
    state: str,
) -> Dict[str, Any]:
    """
    Construct a feature dict consistent with the model schema,
    similar to what we do in demo_predict.build_demo_features.
    """
    features = make_base_features(feature_names)

    # Core numeric fields
    features.update(
        {
            "loan_amnt": loan_amnt,
            "int_rate": int_rate,
            "installment": round(loan_amnt * 0.03, 2),
            "annual_inc": annual_inc,
            "dti": dti,
            "revol_bal": 5000,
            "revol_util": 20.0,
            "total_acc": 18,
            "open_acc": 8,
            "pub_rec": 0,
            "delinq_2yrs": 0,
            "inq_last_6mths": 0,
            "emp_length": 10,
            "credit_history_years": 15.0,
            "income_to_debt_ratio": max(annual_inc / (loan_amnt + 1e-6), 0.0),
            "available_credit_ratio": 0.8,
            "payment_to_income_ratio": (loan_amnt / max(annual_inc, 1.0)) * 0.25,
            "loan_to_income_ratio": loan_amnt / max(annual_inc, 1.0),
            "revol_util_squared": (0.20) ** 2,
            "dti_squared": (dti / 100.0) ** 2,
            "inquiry_ratio": 0.01,
            "open_acc_ratio": 0.45,
            "has_delinq": 0,
        }
    )

    # Term dummy
    features["term_ 60 months"] = 1.0 if term_60 else 0.0

    grade = grade.upper()

    # Grade one-hot
    for g in ["B", "C", "D", "E", "F", "G"]:
        key = f"grade_{g}"
        if key in features:
            features[key] = 1.0 if grade == g else 0.0

    # Subgrade: middle bucket for the grade
    sub_map = {
        "A": "sub_grade_A3",
        "B": "sub_grade_B3",
        "C": "sub_grade_C3",
        "D": "sub_grade_D3",
        "E": "sub_grade_E3",
        "F": "sub_grade_F3",
        "G": "sub_grade_G3",
    }
    for k in list(features.keys()):
        if k.startswith("sub_grade_"):
            features[k] = 0.0
    sub_feat = sub_map.get(grade)
    if sub_feat and sub_feat in features:
        features[sub_feat] = 1.0

    # Home ownership
    home_own = home_own.upper()
    for opt in ["MORTGAGE", "NONE", "OTHER", "OWN", "RENT"]:
        key = f"home_ownership_{opt}"
        if key in features:
            features[key] = 1.0 if home_own == opt else 0.0

    # Verification status (Verified)
    if "verification_status_Verified" in features:
        features["verification_status_Verified"] = 1.0

    # Purpose
    for p in [
        "credit_card",
        "debt_consolidation",
        "educational",
        "home_improvement",
        "house",
        "major_purchase",
        "medical",
        "moving",
        "other",
        "renewable_energy",
        "small_business",
        "vacation",
        "wedding",
    ]:
        key = f"purpose_{p}"
        if key in features:
            features[key] = 1.0 if purpose == p else 0.0

    # State
    state = state.upper()
    for k in list(features.keys()):
        if k.startswith("addr_state_"):
            features[k] = 0.0
    state_key = f"addr_state_{state}"
    if state_key in features:
        features[state_key] = 1.0

    return features


def call_predict(features: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{API_BASE}/predict", json={"features": features}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    feature_names = load_feature_names()

    # Define a small set of increasingly conservative scenarios
    scenarios = [
        dict(
            loan_amnt=10000,
            int_rate=10.0,
            annual_inc=120000,
            dti=10.0,
            grade="A",
            term_60=False,
            home_own="MORTGAGE",
            purpose="credit_card",
            state="CA",
        ),
        dict(
            loan_amnt=8000,
            int_rate=8.5,
            annual_inc=150000,
            dti=7.0,
            grade="A",
            term_60=False,
            home_own="MORTGAGE",
            purpose="credit_card",
            state="CA",
        ),
        dict(
            loan_amnt=5000,
            int_rate=7.0,
            annual_inc=180000,
            dti=5.0,
            grade="A",
            term_60=False,
            home_own="MORTGAGE",
            purpose="credit_card",
            state="CA",
        ),
        dict(
            loan_amnt=3000,
            int_rate=6.0,
            annual_inc=200000,
            dti=3.0,
            grade="A",
            term_60=False,
            home_own="MORTGAGE",
            purpose="credit_card",
            state="CA",
        ),
    ]

    best = None

    for i, params in enumerate(scenarios, start=1):
        print(f"\n=== Scenario {i} ===")
        feats = build_scenario(feature_names, **params)
        pred = call_predict(feats)
        print(json.dumps(pred, indent=2))

        if pred.get("approved", False):
            best = (params, pred)
            print("\nFound approved scenario.")
            break

    if best is None:
        print(
            "\nDid not find an approved scenario within this small search space.\n"
            "You can still use the scenario with the lowest default_probability "
            "from above as your 'relatively safer' example."
        )
    else:
        params, pred = best
        print("\n=== Approved scenario parameters ===")
        print(json.dumps(params, indent=2))
        print("\n=== Approved scenario prediction ===")
        print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()


