"""
Helper to test a clearly risky loan profile and ensure it gets REJECTED.

Takes a partial feature dict and fills ALL 134 features with realistic defaults,
so the model sees a complete risky profile.
"""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import requests

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_classifier_final.pkl"
API_BASE = "http://127.0.0.1:8000"


def build_complete_risky_features(partial: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take a partial feature dict and fill ALL 134 features.
    Missing features get risky defaults to ensure rejection.
    """
    artifacts = joblib.load(MODEL_PATH)
    feature_names = artifacts["features"]

    # Start with zeros for all features
    features: Dict[str, Any] = {name: 0.0 for name in feature_names}

    # Override with user-provided values
    features.update(partial)

    # Fill missing numeric features with RISKY defaults
    if "installment" not in partial:
        loan_amt = partial.get("loan_amnt", 950000)
        int_rate = partial.get("int_rate", 16.5) / 100.0
        term_months = 60 if partial.get("term_ 60 months", 0) == 1 else 36
        # Rough installment calculation
        features["installment"] = round(
            loan_amt * (int_rate / 12) / (1 - (1 + int_rate / 12) ** (-term_months)), 2
        )

    annual_inc = partial.get("annual_inc", 20000)
    loan_amt = partial.get("loan_amnt", 950000)
    dti = partial.get("dti", 18.0)

    # Fill missing derived features with EXTREMELY RISKY values
    defaults = {
        "revol_bal": partial.get("revol_bal", 80000),  # very high revolving debt
        "revol_util": partial.get("revol_util", 95.0),  # maxed out utilization
        "total_acc": partial.get("total_acc", 40),  # many accounts
        "open_acc": partial.get("open_acc", 25),  # many open accounts
        "pub_rec": partial.get("pub_rec", 2),  # multiple public records
        "delinq_2yrs": partial.get("delinq_2yrs", 3),  # multiple recent delinquencies
        "inq_last_6mths": partial.get("inq_last_6mths", 8),  # many recent inquiries
        "emp_length": partial.get("emp_length", 0.5),  # very short employment (< 1 year)
        "credit_history_years": partial.get("credit_history_years", 1.0),  # very short history
        "income_to_debt_ratio": max(annual_inc / (loan_amt + 1e-6), 0.01),  # extremely low
        "available_credit_ratio": 0.05,  # almost no available credit
        "payment_to_income_ratio": min((features.get("installment", 5000) / max(annual_inc, 1.0)), 0.95),  # payment is most of income
        "loan_to_income_ratio": loan_amt / max(annual_inc, 1.0),  # EXTREMELY high (47.5x)
        "revol_util_squared": (95.0 / 100.0) ** 2,  # maxed out squared
        "dti_squared": (25.0 / 100.0) ** 2,  # higher DTI squared
        "inquiry_ratio": 0.4,  # very high inquiry ratio
        "open_acc_ratio": 0.9,  # almost all accounts open
        "has_delinq": 1,  # definitely has delinquencies
    }

    for k, v in defaults.items():
        if k not in partial:
            features[k] = v

    # Ensure all one-hot features are set (zeros for missing, 1 for provided)
    # Grade dummies - use the WORST grade if not specified
    if not any(features.get(f"grade_{g}", 0) == 1 for g in ["B", "C", "D", "E", "F", "G"]):
        # If no grade set, default to G (worst possible)
        features["grade_G"] = 1
    elif features.get("grade_F", 0) == 1:
        # Upgrade to G for maximum risk
        features["grade_F"] = 0
        features["grade_G"] = 1

    # Subgrade: use worst subgrade (G5)
    for k in features.keys():
        if k.startswith("sub_grade_"):
            features[k] = 0.0
    if features.get("grade_G", 0) == 1:
        features["sub_grade_G5"] = 1.0
    elif features.get("grade_F", 0) == 1:
        features["sub_grade_F5"] = 1.0

    # Home ownership: ensure one is set
    if not any(features.get(f"home_ownership_{opt}", 0) == 1 for opt in ["MORTGAGE", "OWN", "RENT", "OTHER", "NONE"]):
        features["home_ownership_RENT"] = 1  # RENT is riskier than MORTGAGE

    # Verification status
    if "verification_status_Verified" not in partial and "verification_status_Source Verified" not in partial:
        features["verification_status_Verified"] = 0  # Not verified = riskier

    # Purpose: ensure one is set
    if not any(features.get(f"purpose_{p}", 0) == 1 for p in [
        "credit_card", "debt_consolidation", "educational", "home_improvement",
        "house", "major_purchase", "medical", "moving", "other", "renewable_energy",
        "small_business", "vacation", "wedding"
    ]):
        features["purpose_debt_consolidation"] = 1

    # State: ensure one is set
    if not any(features.get(f"addr_state_{s}", 0) == 1 for s in [
        "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID",
        "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT",
        "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI",
        "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"
    ]):
        features["addr_state_CA"] = 1

    return features


def main() -> None:
    # Your EXTREMELY risky profile
    partial_features = {
        "loan_amnt": 950000,
        "int_rate": 16.5,
        "installment": 20000.0,  # very high payment
        "annual_inc": 20000,  # extremely low income
        "dti": 25.0,  # high DTI
        "revol_bal": 80000,  # high revolving debt
        "revol_util": 95.0,  # maxed out
        "total_acc": 40,
        "open_acc": 25,
        "pub_rec": 2,  # public records
        "delinq_2yrs": 3,  # recent delinquencies
        "inq_last_6mths": 8,  # many inquiries
        "emp_length": 0.5,  # < 1 year employment
        "credit_history_years": 1.0,  # very short history
        "term_ 60 months": 1,  # longer term = riskier
        "grade_B": 0,
        "grade_C": 0,
        "grade_D": 0,
        "grade_E": 0,
        "grade_F": 0,
        "grade_G": 1,  # worst grade
        "home_ownership_RENT": 1,  # RENT is riskier than MORTGAGE
        "home_ownership_OWN": 0,
        "home_ownership_MORTGAGE": 0,
        "purpose_debt_consolidation": 1,
        "addr_state_CA": 1,
    }

    print("Building complete feature vector (134 features)...")
    complete_features = build_complete_risky_features(partial_features)

    print(f"\nCalling /predict with {len(complete_features)} features...")
    resp = requests.post(f"{API_BASE}/predict", json={"features": complete_features}, timeout=60)
    resp.raise_for_status()
    result = resp.json()

    print("\n=== Prediction Result ===")
    print(json.dumps(result, indent=2))

    if result["approved"]:
        print("\n⚠️  WARNING: This risky loan was APPROVED (unexpected!)")
        print("   The model may need adjustment or the feature vector needs more risky defaults.")
    else:
        print("\n✅ CORRECT: This risky loan was REJECTED (as expected)")


if __name__ == "__main__":
    main()

