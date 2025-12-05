"""
LLM layer using Ollama on top of the REST API + KG.

Assumptions:
- FastAPI app from `api.py` is running on http://localhost:8000
- Ollama is running locally on http://localhost:11434
- You have at least one model available, e.g. 'llama3' or 'mistral'

This script:
- Routes questions either to /ask (KG cohorts) or /predict (model)
- Uses Ollama to turn structured JSON responses into natural language answers
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import requests


API_BASE = "http://localhost:8000"
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "llama3"  # change if you prefer a different local model

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "loan_classifier_final.pkl"


@dataclass
class ChatTurn:
    role: str
    content: str


def call_ollama(messages: List[ChatTurn]) -> str:
    """
    Call the local Ollama API using the /api/generate endpoint.

    We flatten the list of chat messages into a single prompt, since /api/generate
    expects plain text rather than role-tagged turns.
    """
    # Simple flattening: system + user + assistant turns concatenated
    parts = []
    for m in messages:
        prefix = m.role.upper()
        parts.append(f"{prefix}: {m.content}")
    prompt = "\n\n".join(parts)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # /api/generate returns the full text in 'response' when stream=False
    return data.get("response", "").strip()


def _build_features_from_user() -> Dict[str, Any]:
    """
    Interactive helper for CLI use:
    - Ask the user for a few key inputs.
    - Build a feature dict matching the trained model's columns.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifacts not found at {MODEL_PATH}")

    artifacts = joblib.load(MODEL_PATH)
    feature_names = artifacts["features"]

    # Base: zeros for all features
    features: Dict[str, Any] = {name: 0.0 for name in feature_names}

    # Collect minimal inputs
    def ask_float(prompt: str, default: float) -> float:
        val = input(f"{prompt} [{default}]: ").strip()
        if not val:
            return default
        try:
            return float(val)
        except ValueError:
            print("  Could not parse number, using default.")
            return default

    def ask_str(prompt: str, default: str) -> str:
        val = input(f"{prompt} [{default}]: ").strip()
        return val or default

    loan_amnt = ask_float("Loan amount", 15000)
    term = ask_str("Term (36 or 60 months)", "36")
    int_rate = ask_float("Interest rate (%)", 13.5)
    annual_inc = ask_float("Annual income", 80000)
    dti = ask_float("Debt-to-income ratio (%)", 18.0)
    grade = ask_str("Grade (A-G)", "B").upper()
    home_own = ask_str("Home ownership (RENT/OWN/MORTGAGE/OTHER)", "MORTGAGE").upper()
    purpose = ask_str("Purpose (debt_consolidation, credit_card, etc.)", "debt_consolidation")
    state = ask_str("State code (e.g., CA, NY)", "CA").upper()

    # Fill key numeric features (similar to demo_predict defaults, but using user values)
    features.update(
        {
            "loan_amnt": loan_amnt,
            "int_rate": int_rate,
            "installment": round(loan_amnt * 0.035, 2),
            "annual_inc": annual_inc,
            "dti": dti,
            "revol_bal": 12000,
            "revol_util": 45.0,
            "total_acc": 24,
            "open_acc": 12,
            "pub_rec": 0,
            "delinq_2yrs": 0,
            "inq_last_6mths": 1,
            "emp_length": 6,
            "credit_history_years": 12.0,
            "income_to_debt_ratio": max(annual_inc / (loan_amnt + 1e-6), 0.0),
            "available_credit_ratio": 0.65,
            "payment_to_income_ratio": (loan_amnt / max(annual_inc, 1.0)) * 0.4,
            "loan_to_income_ratio": loan_amnt / max(annual_inc, 1.0),
            "revol_util_squared": (0.45) ** 2,
            "dti_squared": (dti / 100.0) ** 2,
            "inquiry_ratio": 0.05,
            "open_acc_ratio": 0.55,
            "has_delinq": 0,
        }
    )

    # Term dummy: only 60 months dummy in features
    if term.strip().startswith("60"):
        features["term_ 60 months"] = 1.0
    else:
        features["term_ 60 months"] = 0.0

    # Grade dummy
    for g in ["B", "C", "D", "E", "F", "G"]:
        key = f"grade_{g}"
        if key in features:
            features[key] = 1.0 if grade == g else 0.0

    # Subgrade: map roughly to middle bucket for the grade
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
    for opt in ["MORTGAGE", "NONE", "OTHER", "OWN", "RENT"]:
        key = f"home_ownership_{opt}"
        if key in features:
            features[key] = 1.0 if home_own == opt else 0.0

    # Verification status (default Verified)
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
    for k in list(features.keys()):
        if k.startswith("addr_state_"):
            features[k] = 0.0
    state_key = f"addr_state_{state}"
    if state_key in features:
        features[state_key] = 1.0

    return features


def route_question_with_llm(user_question: str) -> str:
    """
    Ask the LLM to decide which backend to use:
    - 'cohort'  -> /ask
    - 'predict' -> /predict

    Returns a simple string label.
    """
    system = ChatTurn(
        role="system",
        content=(
            "You are a router for a loan analytics assistant. "
            "Decide whether the question is: "
            "'cohort' (analytics over many historical loans) or "
            "'predict' (run the ML model for a specific applicant). "
            "Respond with ONLY one word: cohort or predict."
        ),
    )
    user = ChatTurn(role="user", content=user_question)
    raw = call_ollama([system, user]).strip().lower()
    if "predict" in raw:
        return "predict"
    return "cohort"


def summarize_cohort_answer(question: str, api_result: Dict[str, Any]) -> str:
    """
    Use the LLM to turn /ask JSON into a conversational answer.
    """
    context = json.dumps(api_result, indent=2)
    system = ChatTurn(
        role="system",
        content=(
            "You are a precise loan portfolio analyst. "
            "Use ONLY the structured context to answer the question. "
            "Cite key numbers (default rates, loan counts, averages). "
            "If the question cannot be answered from the context, say so clearly."
        ),
    )
    user = ChatTurn(
        role="user",
        content=(
            f"Question:\n{question}\n\n"
            f"Structured context (JSON):\n{context}\n\n"
            "Now answer the question concisely (3-6 sentences)."
        ),
    )
    return call_ollama([system, user])


def summarize_predict_answer(features: Dict[str, Any], api_result: Dict[str, Any]) -> str:
    """
    Use the LLM to explain a prediction from /predict.
    """
    context = json.dumps({"features": features, "prediction": api_result}, indent=2)
    system = ChatTurn(
        role="system",
        content=(
            "You are a risk officer explaining a loan default prediction. "
            "Explain the default probability, threshold, and approval decision "
            "in clear language. Mention limitations and that this is a model-based estimate."
        ),
    )
    user = ChatTurn(
        role="user",
        content=(
            "Here is the input feature set and the model prediction as JSON:\n"
            f"{context}\n\n"
            "Explain the result in 3-6 sentences for a non-technical audience."
        ),
    )
    return call_ollama([system, user])


def call_api_ask(question: str) -> Dict[str, Any]:
    resp = requests.post(f"{API_BASE}/ask", json={"question": question}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def call_api_predict(features: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{API_BASE}/predict", json={"features": features}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def chat_once(question: str, features_for_predict: Dict[str, Any] | None = None) -> str:
    """
    High-level entrypoint:
    - use LLM to choose route
    - call backend
    - use LLM to verbalize answer
    """
    route = route_question_with_llm(question)
    if route == "predict":
        # If no features provided programmatically, collect them interactively
        if not features_for_predict:
            print(
                "This looks like a prediction question. "
                "I'll ask a few follow-up questions to build the input features."
            )
            features_for_predict = _build_features_from_user()

        api_res = call_api_predict(features_for_predict)
        return summarize_predict_answer(features_for_predict, api_res)
    else:
        api_res = call_api_ask(question)
        return summarize_cohort_answer(question, api_res)


if __name__ == "__main__":
    print("Loan Chatbot (Ollama + REST API). Ctrl-C to exit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q:
            continue
        try:
            answer = chat_once(q)
        except Exception as exc:
            print(f"[error] {exc}")
            continue
        print(f"\nAssistant:\n{answer}\n")


