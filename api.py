"""
FastAPI application exposing:
- /predict: model-backed loan default prediction
- /ask: KG-backed analytics over cohort graph (simple, no external LLM here)

We can plug in Ollama for natural language handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rdflib import Graph

from kg_cohorts import build_cohort_graph, compute_cohorts, load_cleaned_data


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_classifier_final.pkl"
DATA_PATH = BASE_DIR / "cleaned_loan_dataset.csv"
KG_TTL_PATH = BASE_DIR / "loan_cohorts.ttl"


class PredictRequest(BaseModel):
    """Input features for prediction. Accepts a partial dict of features."""

    features: Dict[str, Any] = Field(
        ...,
        description="Dictionary of feature_name -> value, using the same names as in training.",
    )


class PredictResponse(BaseModel):
    approved: bool
    default_probability: float
    threshold: float
    model_metadata: Dict[str, Any]


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    interpretation: Dict[str, Any]
    results: List[Dict[str, Any]]
    answer_text: str


app = FastAPI(title="Loan Intelligence API", version="1.0.0")


@app.on_event("startup")
def load_artifacts() -> None:
    """
    Load model and KG at startup.
    """
    # Model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")

    model_artifacts = joblib.load(MODEL_PATH)
    if "model" not in model_artifacts or "features" not in model_artifacts:
        raise RuntimeError("Model artifacts missing required keys.")

    app.state.model = model_artifacts["model"]
    app.state.threshold = float(model_artifacts.get("threshold", 0.5))
    app.state.model_features = list(model_artifacts["features"])
    app.state.model_metadata = dict(model_artifacts.get("metadata", {}))

    # KG: if TTL exists, load; otherwise, build from data
    g = Graph()
    if KG_TTL_PATH.exists():
        g.parse(str(KG_TTL_PATH), format="turtle")
    else:
        df = load_cleaned_data(DATA_PATH)
        cohorts = compute_cohorts(df)
        g = build_cohort_graph(cohorts)
        g.serialize(destination=str(KG_TTL_PATH), format="turtle")

    app.state.kg_graph = g


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Run the trained classifier on the provided feature dict.
    
    Note: For realistic predictions, provide all 134 features the model expects.
    Missing features are filled with NaN, which may lead to unreliable predictions.
    """
    model = app.state.model
    threshold = app.state.threshold
    feature_names: List[str] = app.state.model_features

    # Build a single-row DataFrame matching training columns
    row = {}
    missing_count = 0
    for f in feature_names:
        val = req.features.get(f, np.nan)
        row[f] = val
        if pd.isna(val):
            missing_count += 1

    X = pd.DataFrame([row], columns=feature_names)

    # Warn if too many features are missing (but still proceed)
    missing_pct = missing_count / len(feature_names)
    if missing_pct > 0.5:
        import warnings
        warnings.warn(
            f"Warning: {missing_count}/{len(feature_names)} features are missing. "
            f"Prediction may be unreliable. Provide full feature vector for realistic results."
        )

    try:
        proba = model.predict_proba(X)[:, 1][0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    approved = bool(proba < threshold)

    return PredictResponse(
        approved=approved,
        default_probability=float(proba),
        threshold=threshold,
        model_metadata=app.state.model_metadata,
    )


def _interpret_question_simple(question: str) -> Dict[str, Any]:
    """
    Very simple heuristic interpreter that maps a few patterns to
    cohort queries. In a full system, replace this with an Ollama-based
    LLM classifier + SPARQL generator.
    """
    q_lower = question.lower()

    if "grade" in q_lower:
        return {"type": "cohort", "dimension": "grade"}
    if "term" in q_lower:
        return {"type": "cohort", "dimension": "term"}
    if "purpose" in q_lower:
        return {"type": "cohort", "dimension": "purpose"}
    if "home ownership" in q_lower or "homeownership" in q_lower:
        return {"type": "cohort", "dimension": "home_ownership"}
    if "income" in q_lower:
        return {"type": "cohort", "dimension": "income_band"}
    if "state" in q_lower:
        return {"type": "cohort", "dimension": "state"}

    return {"type": "unknown"}


def _run_cohort_query(kg: Graph, dimension: str) -> List[Dict[str, Any]]:
    """
    Run a simple SPARQL query to fetch all cohorts of a given type.
    """
    prefix = "http://example.org/loan#"

    query = f"""
    PREFIX ex: <{prefix}>
    SELECT ?cohort ?key ?loanCount ?defaultRate ?avgInterestRate ?avgLoanAmount
    WHERE {{
      ?cohort a ex:Cohort ;
              ex:cohortType "{dimension}" ;
              ex:loanCount ?loanCount ;
              ex:avgInterestRate ?avgInterestRate ;
              ex:avgLoanAmount ?avgLoanAmount .

      OPTIONAL {{ ?cohort ex:defaultRate ?defaultRate . }}

      OPTIONAL {{
        {'?cohort ex:grade ?key .' if dimension == 'grade' else ''}
        {'?cohort ex:term ?key .' if dimension == 'term' else ''}
        {'?cohort ex:purpose ?key .' if dimension == 'purpose' else ''}
        {'?cohort ex:loanStatus ?key .' if dimension == 'status' else ''}
        {'?cohort ex:homeOwnership ?key .' if dimension == 'home_ownership' else ''}
        {'?cohort ex:incomeBand ?key .' if dimension == 'income_band' else ''}
        {'?cohort ex:state ?key .' if dimension == 'state' else ''}
      }}
    }}
    ORDER BY ?key
    """

    results: List[Dict[str, Any]] = []
    for row in kg.query(query):
        _, key, loan_count, default_rate, avg_ir, avg_amt = row
        results.append(
            {
                "key": str(key) if key is not None else None,
                "loanCount": int(loan_count),
                "defaultRate": float(default_rate) if default_rate is not None else None,
                "avgInterestRate": float(avg_ir),
                "avgLoanAmount": float(avg_amt),
            }
        )
    return results


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """
    Simple KG-backed question handler.

    For now, we:
    - Heuristically interpret the question to choose a cohort dimension.
    - Run a SPARQL query over the cohort KG.
    - Return raw stats plus a brief textual summary.
    """
    interpretation = _interpret_question_simple(req.question)
    kg: Graph = app.state.kg_graph

    if interpretation.get("type") != "cohort":
        return AskResponse(
            question=req.question,
            interpretation=interpretation,
            results=[],
            answer_text=(
                "I can currently answer questions about aggregate statistics by "
                "grade, term, purpose, home ownership, income band, or state."
            ),
        )

    dim = interpretation["dimension"]
    results = _run_cohort_query(kg, dim)

    if not results:
        answer = f"No cohort statistics found for dimension '{dim}'."
    else:
        # Very simple summary: show top few rows
        top = results[:5]
        dim_label = dim.replace("_", " ")
        answer_lines = [
            f"Found {len(results)} {dim_label} cohorts. Showing first {len(top)}:",
        ]
        for r in top:
            frag = f"{r['key']}: defaultRate={r.get('defaultRate')}, loanCount={r['loanCount']}"
            answer_lines.append(f" - {frag}")
        answer = "\n".join(answer_lines)

    return AskResponse(
        question=req.question,
        interpretation=interpretation,
        results=results,
        answer_text=answer,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


