# Retrieval Logic - Complete Breakdown

## Overview

The system implements a **dual-backend routing system** that directs questions to either:
- **Knowledge Graph (KG)**: For analytical queries over aggregated loan statistics
- **ML Model**: For individual loan default predictions

The routing logic follows a 2-step process:

1. **Question Interpretation** → Determine question type (KG query vs ML prediction)
2. **Backend Execution** → Route to appropriate backend (KG SPARQL query or ML model inference)
3. **Result Formatting** → Transform results into JSON response

---

## Step 1: Question Interpretation and Routing

**Location**: `api.py` lines 138-177

```python
def _interpret_question_simple(question: str) -> Dict[str, Any]:
    """
    Very simple heuristic interpreter that maps a few patterns to
    KG queries or ML model predictions. In a full system, replace this with an Ollama-based
    LLM classifier + SPARQL generator.
    """
    q_lower = question.lower()

    # Check for prediction keywords first (before KG keywords)
    # This ensures prediction questions are routed correctly even if they mention KG dimensions
    prediction_keywords = [
        "approve",
        "predict",
        "predicted",
        "default probability",
        "probability",
        "should we",
        "will this",
        "risk of default",
        "default risk",
    ]
    for keyword in prediction_keywords:
        if keyword in q_lower:
            return {"type": "predict"}

    # Check for KG dimension keywords
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
```

### How it works:
- **Input**: User question (e.g., "What is the default rate by grade?" or "Should we approve this loan?")
- **Process**: 
  1. First checks for prediction keywords (approve, predict, probability, etc.)
  2. If no prediction keywords, checks for KG dimension keywords
  3. Returns routing decision
- **Output**: Dictionary with `type` ("predict", "cohort", or "unknown") and optionally `dimension` for KG queries

### Examples:

**KG Query:**
```python
question = "What is the default rate by grade?"
interpretation = _interpret_question_simple(question)
# Returns: {"type": "cohort", "dimension": "grade"}
```

**ML Model Prediction:**
```python
question = "Should we approve a $15,000 loan for a borrower with grade B?"
interpretation = _interpret_question_simple(question)
# Returns: {"type": "predict"}
```

**Unknown:**
```python
question = "What is the current employer of a borrower?"
interpretation = _interpret_question_simple(question)
# Returns: {"type": "unknown"}
```

### Limitations:
-  Only supports exact keyword matches
-  Fails on synonyms ("rent vs mortgage" doesn't match "home ownership")
-  No support for multi-dimensional queries
-  No support for temporal queries
-  Prediction keywords checked first, but may miss some phrasings

---

## Step 2A: KG Backend - SPARQL Query Execution

**Location**: `api.py` lines 162-205

```python
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
```

### How it works:

1. **Build SPARQL Query**:
   - Uses the dimension (e.g., "grade") to filter cohorts
   - Dynamically selects the appropriate property based on dimension
   - Retrieves: key, loanCount, defaultRate, avgInterestRate, avgLoanAmount

2. **Execute Query**:
   - `kg.query(query)` executes SPARQL against the RDF graph
   - Returns rows of results

3. **Transform Results**:
   - Converts RDF literals to Python types
   - Structures as list of dictionaries

### Example SPARQL Query (for grade dimension):

```sparql
PREFIX ex: <http://example.org/loan#>
SELECT ?cohort ?key ?loanCount ?defaultRate ?avgInterestRate ?avgLoanAmount
WHERE {
  ?cohort a ex:Cohort ;
          ex:cohortType "grade" ;
          ex:loanCount ?loanCount ;
          ex:avgInterestRate ?avgInterestRate ;
          ex:avgLoanAmount ?avgLoanAmount .
  OPTIONAL { ?cohort ex:defaultRate ?defaultRate . }
  OPTIONAL { ?cohort ex:grade ?key . }
}
ORDER BY ?key
```

### Example Result:

```python
[
    {
        "key": "A",
        "loanCount": 228592,
        "defaultRate": 0.070,
        "avgInterestRate": 7.13,
        "avgLoanAmount": 13892.66
    },
    {
        "key": "B",
        "loanCount": 388102,
        "defaultRate": 0.151,
        "avgInterestRate": 10.69,
        "avgLoanAmount": 13273.57
    },
    # ... more grades
]
```

---

## Step 2B: ML Model Backend - Prediction Execution

**Location**: `api.py` lines 91-135

```python
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
```

### How it works:

1. **Feature Vector Construction**:
   - Takes partial feature dictionary from request
   - Fills missing features with NaN
   - Creates DataFrame matching training column order

2. **Model Inference**:
   - Runs `model.predict_proba()` to get default probability
   - Compares probability to threshold (0.67) to determine approval

3. **Response**:
   - Returns approval decision, default probability, threshold, and model metadata

### Example Request:
```json
{
  "features": {
    "loan_amnt": 15000,
    "int_rate": 13.5,
    "annual_inc": 80000,
    "dti": 18.0,
    "grade_B": 1.0
  }
}
```

### Example Response:
```json
{
  "approved": true,
  "default_probability": 0.23,
  "threshold": 0.67,
  "model_metadata": {...}
}
```

---

## Step 3: Main Endpoint Handler

**Location**: `api.py` lines 226-283

```python
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """
    Simple question handler that routes to KG or ML model.
    """
    interpretation = _interpret_question_simple(req.question)
    kg: Graph = app.state.kg_graph

    # Route to ML model if prediction question detected
    if interpretation.get("type") == "predict":
        return AskResponse(
            question=req.question,
            interpretation=interpretation,
            results=[],
            answer_text=(
                "This question requires a prediction from the ML model. "
                "Please use the /predict endpoint with the required feature vector."
            ),
        )

    # Route to KG if cohort question detected
    if interpretation.get("type") == "cohort":
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

    # Unknown question type
    return AskResponse(
        question=req.question,
        interpretation=interpretation,
        results=[],
        answer_text=(
            "I can currently answer questions about aggregate statistics by "
            "grade, term, purpose, home ownership, income band, or state."
        ),
    )
```

### Complete Flow:

**For KG Queries:**
```
User Question: "What is the default rate by grade?"
    ↓
_interpret_question_simple() → {"type": "cohort", "dimension": "grade"}
    ↓
_run_cohort_query(kg, "grade") → [{"key": "A", "loanCount": 228592, ...}, ...]
    ↓
Format response → AskResponse with results and answer_text
```

**For ML Model Predictions:**
```
User Question: "Should we approve a $15,000 loan for a borrower with grade B?"
    ↓
_interpret_question_simple() → {"type": "predict"}
    ↓
Route to /predict endpoint → Requires feature vector
    ↓
ML model inference → PredictResponse with approval and probability
```

---

## Alternative: Direct KG Query (for Evaluation)

**Location**: `evaluate_system.py` lines 33-72

This is used for evaluation to get ground truth directly from KG:

```python
def get_ground_truth_from_kg(dimension: str) -> Dict[str, Any]:
    """Query KG directly to get ground truth for cohort questions."""
    g = Graph()
    g.parse(str(KG_TTL_PATH), format="turtle")
    prefix = "http://example.org/loan#"

    query = f"""
    PREFIX ex: <{prefix}>
    SELECT ?key ?loanCount ?defaultRate ?avgInterestRate ?avgLoanAmount
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

    results = {}
    for row in g.query(query):
        key, loan_count, default_rate, avg_ir, avg_amt = row
        key_str = str(key) if key else "unknown"
        results[key_str] = {
            "loanCount": int(loan_count),
            "defaultRate": float(default_rate) if default_rate is not None else None,
            "avgInterestRate": float(avg_ir),
            "avgLoanAmount": float(avg_amt),
        }
    return results
```

**Difference**: Returns a dictionary keyed by cohort value (e.g., `{"A": {...}, "B": {...}}`) instead of a list.

---

## Complete Retrieval Flow Diagram

### KG Query Flow:
```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUESTION                            │
│  "What is the default rate by grade?"                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Step 1: Question Interpretation                     │
│         _interpret_question_simple()                        │
│                                                              │
│  Input:  "What is the default rate by grade?"              │
│  Output: {"type": "cohort", "dimension": "grade"}          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Step 2A: SPARQL Query Execution                     │
│         _run_cohort_query(kg, "grade")                      │
│                                                              │
│  Query:  SELECT ?key ?loanCount ?defaultRate ...           │
│          WHERE { ?cohort ex:cohortType "grade" ... }        │
│                                                              │
│  Results: [                                                 │
│    {"key": "A", "loanCount": 228592, "defaultRate": 0.070},│
│    {"key": "B", "loanCount": 388102, "defaultRate": 0.151},│
│    ...                                                      │
│  ]                                                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Step 3: Response Formatting                         │
│         ask() endpoint                                       │
│                                                              │
│  Returns: AskResponse {                                      │
│    question: "What is the default rate by grade?",          │
│    interpretation: {"type": "cohort", "dimension": "grade"},│
│    results: [...],                                           │
│    answer_text: "Found 7 grade cohorts. Showing first 5:..."│
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

### ML Model Prediction Flow:
```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUESTION                            │
│  "Should we approve a $15,000 loan for grade B?"            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Step 1: Question Interpretation                     │
│         _interpret_question_simple()                        │
│                                                              │
│  Input:  "Should we approve a $15,000 loan..."             │
│  Output: {"type": "predict"}                                │
│  (Keyword "approve" detected)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Step 2B: ML Model Inference                         │
│         /predict endpoint                                    │
│                                                              │
│  Input:  Feature vector {loan_amnt: 15000, grade_B: 1, ...}│
│  Process: XGBoost model.predict_proba()                     │
│  Output: default_probability = 0.23                        │
│          approved = true (0.23 < 0.67 threshold)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Step 3: Response Formatting                         │
│         PredictResponse                                      │
│                                                              │
│  Returns: {                                                  │
│    approved: true,                                           │
│    default_probability: 0.23,                               │
│    threshold: 0.67,                                         │
│    model_metadata: {...}                                     │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

###  What's Good:

1. **Standard SPARQL**: Uses industry-standard RDF/SPARQL
2. **Separation of Concerns**: Interpretation, querying, and formatting are separate
3. **Type Safety**: Returns structured dictionaries with consistent keys
4. **Error Handling**: Returns empty results gracefully

###  Limitations:

1. **Simple Keyword Matching**: Only exact keyword matches work
2. **Single Dimension**: Can't query multiple dimensions at once
3. **No Filtering**: Can't filter by specific values (e.g., "grade A only")
4. **No Aggregation**: Can't compute averages across dimensions
5. **No Relationships**: Can't traverse relationships between cohorts

---

## Example: Complete Retrieval Trace

### Input:
```json
{
  "question": "What is the default rate by grade?"
}
```

### Step 1 Output:
```python
{"type": "cohort", "dimension": "grade"}
```

### Step 2 SPARQL Query:
```sparql
PREFIX ex: <http://example.org/loan#>
SELECT ?cohort ?key ?loanCount ?defaultRate ?avgInterestRate ?avgLoanAmount
WHERE {
  ?cohort a ex:Cohort ;
          ex:cohortType "grade" ;
          ex:grade ?key ;
          ex:loanCount ?loanCount ;
          ex:defaultRate ?defaultRate ;
          ex:avgInterestRate ?avgInterestRate ;
          ex:avgLoanAmount ?avgLoanAmount .
}
ORDER BY ?key
```

### Step 2 Results:
```python
[
    {"key": "A", "loanCount": 228592, "defaultRate": 0.070, "avgInterestRate": 7.13, "avgLoanAmount": 13892.66},
    {"key": "B", "loanCount": 388102, "defaultRate": 0.151, "avgInterestRate": 10.69, "avgLoanAmount": 13273.57},
    {"key": "C", "loanCount": 382727, "defaultRate": 0.250, "avgInterestRate": 14.03, "avgLoanAmount": 14253.87},
    # ... more grades
]
```

### Step 3 Final Response:
```json
{
  "question": "What is the default rate by grade?",
  "interpretation": {"type": "cohort", "dimension": "grade"},
  "results": [
    {"key": "A", "loanCount": 228592, "defaultRate": 0.070, ...},
    {"key": "B", "loanCount": 388102, "defaultRate": 0.151, ...},
    ...
  ],
  "answer_text": "Found 7 grade cohorts. Showing first 5:\n - A: defaultRate=0.070, loanCount=228592\n - B: defaultRate=0.151, loanCount=388102\n ..."
}
```

---

## Summary

The retrieval logic implements a **dual-backend routing system**:

### KG Backend:
-  **Works**: Successfully retrieves cohort data from KG using SPARQL
-  **Standard**: Uses proper SPARQL/RDF
-  **Limited**: Only supports single-dimension queries with keyword matching

### ML Model Backend:
-  **Works**: Routes prediction questions to XGBoost model
-  **Requires**: Full feature vector (134 features) for accurate predictions
-  **Returns**: Default probability and approval decision

### Routing:
-  **Keyword-based**: Simple heuristic matching
-  **Priority**: Prediction keywords checked first, then KG dimensions
-  **Accuracy**: 73.3% (11/15 questions) - see EVALUATION_SUMMARY.md

**Key Files:**
- `api.py` lines 91-135: ML model prediction endpoint
- `api.py` lines 138-177: Question interpretation and routing
- `api.py` lines 180-223: KG SPARQL query execution
- `api.py` lines 226-283: Main /ask endpoint handler
- `evaluate_system.py` lines 33-72: Ground truth retrieval for evaluation
- `kg_cohorts.py`: KG construction (not retrieval, but prerequisite)

---

## KG Structure: Entities and Relationships

### Entities

Each entity is a **Cohort node** representing aggregated loan statistics:

```
ex:Cohort_grade_A
ex:Cohort_grade_B
ex:Cohort_state_CA
ex:Cohort_purpose_debt_consolidation
ex:Cohort_term_36_months
```

### Entity Properties (Not Relationships)

Each cohort has **properties** (not relationships to other entities):

```
ex:Cohort_grade_A
    ex:grade "A"
    ex:cohortType "grade"
    ex:loanCount 228592
    ex:defaultRate 0.070
    ex:avgInterestRate 7.13
    ex:avgLoanAmount 13892.66
    ex:avgDTI 15.61
    ex:avgIncome 88882.97
```

### Relationships

**There are NO relationships between entities.** Each cohort is isolated.

The KG is a **flat structure** - just a collection of cohort nodes with properties. No edges connect them.

### Example Structure

```
[ex:Cohort_grade_A] --(no edges)--> [ex:Cohort_grade_B]
     |                                    |
  (properties)                        (properties)
     |                                    |
  grade: "A"                         grade: "B"
  defaultRate: 0.070                 defaultRate: 0.151
  loanCount: 228592                  loanCount: 388102
```

**In RDF terms:**
- **Subject**: `ex:Cohort_grade_A` (the entity)
- **Predicate**: `ex:defaultRate` (the property name)
- **Object**: `0.070` (the property value)

This is a **property graph** without relationships, not a connected graph.

