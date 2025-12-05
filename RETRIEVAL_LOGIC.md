# Retrieval Logic - Complete Breakdown

## Overview

The retrieval logic implements **graph-based retrieval** using SPARQL queries over an RDF knowledge graph. It follows a 3-step process:

1. **Question Interpretation** → Extract dimension from user question
2. **SPARQL Query Execution** → Query the KG for cohort data
3. **Result Formatting** → Transform RDF results into JSON

---

## Step 1: Question Interpretation

**Location**: `api.py` lines 138-159

```python
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
```

### How it works:
- **Input**: User question (e.g., "What is the default rate by grade?")
- **Process**: Simple keyword matching (case-insensitive)
- **Output**: Dictionary with `type` and `dimension` keys

### Example:
```python
question = "What is the default rate by grade?"
interpretation = _interpret_question_simple(question)
# Returns: {"type": "cohort", "dimension": "grade"}
```

### Limitations:
-  Only supports exact keyword matches
-  Fails on synonyms ("rent vs mortgage" doesn't match "home ownership")
-  No support for multi-dimensional queries
-  No support for temporal queries

---

## Step 2: SPARQL Query Execution

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

## Step 3: Main Endpoint Handler

**Location**: `api.py` lines 208-254

```python
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
```

### Complete Flow:

```
User Question
    ↓
_interpret_question_simple() → {"type": "cohort", "dimension": "grade"}
    ↓
_run_cohort_query(kg, "grade") → [{"key": "A", "loanCount": 228592, ...}, ...]
    ↓
Format response → AskResponse with results and answer_text
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
│         Step 2: SPARQL Query Execution                      │
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
│         Step 3: Response Formatting                          │
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

The retrieval logic is **simple but functional**:

-  **Works**: Successfully retrieves cohort data from KG
-  **Standard**: Uses proper SPARQL/RDF
-  **Limited**: Only supports single-dimension queries with keyword matching

**Key Files:**
- `api.py` lines 138-254: Main retrieval logic
- `evaluate_system.py` lines 33-72: Ground truth retrieval
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

