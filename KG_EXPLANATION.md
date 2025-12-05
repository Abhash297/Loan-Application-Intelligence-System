# Knowledge Graph (KG) Implementation - Detailed Explanation

## Overview

The Knowledge Graph (KG) component is a **bonus feature** that implements graph-based retrieval for the loan chatbot system. It transforms loan application data into a structured RDF (Resource Description Framework) graph using cohort-based aggregation.

---

## 1. What is the Knowledge Graph?

The KG is a structured representation of loan data organized by **cohorts** (aggregated groups). Instead of storing individual loan records, it stores statistical summaries grouped by various dimensions:

- **Grade** (A-G): Loan risk grades
- **Term** (36/60 months): Loan duration
- **Purpose** (debt_consolidation, credit_card, etc.): Loan purpose
- **Loan Status** (Fully Paid, Charged Off, etc.): Current status
- **Home Ownership** (RENT, OWN, MORTGAGE, etc.): Borrower's housing situation
- **Income Band** (<50k, 50k-75k, 75k-100k, 100k-150k, >150k): Income ranges
- **State** (CA, NY, TX, etc.): Borrower's state

Each cohort node contains:
- `loanCount`: Number of loans in this cohort
- `defaultRate`: Percentage of loans that defaulted
- `avgInterestRate`: Average interest rate
- `avgLoanAmount`: Average loan amount
- `avgDTI`: Average debt-to-income ratio
- `avgIncome`: Average annual income

---

## 2. How is the KG Built?

### Architecture (`kg_cohorts.py`)

The KG construction follows these steps:

#### Step 1: Data Loading
```python
df = load_cleaned_data("cleaned_loan_dataset.csv")
```
- Loads the cleaned loan dataset
- Validates required columns are present

#### Step 2: Cohort Computation
```python
cohorts = compute_cohorts(df)
```
- Groups loans by each dimension (grade, term, purpose, etc.)
- Computes aggregate statistics for each group:
  - Total loan count
  - Default count and rate
  - Average interest rate, loan amount, DTI, income

#### Step 3: RDF Graph Construction
```python
g = build_cohort_graph(cohorts)
```
- Creates an RDF graph using `rdflib`
- Each cohort becomes a node with URI: `ex:Cohort_{type}_{value}`
  - Example: `ex:Cohort_grade_A`, `ex:Cohort_state_CA`
- Properties are stored as RDF triples:
  ```
  ex:Cohort_grade_A ex:defaultRate 0.070 .
  ex:Cohort_grade_A ex:loanCount 228592 .
  ```

#### Step 4: Serialization
```python
g.serialize(destination="loan_cohorts.ttl", format="turtle")
```
- Saves the graph to a Turtle (.ttl) file
- Turtle is a human-readable RDF format

### Example Output

From `loan_cohorts.ttl`:
```turtle
ex:Cohort_grade_A a ex:Cohort ;
    ex:avgDTI 15.61 ;
    ex:avgIncome 88882.97 ;
    ex:avgInterestRate 7.13 ;
    ex:avgLoanAmount 13892.66 ;
    ex:cohortType "grade" ;
    ex:defaultRate 0.070 ;
    ex:grade "A" ;
    ex:loanCount 228592 .
```

This represents: Grade A loans have a 7.0% default rate, average interest rate of 7.13%, etc.

---

## 3. How is the KG Used in the System?

### Integration Points

#### A. REST API (`api.py`)

The `/ask` endpoint uses the KG for cohort-based queries:

1. **Question Interpretation** (`_interpret_question_simple`):
   - Simple keyword matching to identify dimension (grade, term, purpose, etc.)
   - Maps user questions to cohort types

2. **SPARQL Query Execution** (`_run_cohort_query`):
   - Constructs SPARQL queries dynamically based on dimension
   - Example query for grade dimension:
     ```sparql
     PREFIX ex: <http://example.org/loan#>
     SELECT ?cohort ?key ?loanCount ?defaultRate ?avgInterestRate
     WHERE {
       ?cohort a ex:Cohort ;
               ex:cohortType "grade" ;
               ex:grade ?key ;
               ex:loanCount ?loanCount ;
               ex:defaultRate ?defaultRate .
     }
     ```

3. **Response Generation**:
   - Returns structured JSON with cohort statistics
   - Provides a simple text summary

#### B. Chatbot (`chat_ollama.py`)

The chatbot uses the KG indirectly through the API:

1. **Routing**: LLM decides if question is "cohort" (KG) or "predict" (ML model)
2. **API Call**: Calls `/ask` endpoint which queries the KG
3. **Answer Summarization**: LLM converts structured KG results into natural language

#### C. Evaluation System (`evaluate_system.py`)

- Uses KG as **ground truth** for cohort questions
- Queries KG directly to verify system accuracy
- Compares API responses against KG data

---

## 4. Does it Satisfy Project Requirements?

### Bonus Requirements Analysis

#### ✅ Requirement 1: "Build a knowledge graph from loan application data"
**Status: FULLY SATISFIED**

- ✅ KG is built from loan CSV data
- ✅ Uses proper RDF/OWL standards (rdflib, Turtle format)
- ✅ Contains meaningful domain entities (cohorts)
- ✅ Includes statistical properties (default rates, averages)

**Evidence:**
- `kg_cohorts.py` implements full KG construction pipeline
- `loan_cohorts.ttl` contains 100+ cohort nodes with properties
- Graph has 937 triples (from test output)

#### ✅ Requirement 2: "Implement graph-based retrieval for the chatbot"
**Status: PARTIALLY SATISFIED**

- ✅ SPARQL queries implemented (`_run_cohort_query`)
- ✅ Integrated into `/ask` API endpoint
- ✅ Used by chatbot for cohort questions
- ⚠️ **Limitation**: Simple keyword-based routing, not full semantic retrieval

**Evidence:**
- `api.py` lines 162-205: SPARQL query execution
- `api.py` lines 208-254: `/ask` endpoint uses KG
- `chat_ollama.py` routes cohort questions to KG-backed API

**What's Missing:**
- No semantic similarity search (only exact dimension matching)
- No graph traversal (only single-dimension queries)
- No relationship inference between cohorts

#### ⚠️ Requirement 3: "Compare performance against base implementation"
**Status: NOT FULLY ADDRESSED**

- ❌ No explicit comparison with non-KG baseline
- ❌ No performance metrics comparing KG vs. direct database queries
- ✅ Evaluation shows KG retrieval works (66.67% retrieval quality)

**What's Needed:**
- Baseline implementation without KG (direct pandas queries)
- Side-by-side comparison of:
  - Query performance (speed)
  - Answer accuracy
  - Query expressiveness
- Performance report showing KG advantages/disadvantages

---

## 5. Current Limitations

### A. Simple Routing
- **Issue**: `_interpret_question_simple` uses basic keyword matching
- **Impact**: Fails on complex phrasings (e.g., "rent versus mortgage" doesn't match "home ownership")
- **Evidence**: Routing accuracy only 60% (9/15 questions)

### B. Limited Query Types
- **Issue**: Only supports single-dimension cohort queries
- **Missing**:
  - Multi-dimensional queries (e.g., "Grade C loans with 36-month term")
  - Temporal queries (e.g., "loans issued in 2015")
  - Comparative queries (e.g., "compare grade A vs grade B")
  - Aggregations across cohorts

### C. No Graph Relationships
- **Issue**: Cohorts are isolated nodes, no edges between them
- **Missing**:
  - Relationships like "grade A loans have lower default rate than grade B"
  - Hierarchical relationships (e.g., state → region)
  - Causal relationships

### D. No Performance Comparison
- **Issue**: No baseline implementation to compare against
- **Missing**: Metrics showing KG advantages (if any)

---

## 6. Strengths

### ✅ Proper RDF Implementation
- Uses standard RDF/OWL technologies
- Turtle serialization is human-readable
- SPARQL queries are standard-compliant

### ✅ Domain-Appropriate Design
- Cohort-based aggregation is sensible for loan analytics
- Covers key business dimensions (grade, term, purpose, etc.)
- Statistical properties are relevant (default rates, averages)

### ✅ Integration with System
- Seamlessly integrated into REST API
- Used by chatbot for analytics questions
- Serves as ground truth for evaluation

### ✅ Functional for Simple Queries
- Works perfectly for direct questions (e.g., "What is the default rate by grade?")
- 100% accuracy when correctly routed
- Fast query execution

---

## 7. Recommendations for Improvement

### High Priority

1. **Replace Simple Router with LLM-Based Classification**
   - Current: Keyword matching
   - Proposed: Use Ollama to classify question type and extract dimensions
   - Expected: Routing accuracy 90%+ (vs. current 60%)

2. **Add Multi-Dimensional Queries**
   - Support queries like "Grade C loans with 36-month term"
   - Requires SPARQL JOINs across cohort types

3. **Implement Performance Comparison**
   - Create baseline (direct pandas queries)
   - Measure: query speed, accuracy, expressiveness
   - Document findings

### Medium Priority

4. **Add Graph Relationships**
   - Create edges between related cohorts
   - Enable graph traversal queries
   - Example: "Find all cohorts related to grade A"

5. **Support Temporal Queries**
   - Add year-based cohorts
   - Or explicitly handle unsupported queries with clear messages

6. **Enhance SPARQL Query Builder**
   - Support comparative queries ("compare X vs Y")
   - Support aggregation queries ("average default rate across all grades")

---

## 8. Summary

### What Works Well ✅
- Proper RDF/SPARQL implementation
- Functional for simple cohort queries
- Good integration with chatbot system
- Domain-appropriate design

### What Needs Improvement ⚠️
- Simple routing (60% accuracy)
- Limited query expressiveness
- No performance comparison with baseline
- No graph relationships

### Overall Assessment

**The KG implementation satisfies the core bonus requirements** (build KG, implement retrieval) but has room for improvement in:
- Query sophistication
- Routing accuracy
- Performance benchmarking

**Grade: B+** (Good implementation, but missing comparison component and advanced features)

---

## 9. Code References

- **KG Construction**: `kg_cohorts.py` (290 lines)
- **KG Usage**: `api.py` lines 78-88 (loading), 162-205 (queries), 208-254 (endpoint)
- **KG Testing**: `test_sparql.py` (184 lines)
- **KG Serialization**: `loan_cohorts.ttl` (937 triples)
- **Chatbot Integration**: `chat_ollama.py` lines 204-226 (routing), 229-251 (summarization)
- **Evaluation**: `evaluate_system.py` lines 33-72 (ground truth queries)

