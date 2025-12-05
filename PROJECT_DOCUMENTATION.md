# Loan Application Intelligence System - Project Documentation

## Overview

This project implements an intelligent loan application system that combines classical machine learning for default prediction with a knowledge graph-based chatbot for loan analytics. The system provides both predictive capabilities (should we approve this loan?) and analytical capabilities (what are the default rates by grade?).

---

## Part 1: Machine Learning Classification

### Data Preprocessing

The loan dataset (Lending Club data, ~2.26M rows, 145 columns) underwent extensive preprocessing to prepare it for model training:

#### Data Loading and Type Conversion
- Initial load with string dtype to preserve all values
- Conversion of numeric columns using `pd.to_numeric` with error handling
- Date parsing for temporal columns: `issue_d`, `earliest_cr_line`, `last_pymnt_d`, `last_credit_pull_d`
- String conversion for categorical columns

#### Feature Engineering

**Temporal Features:**
- `credit_history_months`: Time between earliest credit line and loan issue date
- `issue_year` and `issue_month`: Extracted from issue date
- `last_pymnt_recency_months`: Time since last payment (removed later due to leakage)

**Employment Length:**
- Mapped text values ("10+ years", "9 years", etc.) to numeric years
- Handled missing values ("None", "n/a") as NaN

**Geographic Features:**
- `zip3`: First 3 digits of zip code for regional analysis
- State information preserved for geographic patterns

**Derived Financial Ratios:**
- `income_to_debt_ratio`: Annual income relative to loan amount
- `available_credit_ratio`: Available credit utilization
- `payment_to_income_ratio`: Monthly payment relative to income
- `loan_to_income_ratio`: Loan amount relative to annual income
- `revol_util_squared`: Squared revolving utilization (non-linear feature)
- `dti_squared`: Squared debt-to-income ratio
- `inquiry_ratio`: Recent inquiries relative to total accounts
- `open_acc_ratio`: Open accounts relative to total accounts
- `has_delinq`: Binary flag for any delinquencies

#### Data Cleaning

**Removal of High-Missing Columns:**
- Dropped columns with >50% missing values
- Removed very sparse columns (e.g., `il_util`, `mths_since_rcnt_il`, `all_util`)

**Target Variable Creation:**
- Good status: "Fully Paid", "Does not meet the credit policy. Status:Fully Paid"
- Bad status: "Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)", "In Grace Period", "Does not meet the credit policy. Status:Charged Off"
- Target: 1 for bad status, 0 for good status
- Dropped "Current" loans (incomplete)

**Leakage Removal:**
Critical step to remove features that would not be available at loan origination:
- Payment-related: `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `recoveries`, `collection_recovery_fee`, `last_pymnt_amnt`
- Outstanding principal: `out_prncp`, `out_prncp_inv`
- Temporal leakage: `last_pymnt_recency_months`, `last_credit_pull_recency_months`, `mths_since_recent_bc`, `mths_since_recent_inq`
- Other: `pymnt_plan_y`, `disbursement_method_DirectPay`

**Final Feature Set:**
- 134 features after one-hot encoding of categorical variables
- All numeric features imputed with median
- Categorical features imputed with mode

### Model Training

#### Algorithm Selection

**XGBoost Classifier** was chosen as the final model after comparing:
- Random Forest: Good baseline, but lower F1 score
- LightGBM: Competitive performance, but XGBoost slightly better
- XGBoost: Best F1 score with proper hyperparameter tuning

#### Hyperparameters

Final XGBoost configuration:
- `n_estimators`: 500
- `max_depth`: 6
- `learning_rate`: 0.05
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `min_child_weight`: 1
- `gamma`: 0.1
- `reg_alpha`: 0.5 (L1 regularization)
- `reg_lambda`: 2.0 (L2 regularization)
- `scale_pos_weight`: 2.0 (handles class imbalance)
- `tree_method`: "hist" (fast CPU training)
- `eval_metric`: "logloss"
- `random_state`: 42

#### Training Process

1. **Train/Test Split**: 80/20 split with stratification to preserve class distribution
2. **No SMOTE**: Chose not to use SMOTE to avoid overfitting; relied on `scale_pos_weight` instead
3. **Threshold Optimization**: Optimized decision threshold on validation set (not test set) to maximize F1 score
4. **Optimal Threshold**: 0.67 (default probability >= 0.67 triggers rejection)

#### Model Performance

**Final Metrics:**
- Train F1: 0.9301
- Validation F1: 0.9297
- Test F1: 0.9295
- Train-Test Gap: 0.0006 (no overfitting)

**5-Fold Cross-Validation:**
- Mean F1: 0.9296
- Std Dev: 0.0001
- CV-Test Gap: 0.0001 (excellent generalization)

The model achieves F1 score of 0.93, well above the 0.75 requirement, with no evidence of overfitting.

#### Model Artifacts

Saved model includes:
- Trained XGBoost model
- Optimal threshold (0.67)
- Feature list (134 features in exact order)
- Metadata (F1 scores, training flags)

---

## Part 2: Knowledge Graph and Retrieval System

### Knowledge Graph Construction

#### Cohort-Based Design

Instead of storing individual loans (which would be too large), the KG uses aggregated cohorts:

**Cohort Types:**
1. **By Grade** (A, B, C, D, E, F, G): Default rates, average interest rates, loan counts per grade
2. **By Term** (36 months, 60 months): Comparison of short vs long-term loans
3. **By Purpose** (debt_consolidation, credit_card, home_improvement, etc.): Risk by loan purpose
4. **By Status** (Fully Paid, Charged Off): Outcomes and their characteristics
5. **By Home Ownership** (MORTGAGE, RENT, OWN, OTHER, NONE): Borrower characteristics
6. **By Income Band** (<50k, 50k-75k, 75k-100k, 100k-150k, >150k): Income-based risk analysis
7. **By State**: Geographic patterns in interest rates and defaults

**Cohort Properties:**
- `cohortType`: Dimension being aggregated
- `loanCount`: Number of loans in this cohort
- `defaultRate`: Proportion of loans that defaulted (0-1)
- `avgInterestRate`: Average interest rate
- `avgLoanAmount`: Average loan size
- `avgDTI`: Average debt-to-income ratio (where applicable)
- `avgIncome`: Average annual income (where applicable)

#### RDF Schema

**Namespaces:**
- `ex:`: http://example.org/loan# (domain ontology)
- Standard RDF/RDFS/OWL namespaces

**Classes:**
- `ex:Cohort`: Represents an aggregated group of loans

**Properties:**
- Object properties: `ex:cohortType`, `ex:grade`, `ex:term`, `ex:purpose`, etc.
- Datatype properties: `ex:loanCount`, `ex:defaultRate`, `ex:avgInterestRate`, etc.

**Graph Size:**
- 838 triples covering all cohort combinations
- Serialized as Turtle format (`loan_cohorts.ttl`)

### Retrieval Logic

#### Custom SPARQL Implementation

The system implements custom retrieval using SPARQL queries over the RDF graph. No frameworks like LangChain or LlamaIndex are used for retrieval (as per requirements).

#### Query Patterns

**Simple Cohort Query:**
```sparql
PREFIX ex: <http://example.org/loan#>
SELECT ?key ?loanCount ?defaultRate ?avgInterestRate
WHERE {
  ?cohort a ex:Cohort ;
          ex:cohortType "grade" ;
          ex:grade ?key ;
          ex:loanCount ?loanCount ;
          ex:defaultRate ?defaultRate ;
          ex:avgInterestRate ?avgInterestRate .
}
ORDER BY ?key
```

**Comparative Query:**
```sparql
PREFIX ex: <http://example.org/loan#>
SELECT ?term ?defaultRate
WHERE {
  ?cohort a ex:Cohort ;
          ex:cohortType "term" ;
          ex:term ?term ;
          ex:defaultRate ?defaultRate .
}
ORDER BY ?defaultRate DESC
```

#### Question Interpretation

The system uses a simple heuristic router to classify questions:

**Pattern Matching:**
- "grade" → grade dimension
- "term" → term dimension
- "purpose" → purpose dimension
- "home ownership" / "rent" / "mortgage" → home_ownership dimension
- "income" → income_band dimension
- "state" → state dimension

**Limitations:**
- Simple keyword matching fails on complex phrasings
- Prediction questions not always detected
- Temporal queries not supported

### Types of Questions Using KG

The KG handles analytical and comparative questions:

**Aggregation Questions:**
- "What is the default rate by grade?"
- "Compare default rates across income bands"
- "What is the average loan amount for different income bands?"

**Factual Questions:**
- "What is the default rate for grade B loans?"
- "What is the default rate for grade C, 36-month loans?"

**Comparative Questions:**
- "Which term (36 or 60 months) has a higher default rate?"
- "How does default rate differ between borrowers who rent versus those with a mortgage?"
- "Compare default rates for debt_consolidation versus credit_card loans"

**Ranking Questions:**
- "Which three states have the highest average interest rates?"
- "Which loan purpose has the highest default rate?"

**Status/Outcome Questions:**
- "What is the average interest rate for fully paid loans versus charged off loans?"

### Types of Questions Using the Model

The trained XGBoost model handles prediction questions:

**Approval Questions:**
- "Should we approve a $15,000, 36-month loan for a borrower earning $80,000 with 18% DTI and grade B?"
- "Will this loan default?"

**Probability Questions:**
- "What is the predicted default probability for a borrower with grade F, 60-month term, and $20,000 annual income?"
- "What is the risk of default for this applicant?"

**Feature Requirements:**
- Model requires all 134 features for realistic predictions
- Missing features are filled with NaN (which may lead to optimistic predictions)
- For production use, all features should be provided

### How the System Works

**Question Flow:**

1. **User asks question** → Chatbot receives natural language input

2. **Routing Decision:**
   - Simple keyword-based router classifies question type
   - Routes to either KG (`/ask`) or model (`/predict`)

3. **If KG Route (`/ask`):**
   - Question interpreted to extract dimension (grade, term, purpose, etc.)
   - SPARQL query generated and executed over `loan_cohorts.ttl`
   - Results returned as structured JSON with cohort statistics
   - LLM (Ollama) generates natural language summary

4. **If Model Route (`/predict`):**
   - User prompted for key features (loan amount, term, rate, income, DTI, grade, etc.)
   - Complete 134-feature vector constructed with defaults
   - Model predicts default probability
   - Decision made: approve if probability < 0.67, reject if >= 0.67
   - LLM explains the prediction in natural language

5. **Response Generation:**
   - Source information: Which cohorts or model was used
   - Confidence metrics: Default probability, loan counts, model metadata
   - Natural language explanation from LLM

---

## Part 3: REST API

### Architecture

The system exposes a REST API using FastAPI, providing two main endpoints:

### Endpoints

#### POST /predict

**Purpose:** Run the trained XGBoost model on a loan application to predict default probability.

**Request:**
```json
{
  "features": {
    "loan_amnt": 15000,
    "int_rate": 13.5,
    "annual_inc": 80000,
    "dti": 18.0,
    "grade_B": 1,
    ...
  }
}
```

**Response:**
```json
{
  "approved": false,
  "default_probability": 0.937,
  "threshold": 0.67,
  "model_metadata": {
    "train_f1": 0.9301,
    "val_f1": 0.9297,
    "test_f1": 0.9295,
    "no_smote": true,
    "no_test_leakage": true
  }
}
```

**Features:**
- Accepts partial feature dict (missing features filled with NaN)
- Validates feature names against trained model
- Returns approval decision based on threshold (0.67)
- Includes model metadata for transparency

**Limitations:**
- Partial feature vectors may produce unrealistic predictions
- Full 134-feature vector recommended for production use

#### POST /ask

**Purpose:** Query the knowledge graph for analytical questions about loan cohorts.

**Request:**
```json
{
  "question": "What is the default rate by grade?"
}
```

**Response:**
```json
{
  "interpretation": {
    "type": "cohort",
    "dimension": "grade"
  },
  "results": [
    {
      "key": "A",
      "loanCount": 228592,
      "defaultRate": 0.070,
      "avgInterestRate": 7.13
    },
    ...
  ],
  "answer_text": "The default rates by grade are: A: 7.01%, B: 15.14%, ..."
}
```

**Features:**
- Simple keyword-based question interpretation
- SPARQL query generation and execution
- Returns structured cohort statistics
- Includes natural language summary

**Supported Dimensions:**
- grade, term, purpose, status, home_ownership, income_band, state

### API Implementation Details

#### Startup

On application startup:
1. Loads trained model from `loan_classifier_final.pkl`
2. Extracts model, threshold, feature list, metadata
3. Loads or builds knowledge graph from `loan_cohorts.ttl`
4. If KG doesn't exist, builds it from `cleaned_loan_dataset.csv`

#### Error Handling

- Missing features: Filled with NaN (with warning if >50% missing)
- Invalid questions: Returns "unknown" type
- SPARQL errors: Caught and returned as error messages
- Model prediction errors: HTTP 400 with error details

#### Source Information

**For KG Queries:**
- Explicit cohort keys and statistics
- Loan counts for confidence assessment
- Dimension type (grade, term, etc.)

**For Model Predictions:**
- Default probability (0-1 scale)
- Decision threshold used
- Model performance metrics (F1 scores)

#### Confidence Metrics

**KG Queries:**
- `loanCount`: Higher counts indicate more reliable statistics
- Multiple cohorts: Allows comparison and validation

**Model Predictions:**
- `default_probability`: Direct probability estimate
- `threshold`: Decision boundary (0.67)
- `model_metadata`: Training performance indicators

### Integration with LLM

The API is designed to work with external LLM services (Ollama):

**Chatbot Flow:**
1. User question → `chat_ollama.py` receives input
2. LLM routes question (or simple keyword matching)
3. Calls appropriate API endpoint (`/ask` or `/predict`)
4. Receives structured JSON response
5. LLM generates natural language explanation
6. Returns final answer to user

**LLM Responsibilities:**
- Question classification (cohort vs predict)
- Natural language generation from structured data
- Explanation of model decisions
- Handling unanswerable questions

**API Responsibilities:**
- Data retrieval (KG or model)
- Structured response format
- Source attribution
- Confidence metrics

### API Usage Examples

**Example 1: Cohort Query**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the default rate by grade?"}'
```

**Example 2: Model Prediction**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "loan_amnt": 15000,
      "int_rate": 13.5,
      "annual_inc": 80000,
      "dti": 18.0
    }
  }'
```

### Performance Considerations

- KG queries: Fast (838 triples, in-memory RDF graph)
- Model predictions: Fast (XGBoost with hist tree method)
- LLM integration: Depends on Ollama model size and hardware
- Concurrent requests: FastAPI handles async requests efficiently

---

## System Integration

### Complete Workflow

1. **Data Pipeline:**
   - Raw loan data → Preprocessing → Feature engineering → Train/test split
   - Model training → Validation → Threshold optimization → Model saving

2. **KG Pipeline:**
   - Cleaned data → Cohort computation → RDF graph construction → Turtle serialization

3. **API Service:**
   - Loads model and KG at startup
   - Exposes REST endpoints
   - Handles routing and retrieval

4. **Chatbot Interface:**
   - Natural language input → Routing → API calls → LLM explanation → User response

### File Structure

- `test_2.ipynb`: Model training and evaluation
- `loan_classifier_final.pkl`: Saved model artifacts
- `kg_cohorts.py`: KG construction script
- `loan_cohorts.ttl`: Serialized knowledge graph
- `api.py`: FastAPI application
- `chat_ollama.py`: LLM chatbot interface
- `evaluate_system.py`: Evaluation script
- `evaluation_questions.json`: Test questions
- `evaluation_results.json`: Evaluation metrics

### Dependencies

- Core ML: pandas, numpy, scikit-learn, xgboost
- Graph: rdflib
- API: fastapi, uvicorn
- LLM: requests (for Ollama API)
- Utilities: joblib (model serialization)

---

## Conclusion

This system successfully combines:
- Classical ML for individual loan predictions (F1 = 0.93)
- Knowledge graph for analytical queries over loan cohorts
- REST API for programmatic access
- LLM integration for natural language interaction

The architecture separates concerns: KG handles analytics, model handles predictions, API provides unified interface, LLM provides natural language layer. This design allows for independent improvement of each component while maintaining system coherence.

