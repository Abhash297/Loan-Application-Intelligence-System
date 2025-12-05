# Loan Application Intelligence System

An intelligent loan application system that combines classical machine learning for default prediction with a knowledge graph-based chatbot for loan analytics. The system provides both predictive capabilities (should we approve this loan?) and analytical capabilities (what are the default rates by grade?).

## Overview

This project implements a complete loan intelligence system with three main components:

1. **Machine Learning Classifier**: XGBoost model trained to predict loan defaults with F1 score of 0.93
2. **Knowledge Graph**: RDF-based graph storing aggregated loan cohort statistics for analytical queries
3. **REST API + Chatbot**: FastAPI endpoints with Ollama LLM integration for natural language interaction

The system can answer questions like:
- "What is the default rate by grade?" (KG-based analytics)
- "Should we approve a $15,000 loan for a borrower with grade B?" (ML prediction)
- "Compare default rates between 36-month and 60-month loans" (KG comparison)

## Project Structure

```
archive/
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
│
├── docs/                              # Documentation
│   ├── PROJECT_DOCUMENTATION.md       # Comprehensive project documentation
│   ├── MODEL_CHOICE_ANALYSIS.md       # ML model selection and analysis
│   ├── EVALUATION_SUMMARY.md          # System evaluation results
│   ├── routing_breakdown.md           # Routing accuracy analysis
│   ├── RETRIEVAL_LOGIC.md             # Detailed retrieval logic explanation
│   └── KG_EXPLANATION.md              # Knowledge graph explanation
│
├── notebooks/                         # Jupyter notebooks
│   ├── test_1.ipynb                   # Initial exploration notebook
│   └── test_2.ipynb                   # Model training pipeline
│
├── src/                               # Source code
│   ├── api.py                         # FastAPI REST API
│   ├── chat_ollama.py                 # LLM chatbot interface
│   ├── kg_cohorts.py                  # Knowledge graph construction
│   └── evaluate_system.py             # Evaluation script
│
├── scripts/                           # Utility scripts
│   ├── demo_predict.py                # Demo script for /predict endpoint
│   ├── test_risky_loan.py             # Test script for risky loan scenarios
│   ├── test_sparql.py                 # SPARQL query examples
│   ├── find_low_risk_demo.py         # Helper to find low-risk examples
│   └── find_rejected_example.py      # Helper to find rejected examples
│
├── artifacts/                         # Saved models and KG
│   ├── loan_classifier_final.pkl      # Trained XGBoost model (2.2MB)
│   └── loan_cohorts.ttl               # Serialized RDF graph (838 triples)
│
├── data/                              # Data files (gitignored)
│   ├── loan.csv                       # Original dataset (1.1GB)
│   ├── cleaned_loan_dataset.csv       # Cleaned dataset (634MB)
│   ├── LCDataDictionary.xlsx         # Data dictionary
│   └── backup data/                   # Backup data folder
│
└── evaluation/                        # Evaluation data
    ├── evaluation_questions.json      # Test questions (15 questions)
    └── evaluation_results.json        # Evaluation metrics
```

### Key Files

**Model Training:**
- `notebooks/test_2.ipynb`: Complete model training pipeline with feature engineering, hyperparameter tuning, and evaluation

**Knowledge Graph:**
- `src/kg_cohorts.py`: Builds cohort-based KG from loan data
- `artifacts/loan_cohorts.ttl`: Serialized RDF graph with 7 cohort dimensions
- `scripts/test_sparql.py`: Example SPARQL queries over the KG

**API & Chatbot:**
- `src/api.py`: FastAPI application with `/predict` and `/ask` endpoints
- `src/chat_ollama.py`: Interactive chatbot using Ollama LLM

**Evaluation:**
- `src/evaluate_system.py`: Automated evaluation script
- `evaluation/evaluation_questions.json`: 15 test questions covering all query types
- `evaluation/evaluation_results.json`: Detailed evaluation metrics

## Documentation

### Main Documentation

- **[docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md)**: Comprehensive documentation covering all three parts:
  - Part 1: Machine Learning (data preprocessing, model training, performance)
  - Part 2: Knowledge Graph (construction, retrieval logic, question types)
  - Part 3: REST API (endpoints, implementation, integration)

### Additional Documentation

- **[docs/MODEL_CHOICE_ANALYSIS.md](docs/MODEL_CHOICE_ANALYSIS.md)**: Detailed analysis of model selection, hyperparameter tuning, and performance metrics

- **[docs/EVALUATION_SUMMARY.md](docs/EVALUATION_SUMMARY.md)**: System evaluation results including:
  - Routing accuracy (73.3% - 11/15 questions)
  - Detailed routing breakdown table
  - Failure mode analysis

- **[docs/routing_breakdown.md](docs/routing_breakdown.md)**: Detailed breakdown of question routing, showing which questions were routed correctly and which failed

- **[docs/RETRIEVAL_LOGIC.md](docs/RETRIEVAL_LOGIC.md)**: Complete explanation of retrieval logic for both KG and ML model backends

- **[docs/KG_EXPLANATION.md](docs/KG_EXPLANATION.md)**: Detailed explanation of the knowledge graph structure and design

## Quick Start

### Prerequisites

- Python 3.8+
- Ollama installed and running (for LLM chatbot)
- Required Python packages (see requirements.txt)

### Installation

1. **Install Python dependencies:**
```bash
pip install fastapi uvicorn rdflib joblib pandas scikit-learn xgboost requests
```

2. **Install Ollama:**
```bash
# Download from https://ollama.com or use:
brew install ollama  # macOS
```

3. **Pull Ollama model:**
```bash
ollama pull llama3
```

### Running the System

1. **Start the REST API:**
```bash
cd /path/to/archive
uvicorn src.api:app --reload
```
API will be available at `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`

2. **Start Ollama (if not already running):**
```bash
ollama serve
```

3. **Run the chatbot:**
```bash
python src/chat_ollama.py
```

### Example Usage

**API Endpoint - KG Query:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the default rate by grade?"}'
```

**API Endpoint - ML Model Prediction:**
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

**Chatbot:**
```bash
python src/chat_ollama.py
# Then type: "What is the default rate by grade?"
```

## System Architecture

```
User Question
    ↓
Chatbot (chat_ollama.py)
    ↓
Router (KG vs ML model)
    ↓
    ├─→ /ask → KG (loan_cohorts.ttl) → SPARQL → Cohort Stats
    │
    └─→ /predict → XGBoost Model → Default Probability
    ↓
LLM (Ollama) → Natural Language Answer
```

## Key Features

- **High-Performance ML Model**: F1 score of 0.93, no overfitting, proper train/test validation
- **Knowledge Graph Analytics**: 7 cohort dimensions (grade, term, purpose, status, home ownership, income band, state)
- **Custom SPARQL Retrieval**: No LangChain/LlamaIndex (as per requirements)
- **REST API**: FastAPI with `/predict` and `/ask` endpoints
- **LLM Integration**: Ollama for natural language generation
- **Source Attribution**: All answers include source information and confidence metrics
- **Comprehensive Evaluation**: 15 test questions with automated metrics

## Model Performance

- **F1 Score**: 0.9295 (test set)
- **Threshold**: 0.67 (optimized on validation set)
- **No Overfitting**: Train-Test gap < 0.001
- **Cross-Validation**: 5-fold CV used to validate model performance

## Evaluation Results

- **Routing Accuracy**: 73.3% (11/15 questions)
  - KG queries: 8/11 correctly routed
  - ML model queries: 2/2 correctly routed
  - Unknown queries: 1/1 correctly identified

See [docs/EVALUATION_SUMMARY.md](docs/EVALUATION_SUMMARY.md) for detailed routing breakdown and analysis.

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, xgboost
- rdflib (for RDF/SPARQL)
- fastapi, uvicorn (for REST API)
- requests (for Ollama API)
- joblib (for model serialization)

## Data Files

Large data files are gitignored:
- `loan.csv` (1.1GB): Original Lending Club dataset
- `cleaned_loan_dataset.csv` (634MB): Preprocessed dataset

The trained model (`loan_classifier_final.pkl`) and knowledge graph (`loan_cohorts.ttl`) are included as they are required to run the system.

## License

This project was created as a take-home assignment for Inductiv AI Engineer position.

## Contact

For questions or issues, refer to the detailed documentation files listed above.

