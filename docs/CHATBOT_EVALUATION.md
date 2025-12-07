# Chatbot Evaluation Guide

This document explains how to evaluate the chatbot system using the `evaluate_chatbot.py` script, which tests the full end-to-end chatbot pipeline (LLM routing + LLM answer generation).

## Overview

The chatbot evaluation script (`src/evaluate_chatbot.py`) evaluates the complete user-facing experience:
- **LLM-based routing**: Uses Ollama to intelligently route questions (more sophisticated than keyword-based routing)
- **LLM answer generation**: Uses Ollama to extract specific answers from structured API responses
- **Ground truth comparison**: Compares chatbot answers against expected values/answers

This is different from `evaluate_routing.py`, which tests system-level components (API routing, KG retrieval) separately.

## Prerequisites

1. **API must be running:**
   ```bash
   uvicorn src.api:app --reload
   ```

2. **Ollama must be running:**
   ```bash
   ollama serve
   ```

3. **Ollama model must be available:**
   ```bash
   ollama pull llama3
   ```

## Running the Evaluation

```bash
python src/evaluate_chatbot.py
```

The script will:
1. Load questions from `evaluation/evaluation_questions.json`
2. For each question with ground truth:
   - Call the chatbot (`chat_once()`)
   - Compare the LLM-generated answer against ground truth
   - Report match/no match
3. Skip questions without ground truth (e.g., Q10, Q11, Q14)
4. Generate summary statistics

## Output

### Console Output

For each question, we'll see:
- Question ID and text
- **Route chosen** - Which backend the LLM router selected (cohort/predict/unknown)
- **API Results Count** - How many results the API returned (for cohort questions)
- **First result keys** - What data fields are in the API response (for debugging)
- Chatbot answer (first 300 characters)
- **Answer Length** - Total length of the chatbot answer
- Ground truth comparison result (MATCH/NO MATCH)
- Comparison type (numeric, text, behavior, key_metrics)
- Explanation of the comparison

The debug information helps diagnose issues:
- If routing is incorrect, we'll see the wrong route chosen
- If API returns incomplete data, we'll see fewer results than expected
- If LLM truncates the answer, we'll see a shorter answer length

### Summary Statistics

The script provides:
- Total questions, evaluated count, skipped count
- List of skipped questions
- Accuracy by comparison type:
  - Numeric comparisons
  - Text comparisons
  - Behavior comparisons
  - Key metrics comparisons
- Overall chatbot accuracy

### JSON Output

Detailed results are saved to:
```
evaluation/chatbot_evaluation.json
```

Each result includes:
- Question details
- Full chatbot answer
- Ground truth comparison details
- Match status
- Explanation
<!-- 
## Evaluation Types

### 1. Numeric Comparison

For questions with `expected_value` in ground truth:
- Extracts numeric value from chatbot answer
- Compares against expected value with tolerance
- Example: Q3 - "What is the default rate for grade B loans?"
  - Expected: ~0.151 (15.1%)
  - Tolerance: 0.02

### 2. Text Comparison

For questions with `expected_answer` in ground truth:
- Extracts key phrases from expected answer
- Checks if chatbot answer contains critical entities
- Verifies numbers match (for questions with specific values)
- Example: Q2 - "Which term has higher default rate?"
  - Expected: "60 months has higher default rate (~0.355 vs ~0.178)"
  - Checks: Both numbers present, comparison term present

### 3. Behavior Comparison

For unanswerable questions with `expected_behavior`:
- Checks if chatbot indicates information is not available
- Looks for keywords like "not available", "cannot", "unable"
- Example: Q13 - "What is the current employer of a specific borrower?"
  - Expected: System should indicate this information is not available

### 4. Key Metrics Comparison

For questions with `key_metrics` in ground truth:
- Context-aware: Only checks metrics relevant to the question
- For "default rate" questions → only checks `defaultRate`
- For "interest rate" questions → only checks `avgInterestRate`
- Verifies answer mentions relevant metrics and contains numbers
- Example: Q1 - "What is the default rate by grade?"
  - Relevant metrics: Only `defaultRate` (not `loanCount` or `avgInterestRate`) -->

## Key Features

### Context-Aware Metric Checking

The evaluation checks important keywords 
- **"default rate" questions** → Only checks `defaultRate`
- **"interest rate" questions** → Only checks `avgInterestRate`
- **"loan amount" questions** → Only checks `avgLoanAmount`
- **General questions** → Checks all listed metrics

This prevents false negatives when the chatbot correctly answers the question but doesn't mention irrelevant metrics.

### Entity Detection

For text comparisons, the evaluation:
- Extracts specific entities from expected answer (e.g., "small_business", "debt_consolidation")
- Requires these critical entities to be present in chatbot answer
- Catches hallucinations where chatbot gives wrong entity names

### Number Verification

For questions with specific numeric values:
- Extracts numbers from expected answer (e.g., "12.64%", "15.71%")
- Verifies these numbers appear in chatbot answer (within tolerance)
- Catches hallucinations where chatbot gives wrong numbers

### Range Checking

For ranking questions:
- Checks if numbers fall within expected ranges
- Example: Q8 - "Which three states have highest interest rates?"
  - Expected: "rates around 15.5-16.8%"
  - Verifies: At least one mentioned rate is in this range

## Skipped Questions

Questions without ground truth are automatically skipped:
- **Q10, Q11**: Prediction questions (require full feature vectors, no simple ground truth)
- **Q14**: Time-based question (may not be directly available in KG)

These questions are still answered by the chatbot, but cannot be automatically evaluated.

## Comparison with `evaluate_routing.py`

| Feature | `evaluate_routing.py` | `evaluate_chatbot.py` |
|---------|----------------------|----------------------|
| **Routing** | Keyword-based (API) | LLM-based (Chatbot) |
| **Answer** | Raw API JSON summary | LLM-generated natural language |
| **Number Extraction** | First number found | LLM extracts correct specific number |
| **Use Case** | System-level testing | User-facing testing |
| **Example** | Gets 0.0701 (wrong grade) | Gets 0.151 (correct grade B) |

## Troubleshooting

### "No ground truth comparison available"

This means the question doesn't have:
- `expected_value`
- `expected_answer`
- `expected_behavior`
- `key_metrics`

The question will be skipped automatically.

### "Critical entity not found"

The chatbot answer doesn't contain the specific entity mentioned in the expected answer. This could indicate:
- Hallucination (chatbot gave wrong entity)
- Answer format issue (entity mentioned differently)

### "Expected numbers not found"

The chatbot answer doesn't contain the specific numeric values from the expected answer. This could indicate:
- Hallucination (chatbot gave wrong numbers)
- Number format issue (percentages vs decimals)


<!-- 
## Best Practices

1. **Run both evaluations**: Use `evaluate_routing.py` for system-level testing and `evaluate_chatbot.py` for user-facing testing
2. **Check skipped questions**: Review why questions were skipped and add ground truth if needed
3. **Review hallucinations**: Pay attention to NO MATCH results - they may indicate LLM hallucinations
4. **Update ground truth**: As you improve the system, update ground truth values to match actual KG/ML model outputs -->

## Example Output

```
============================================================
Question Q3: What is the default rate for grade B loans?
  Route: cohort
  API Results Count: 7
  First result keys: ['key', 'loanCount', 'defaultRate', 'avgInterestRate', 'avgLoanAmount']
  Chatbot Answer: According to the provided data, the default rate for grade B loans is 15.14%...
  Answer Length: 156 characters
  Ground Truth:  MATCH
  Type: numeric
  Actual: 0.1514, Expected: 0.1510, Diff: 0.0004, Tolerance: 0.0200
  Actual: 0.1514
  Expected: 0.1510

============================================================
SUMMARY
============================================================

Total Questions: 15
Evaluated: 12
Skipped (no ground truth): 3

--- Skipped Questions ---
  Q10: Should we approve a $15,000, 36-month loan for a borrower...
  Q11: What is the predicted default probability for a borrower...
  Q14: What is the default rate for loans issued in 2015?

--- Ground Truth Accuracy (Chatbot Answers) ---
Numeric Comparisons: 2/2 (100.0%)
Text Comparisons: 5/7 (71.4%)
Behavior Comparisons: 1/1 (100.0%)
Key Metrics Comparisons: 3/3 (100.0%)

Overall Chatbot Accuracy: 11/13 (84.6%)
```

