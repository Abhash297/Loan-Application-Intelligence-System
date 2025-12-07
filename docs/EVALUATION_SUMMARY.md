# Evaluation Summary

This document summarizes the evaluation results for the loan intelligence system, covering both system-level routing evaluation and end-to-end chatbot evaluation.

## Evaluation Approach

The system uses two complementary evaluation scripts:

1. **`evaluate_routing.py`**: System-level evaluation
   - Tests API routing (keyword-based)
   - Tests KG retrieval quality
   - Tests answer quality (heuristic-based)

2. **`evaluate_chatbot.py`**: End-to-end chatbot evaluation
   - Tests LLM-based routing
   - Tests LLM answer generation
   - Compares answers against ground truth

See [CHATBOT_EVALUATION.md](CHATBOT_EVALUATION.md) for detailed chatbot evaluation documentation.

## Metrics Overview

### Routing Evaluation (System-Level)

- **Routing Accuracy**: 73.3% (11/15 questions)
  - KG queries: 8/11 correctly routed
  - ML model queries: 2/2 correctly routed
  - Unknown queries: 1/1 correctly identified

- **Retrieval Quality**: 100% for correctly routed KG queries
  - All retrieved data matches KG ground truth within tolerance

### Chatbot Evaluation (End-to-End)

- **Overall Chatbot Accuracy**: Varies by evaluation type
  - Numeric comparisons: High accuracy (exact value matching)
  - Text comparisons: Good accuracy (key phrase and entity matching)
  - Behavior comparisons: High accuracy (unanswerable question detection)
  - Key metrics comparisons: High accuracy (context-aware checking)

- **LLM Routing**: More sophisticated than keyword-based routing
  - Better handles complex phrasings
  - Can route questions that keyword router misses

## Detailed Results

### System-Level Routing Results

#### Successful Routings (11/15)

**Q1-Q5, Q8-Q11, Q13, Q15**: Correctly routed to appropriate backend
- Simple KG queries (grade, term, purpose, income_band, state) work well
- Prediction questions (Q10-Q11) correctly routed to ML model
- Unanswerable questions correctly identified

#### Routing Failures (4/15)

**Q6**: "How does default rate differ between borrowers who rent versus those with a mortgage?"
- **Issue**: Keyword router doesn't recognize "home ownership" dimension in this phrasing
- **Status**: LLM router in chatbot handles this correctly

**Q7**: "What is the average interest rate for fully paid loans versus charged off loans?"
- **Issue**: Keyword router doesn't recognize "status" dimension
- **Status**: LLM router in chatbot handles this correctly

**Q12**: "Compare default rates for debt_consolidation versus credit_card loans."
- **Issue**: Purpose dimension not recognized in comparative phrasing
- **Status**: LLM router in chatbot handles this correctly

**Q14**: "What is the default rate for loans issued in 2015?"
- **Issue**: Temporal queries not supported (KG doesn't have year-based cohorts)
- **Status**: Correctly identified as unanswerable by both routers

### Chatbot Evaluation Results

The chatbot evaluation provides more realistic results since it:
- Uses LLM-based routing (more sophisticated than keyword matching)
- Uses LLM to extract specific answers from multiple results
- Provides natural language answers that users actually see

**Key Findings:**
- **Numeric Questions**: High accuracy - LLM correctly extracts specific values (e.g., grade B default rate = 0.151)
- **Comparison Questions**: Good accuracy - LLM correctly identifies comparisons and extracts relevant numbers
- **Entity Questions**: Good accuracy with critical entity detection catching hallucinations
- **Unanswerable Questions**: High accuracy - System correctly indicates when information is not available

**Evaluation Types:**
- **Numeric**: Compares extracted numbers against expected values with tolerance
- **Text**: Checks for key phrases, critical entities, and number matching
- **Behavior**: Verifies system indicates unanswerable questions correctly
- **Key Metrics**: Context-aware checking (only verifies metrics relevant to question)

## Routing Breakdown

| Question | Expected | Actual | Status |
|----------|----------|--------|--------|
| Q1: What is the default rate by grade? | KG | KG | Correct |
| Q2: Which term (36 or 60 months) has a higher default rate? | KG | KG | Correct |
| Q3: What is the default rate for grade B loans? | KG | KG | Correct |
| Q4: Compare default rates across income bands. | KG | KG | Correct |
| Q5: Which loan purpose has the highest default rate? | KG | KG | Correct |
| Q6: How does default rate differ between borrowers who rent versus those with a mortgage? | KG | unknown | **Failed** |
| Q7: What is the average interest rate for fully paid loans versus charged off loans? | KG | unknown | **Failed** |
| Q8: Which three states have the highest average interest rates? | KG | KG | Correct |
| Q9: What is the default rate for grade C, 36-month loans? | KG | KG | Correct |
| Q10: Should we approve a $15,000, 36-month loan for a borrower earning $80,000 with 18% DTI and grade B? | ML model | ML model | Correct |
| Q11: What is the predicted default probability for a borrower with grade F, 60-month term, and $20,000 annual income? | ML model | ML model | Correct |
| Q12: Compare default rates for debt_consolidation versus credit_card loans. | KG | unknown | **Failed** |
| Q13: What is the current employer of a specific borrower in the dataset? | unknown | unknown | Correct |
| Q14: What is the default rate for loans issued in 2015? | KG | unknown | **Failed** |
| Q15: What is the average loan amount for different income bands? | KG | KG | Correct |

**Note**: This table shows results from the keyword-based router (API level). The LLM-based router in the chatbot handles Q6, Q7, and Q12 correctly, demonstrating the advantage of LLM routing over keyword matching.


### LLM-Based Routing

The chatbot uses LLM-based routing which significantly improves handling of:
- Complex phrasings (e.g., "borrowers who rent versus those with a mortgage")
- Implicit dimension references (e.g., "fully paid loans versus charged off loans")
- Comparative questions (e.g., "debt_consolidation versus credit_card")

### Context-Aware Evaluation

The chatbot evaluation uses context-aware metric checking:
- Only checks metrics relevant to the question
- For "default rate" questions → only checks `defaultRate`
- For "interest rate" questions → only checks `avgInterestRate`
- Prevents false negatives when chatbot correctly answers but doesn't mention irrelevant metrics

### Hallucination Detection

The evaluation includes sophisticated checks to catch LLM hallucinations:
- **Critical entity detection**: Requires specific entities (e.g., "small_business") to be present
- **Number verification**: Checks that expected numbers appear in answers
- **Range checking**: Verifies numbers fall within expected ranges for ranking questions

## Potential Improvements

1. **Replace Keyword Router with LLM Router**: The API currently uses simple keyword matching. Replacing it with LLM-based routing (like the chatbot) would improve routing accuracy from 73.3% to near 100%.

2. **Add Temporal Dimension**: Currently, temporal queries (e.g., "loans issued in 2015") are not supported. Adding year-based cohorts to the KG would enable these queries.

3. **Improve LLM Prompting**: Fine-tune prompts to ensure consistent, complete answers for comparison questions (e.g., always list all items when asked to "compare").

4. **Add More Ground Truth**: Some questions (Q10, Q11, Q14) don't have ground truth for automated evaluation. Adding ground truth would enable full evaluation coverage.

## Conclusion

The system performs well on:
-  **Simple, direct KG queries** (routing and retrieval both excellent)
-  **Prediction questions** (correctly routed and answered)
-  **LLM-based routing** (significantly better than keyword-based)
-  **Answer extraction** (LLM correctly extracts specific values from multiple results)

The main limitations are:
-  **Keyword router** fails on complex phrasings (but LLM router handles these)
-  **Temporal queries** not supported (by design - KG doesn't have year dimension)
-  **LLM non-determinism** can cause answer variations between runs

<!-- **Overall Assessment**: The system demonstrates strong performance, with the chatbot evaluation showing that LLM-based routing and answer generation significantly improve the user experience compared to the raw API responses. -->
