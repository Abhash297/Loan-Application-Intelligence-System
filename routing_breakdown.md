# Routing Breakdown

## Questions Routed Correctly (9/15)

| Question | Expected | Actual | Status |
|----------|----------|--------|--------|
| Q1: Default rate by grade | cohort | cohort | ✅ |
| Q2: Term comparison | cohort | cohort | ✅ |
| Q3: Grade B default rate | cohort | cohort | ✅ |
| Q4: Income bands | cohort | cohort | ✅ |
| Q5: Purpose highest default | cohort | cohort | ✅ |
| Q8: States highest rates | cohort | cohort | ✅ |
| Q9: Grade C 36-month | cohort | cohort | ✅ |
| Q13: Unanswerable (employer) | unknown | unknown | ✅ |
| Q15: Loan amount by income | cohort | cohort | ✅ |

## Questions Routed Incorrectly (6/15)

| Question | Expected | Actual | Issue |
|----------|----------|--------|-------|
| Q6: Rent vs mortgage | cohort | **unknown** | Router doesn't recognize "home ownership" phrasing |
| Q7: Fully paid vs charged off | cohort | **unknown** | Router doesn't recognize "status" dimension |
| Q10: Should we approve... | **predict** | **cohort** | Prediction question routed to KG instead of model |
| Q11: Predicted default probability | **predict** | **cohort** | Same - prediction not detected |
| Q12: debt_consolidation vs credit_card | cohort | **unknown** | Purpose not recognized in comparative phrasing |
| Q14: Loans issued in 2015 | cohort | **unknown** | Temporal queries not supported by KG |

## Summary

- **Cohort → Cohort**: 8/9 correct (88.9% for cohort questions)
- **Predict → Predict**: 0/2 correct (0% - prediction routing broken)
- **Unknown → Unknown**: 1/1 correct (unanswerable handled correctly)
- **Overall**: 9/15 = 60%

## Root Cause

The simple heuristic router in `api.py` (`_interpret_question_simple`) only checks for basic keywords:
- "grade" → grade dimension
- "term" → term dimension  
- "purpose" → purpose dimension
- etc.

It fails on:
- Complex phrasings ("rent versus mortgage" doesn't match "home ownership")
- Prediction keywords ("approve", "predict", "probability" not checked)
- Unsupported dimensions (temporal, status not in simple patterns)

