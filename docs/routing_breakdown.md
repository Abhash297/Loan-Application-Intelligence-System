# Routing Breakdown

## Questions Routed Correctly (11/15)

| Question | Expected | Actual | Status |
|----------|----------|--------|--------|
| Q1: Default rate by grade | KG | KG | Correct |
| Q2: Term comparison | KG | KG | Correct |
| Q3: Grade B default rate | KG | KG | Correct |
| Q4: Income bands | KG | KG | Correct |
| Q5: Purpose highest default | KG | KG | Correct |
| Q8: States highest rates | KG | KG | Correct |
| Q9: Grade C 36-month | KG | KG | Correct |
| Q10: Should we approve... | ML model | ML model | Correct |
| Q11: Predicted default probability | ML model | ML model | Correct |
| Q13: Unanswerable (employer) | unknown | unknown | Correct |
| Q15: Loan amount by income | KG | KG | Correct |

## Questions Routed Incorrectly (4/15)

| Question | Expected | Actual | Issue |
|----------|----------|--------|-------|
| Q6: Rent vs mortgage | KG | **unknown** | Router doesn't recognize "home ownership" phrasing |
| Q7: Fully paid vs charged off | KG | **unknown** | Router doesn't recognize "status" dimension |
| Q12: debt_consolidation vs credit_card | KG | **unknown** | Purpose not recognized in comparative phrasing |
| Q14: Loans issued in 2015 | KG | **unknown** | Temporal queries not supported by KG |

## Summary

- **KG → KG**: 8/11 correct (72.7% for KG questions)
- **ML model → ML model**: 2/2 correct (100% - prediction routing fixed)
- **Unknown → Unknown**: 1/1 correct (unanswerable handled correctly)
- **Overall**: 11/15 = 73.3%

## Root Cause

The simple heuristic router in `src/api.py` (`_interpret_question_simple`) checks for keywords:
- **Prediction keywords** (checked first): "approve", "predict", "predicted", "default probability", "probability", "should we", "will this", "risk of default", "default risk"
- **KG dimension keywords**: "grade" → grade dimension, "term" → term dimension, "purpose" → purpose dimension, etc.

It still fails on:
- Complex phrasings ("rent versus mortgage" doesn't match "home ownership")
- Unsupported dimensions (temporal, status not in simple patterns)
- Comparative phrasing variations (e.g., "debt_consolidation versus credit_card" not recognized as purpose)

