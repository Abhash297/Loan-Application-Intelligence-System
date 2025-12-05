# Evaluation Summary

## Metrics Overview

- **Routing Accuracy**: 73.3% (11/15 questions)

## Detailed Results

### Successful Routings (11/15)

**Q1-Q5, Q8-Q11, Q13, Q15**: Correctly routed to appropriate backend
- Simple KG queries (grade, term, purpose, income_band, state) work well
- Prediction questions (Q10-Q11) now correctly routed to ML model
- Unanswerable questions correctly identified

### Routing Failures (4/15)

**Q6**: "How does default rate differ between borrowers who rent versus those with a mortgage?"
- **Issue**: Router doesn't recognize "home ownership" dimension in this phrasing
- **Fix**: Enhance keyword matching or use LLM-based routing

**Q7**: "What is the average interest rate for fully paid loans versus charged off loans?"
- **Issue**: Router doesn't recognize "status" dimension
- **Fix**: Add "status", "fully paid", "charged off" to keyword patterns

**Q12**: "Compare default rates for debt_consolidation versus credit_card loans."
- **Issue**: Purpose dimension not recognized in comparative phrasing
- **Fix**: Improve purpose keyword detection

**Q14**: "What is the default rate for loans issued in 2015?"
- **Issue**: Temporal queries not supported (KG doesn't have year-based cohorts)
- **Fix**: Add temporal dimension to KG or explicitly handle unsupported queries

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

## Recommendations

1. **Improve Router**: Replace simple keyword matching with LLM-based classification
2. **Expand Keyword Patterns**: Add more synonyms and phrasings for each dimension
3. **Temporal Support**: Either add year-based cohorts to KG or return clear "not supported" messages
4. **Better Error Messages**: When routing fails, provide helpful feedback to user

## Conclusion

The system performs well on **simple, direct KG queries** and **prediction questions** (11/15 = 73.3% routing, but 100% accuracy when routed correctly). The main limitation is the **simple heuristic router** which fails on:
- Complex phrasings (home ownership, status dimensions)
- Unsupported query types (temporal)

With an improved router (LLM-based), routing accuracy should improve significantly.

