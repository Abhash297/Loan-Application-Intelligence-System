# Evaluation Summary

## Metrics Overview

- **Routing Accuracy**: 60.0% (9/15 questions)
- **Retrieval Quality**: 66.67% (when KG is queried correctly)
- **Answer Quality**: 80.00% (LLM responses are generally good)

## Detailed Results

### Successful Routings (9/15)

✅ **Q1-Q5, Q8-Q9, Q13, Q15**: Correctly routed to appropriate backend
- Simple cohort queries (grade, term, purpose, income_band, state) work well
- Unanswerable questions correctly identified

### Routing Failures (6/15)

❌ **Q6**: "How does default rate differ between borrowers who rent versus those with a mortgage?"
- **Issue**: Router doesn't recognize "home ownership" dimension in this phrasing
- **Fix**: Enhance keyword matching or use LLM-based routing

❌ **Q7**: "What is the average interest rate for fully paid loans versus charged off loans?"
- **Issue**: Router doesn't recognize "status" dimension
- **Fix**: Add "status", "fully paid", "charged off" to keyword patterns

❌ **Q10-Q11**: Prediction questions
- **Issue**: Questions asking for approval/prediction routed to cohort instead of model
- **Fix**: Add "approve", "predict", "default probability" keywords to router

❌ **Q12**: "Compare default rates for debt_consolidation versus credit_card loans."
- **Issue**: Purpose dimension not recognized in comparative phrasing
- **Fix**: Improve purpose keyword detection

❌ **Q14**: "What is the default rate for loans issued in 2015?"
- **Issue**: Temporal queries not supported (KG doesn't have year-based cohorts)
- **Fix**: Add temporal dimension to KG or explicitly handle unsupported queries

## Retrieval Quality

When questions are correctly routed to KG:
- **100% accuracy** for simple cohort queries (grade, term, purpose, income_band, state)
- Numbers match ground truth from SPARQL queries
- Average retrieval score: **66.67%** (lowered by failed routings)

## Answer Quality

LLM responses are generally strong:
- **80% average score**
- Contains relevant numbers
- Coherent and addresses question type
- Some comparative questions could better highlight differences

## Recommendations

1. **Improve Router**: Replace simple keyword matching with LLM-based classification
2. **Expand Keyword Patterns**: Add more synonyms and phrasings for each dimension
3. **Handle Prediction Questions**: Explicitly detect "approve", "predict", "probability" keywords
4. **Temporal Support**: Either add year-based cohorts to KG or return clear "not supported" messages
5. **Better Error Messages**: When routing fails, provide helpful feedback to user

## Conclusion

The system performs well on **simple, direct cohort queries** (9/15 = 60% routing, but 100% accuracy when routed correctly). The main limitation is the **simple heuristic router** which fails on:
- Complex phrasings
- Prediction questions
- Unsupported query types (temporal)

With an improved router (LLM-based), routing accuracy should approach 90%+, and overall system performance would improve significantly.

