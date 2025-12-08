# Project Limitations

This document outlines the known limitations of the Loan Application Intelligence System across all components: Machine Learning model, Knowledge Graph, API, and Chatbot.

## Machine Learning Model Limitations

### 1. **Feature Availability**
- **Limitation**: The model only uses features available at loan application time (no post-issuance data)
- **Impact**: Cannot leverage payment history or loan performance data that becomes available after approval
- **Rationale**: Designed to be realistic for actual loan approval scenarios

### 2. **Class Imbalance**
- **Limitation**: Significant class imbalance (86.77% good loans vs 13.23% bad loans)
- **Mitigation**: Used `scale_pos_weight` and threshold optimization to handle imbalance
- **Impact**: 
  - The imbalanced training data means the model's raw probability outputs may be biased toward predicting "good loan" (the majority class)
  - However, the high threshold (0.67) counteracts this bias by requiring high confidence before approval
  - This creates a conservative approval system: the model may output high probabilities for "good loan", but we only approve when confidence exceeds 67%

### 3. **Threshold Dependency**
- **Limitation**: Model performance is highly dependent on the decision threshold (optimized at 0.67)
- **Impact**: 
  - Default threshold (0.5) yields lower F1 score (0.77 vs 0.93)
  - High threshold (0.67) means only loans with >67% probability of being "good" are approved
  - **This significantly reduces loan approval rates** - only the most confident predictions are approved
  - More conservative approach: fewer approvals but higher confidence in approved loans
- **Note**: Threshold was optimized on validation set to avoid test set leakage. The high threshold prioritizes precision (avoiding bad loans) over recall (approving all good loans)

### 4. **Model Interpretability**
- **Limitation**: XGBoost is a black-box model with limited interpretability
- **Impact**: Difficult to explain individual predictions or understand feature interactions
- **Alternative**: Could use SHAP values or feature importance, but not implemented

### 5. **Data Leakage Prevention**
- **Limitation**: Many features in the original dataset contain post-issuance information
- **Mitigation**: Explicitly excluded features like payment history, settlement flags, hardship data
- **Impact**: Reduced feature set but ensures realistic deployment scenario

## Knowledge Graph Limitations

### 1. **Cohort Dimensions**
- **Limitation**: Only 7 cohort dimensions supported (grade, term, purpose, status, home ownership, income band, state)
- **Impact**: Cannot answer questions about other dimensions (e.g., employment length, verification status)
- **Rationale**: Focused on most important analytical dimensions

### 2. **Temporal Queries**
- **Limitation**: No temporal dimension (year/month-based cohorts) in the knowledge graph
- **Impact**: Cannot answer questions like "What is the default rate for loans issued in 2015?"
- **Example Failed Query**: Q14 - "What is the default rate for loans issued in 2015?"
- **Rationale**: Designed for aggregate statistics, not time-series analysis

### 3. **Multi-Dimensional Queries**
- **Limitation**: Cannot query combined dimensions that aren't pre-computed (e.g., grade + term combinations)
- **Impact**: Questions like "What is the default rate for grade C, 36-month loans?" may return grade C rate without term specification
- **Example**: Q9 - System provides grade C rate but doesn't indicate term-specific limitation

### 4. **SPARQL Query Complexity**
- **Limitation**: Custom SPARQL queries are hardcoded for specific question patterns
- **Impact**: Requires manual query writing for new question types
- **Alternative**: Could use query templates or natural language to SPARQL translation

### 5. **Data Aggregation**
- **Limitation**: Only stores pre-aggregated statistics (default rates, averages, counts)
- **Impact**: Cannot perform ad-hoc aggregations or drill down to individual loan records
- **Rationale**: Designed for fast analytical queries, not detailed record inspection

### 6. **Graph Size**
- **Limitation**: Knowledge graph contains 838 triples (relatively small)
- **Impact**: Limited coverage of all possible cohort combinations
- **Note**: Size is appropriate for the 7 dimensions but could expand with more dimensions

## API Limitations

### 1. **Keyword-Based Routing**
- **Limitation**: API uses simple keyword matching for routing (not LLM-based)
- **Impact**: Routing accuracy of 73.3% (11/15 questions) - fails on complex phrasings
- **Example Failures**: 
  - Q6: "borrowers who rent versus those with a mortgage" (home ownership not recognized)
  - Q7: "fully paid loans versus charged off loans" (status dimension not recognized)
  - Q12: "debt_consolidation versus credit_card" (comparative phrasing not handled)
- **Note**: Chatbot uses LLM-based routing which handles these correctly

### 2. **Error Handling**
- **Limitation**: Limited error messages for unanswerable queries
- **Impact**: May return empty results or generic errors instead of helpful explanations
- **Improvement**: Could add more descriptive error messages

### 3. **Response Format**
- **Limitation**: API returns raw JSON data without natural language explanation
- **Impact**: Requires client-side processing to generate user-friendly responses
- **Note**: Chatbot handles this by using LLM to format responses

### 4. **Rate Limiting**
- **Limitation**: No rate limiting implemented
- **Impact**: Could be overwhelmed by high request volume
- **Note**: Suitable for demonstration but would need rate limiting for production

## Chatbot Limitations

### 1. **LLM Non-Determinism**
- **Limitation**: Ollama LLM responses can vary between runs for the same question
- **Impact**: Evaluation results may differ slightly between runs (chatbot accuracy: 58.3% ± variation)
- **Mitigation**: Could use temperature=0 for more deterministic outputs

### 2. **Answer Accuracy**
- **Limitation**: Overall chatbot accuracy of 58.3% (7/12 evaluated questions match ground truth)
- **Breakdown**:
  - Numeric comparisons: 100% (1/1)
  - Text comparisons: 33.3% (2/6) - struggles with entity extraction and number matching
  - Behavior comparisons: 50% (1/2) - sometimes doesn't indicate unanswerable questions
  - Key metrics comparisons: 100% (3/3)
- **Impact**: May provide incorrect or incomplete answers, especially for comparative questions

### 3. **Hallucination Risk**
- **Limitation**: LLM may generate plausible-sounding but incorrect information
- **Examples**:
  - Q6: Provided default rates (2.5% vs 4.8%) but critical entity "than" missing
  - Q8: Provided interest rates but numbers don't match expected values
  - Q12: Generated default rates but critical entity "debt_consolidation" not found in answer
- **Mitigation**: Evaluation includes critical entity detection and number verification

### 4. **Context Window**
- **Limitation**: Limited by LLM context window size
- **Impact**: May truncate long responses or fail to include all relevant information
- **Note**: Current implementation handles typical query lengths well

### 5. **Prompt Engineering**
- **Limitation**: Prompts are manually crafted and may not handle all edge cases
- **Impact**: Inconsistent answer formatting, especially for comparison questions
- **Improvement**: Could fine-tune prompts based on evaluation feedback

### 6. **Ground Truth Coverage**
- **Limitation**: Not all questions have ground truth for automated evaluation
- **Impact**: Q10, Q11, Q14 are skipped in evaluation (no verifiable metrics)
- **Note**: These are prediction questions or temporal queries without ground truth

## System-Wide Limitations

### 1. **Evaluation Coverage**
- **Limitation**: Only 15 test questions used for evaluation
- **Impact**: May not cover all possible query types or edge cases
- **Improvement**: Could expand test suite with more diverse questions

### 2. **Data Quality**
- **Limitation**: Depends on quality of input dataset (Lending Club data)
- **Impact**: Missing values, data inconsistencies may affect model and KG quality
- **Note**: Implemented data cleaning but some issues may remain

### 3. **Scalability**
- **Limitation**: Not tested with very large datasets or high concurrent request volumes
- **Impact**: Performance characteristics unknown for production-scale deployment
- **Note**: Suitable for demonstration but would need load testing for production

### 4. **Deployment**
- **Limitation**: Requires Ollama to be running locally for chatbot functionality
- **Impact**: Not suitable for cloud deployment without Ollama server setup
- **Alternative**: Could use cloud-based LLM APIs (OpenAI, Anthropic, etc.)

### 5. **Documentation**
- **Limitation**: Some implementation details may not be fully documented
- **Impact**: May require code inspection to understand certain design decisions
- **Note**: Comprehensive documentation exists but may have gaps
<!-- 
## Known Issues

### 1. **Feature Mismatch**
- **Issue**: Model trained on one feature set may not match features in production data
- **Impact**: Requires careful feature alignment when deploying model
- **Mitigation**: Model artifacts include feature list for validation

### 2. **Datetime Column Handling**
- **Issue**: Datetime columns must be explicitly dropped before model training/prediction
- **Impact**: Error if datetime columns present in input data
- **Mitigation**: Preprocessing code includes datetime column removal

### 3. **Routing Inconsistencies**
- **Issue**: Keyword router and LLM router may route same question differently
- **Impact**: API and chatbot may give different results for same question
- **Note**: LLM router generally performs better -->

## Future Improvements

1. **Replace keyword router with LLM router** in API for consistent routing
2. **Add temporal dimension** to knowledge graph for time-based queries
3. **Improve LLM prompting** for more consistent answer formatting
4. **Add more ground truth** for comprehensive evaluation
5. **Add rate limiting** and error handling improvements
6. **Expand test suite** with more diverse questions
7. **Cloud deployment** support with alternative LLM providers

