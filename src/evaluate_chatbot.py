"""
Evaluation script for the chatbot (LLM-generated answers).

This script:
1. Loads questions from evaluation_questions.json
2. Uses the chatbot (chat_once) to get LLM-generated answers
3. Compares chatbot answers against ground truth (expected_value, expected_answer)
4. Reports accuracy metrics

This evaluates the FULL chatbot pipeline:
- LLM routing
- API calls
- LLM answer generation

For API-only evaluation, see evaluate_routing.py
For API ground truth comparison, see evaluate_with_ground_truth.py
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from chat_ollama import chat_once


BASE_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = BASE_DIR / "evaluation" / "evaluation_questions.json"
OUTPUT_FILE = BASE_DIR / "evaluation" / "chatbot_evaluation.json"


def load_questions() -> List[Dict[str, Any]]:
    """Load evaluation questions from JSON."""
    with open(QUESTIONS_FILE) as f:
        data = json.load(f)
    return data["questions"]


def extract_number_from_text(text: str, question: Optional[str] = None) -> Optional[float]:
    """
    Extract the relevant numeric value from text (for single-value questions).
    
    If question is provided, tries to extract the number associated with the question's key term.
    For example, if question asks about "grade B", extracts the number near "B" or "grade B".
    """
    # If question mentions a specific key (e.g., "grade B", "grade C"), try to find number near that key
    if question:
        question_lower = question.lower()
        
        # Extract key terms from question (e.g., "grade B", "grade C", "36 months", "60 months")
        grade_match = re.search(r"grade\s+([A-G])", question_lower)
        term_match = re.search(r"(\d+)\s*months?", question_lower)
        
        if grade_match:
            target_grade = grade_match.group(1).upper()
            # Find the number associated with this grade in the answer
            # Look for patterns like "B: defaultRate=0.151" or "grade B: 0.151" or "B: 15.1%"
            grade_patterns = [
                rf"{target_grade}.*?defaultRate[=:]\s*(\d+\.\d+)",
                rf"{target_grade}.*?(\d+\.\d+)",
                rf"grade\s+{target_grade}.*?(\d+\.\d+)",
                rf"{target_grade}.*?(\d+\.?\d*)\s*%",
                rf"grade\s+{target_grade}.*?(\d+\.?\d*)\s*%",
            ]
            for pattern in grade_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    num_str = match.group(1)
                    if "%" in pattern:
                        return float(num_str) / 100.0
                    return float(num_str)
        
        if term_match:
            target_term = term_match.group(1)
            # Find the number associated with this term
            term_patterns = [
                rf"{target_term}\s*months?.*?defaultRate[=:]\s*(\d+\.\d+)",
                rf"{target_term}\s*months?.*?(\d+\.\d+)",
            ]
            for pattern in term_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return float(match.group(1))
    
    # Fallback: Try to find percentages first
    percent_match = re.search(r"(\d+\.?\d*)\s*%", text)
    if percent_match:
        return float(percent_match.group(1)) / 100.0
    
    # Try to find decimals
    decimal_match = re.search(r"(\d+\.\d+)", text)
    if decimal_match:
        return float(decimal_match.group(1))
    
    # Try to find integers
    int_match = re.search(r"\b(\d+)\b", text)
    if int_match:
        return float(int_match.group(1))
    
    return None


def parse_expected_value(expected_value_str: str) -> Optional[float]:
    """Parse expected_value string like '~0.151 (15.1%)' to get numeric value."""
    # Extract number from string
    # Try to find decimal first
    decimal_match = re.search(r"(\d+\.\d+)", expected_value_str)
    if decimal_match:
        return float(decimal_match.group(1))
    
    # Try percentage
    percent_match = re.search(r"(\d+\.?\d*)\s*%", expected_value_str)
    if percent_match:
        return float(percent_match.group(1)) / 100.0
    
    return None


def compare_numeric_answer(
    chatbot_answer: str,
    expected_value: str,
    tolerance: Optional[float] = None,
    question: Optional[str] = None
) -> Tuple[bool, str, Optional[float], Optional[float]]:
    """
    Compare chatbot answer against expected numeric value.
    
    Returns:
        (is_match, explanation, actual_num, expected_num)
    """
    actual_num = extract_number_from_text(chatbot_answer, question)
    expected_num = parse_expected_value(expected_value)
    
    if actual_num is None:
        return (False, "No number found in chatbot answer", None, expected_num)
    
    if expected_num is None:
        return (False, "Could not parse expected value", actual_num, None)
    
    if tolerance is None:
        tolerance = 0.02  # Default 2% tolerance
    
    diff = abs(actual_num - expected_num)
    is_match = diff <= tolerance
    
    explanation = (
        f"Actual: {actual_num:.4f}, Expected: {expected_num:.4f}, "
        f"Diff: {diff:.4f}, Tolerance: {tolerance:.4f}"
    )
    
    return (is_match, explanation, actual_num, expected_num)


def compare_text_answer(
    chatbot_answer: str,
    expected_answer: str
) -> Tuple[bool, str]:
    """
    Compare chatbot answer text against expected answer.
    
    Returns:
        (is_match, explanation)
    """
    actual_lower = chatbot_answer.lower()
    expected_lower = expected_answer.lower()
    
    # First, check if expected answer contains specific numeric values that must be present
    # Extract percentages and decimals from expected answer
    expected_numbers = []
    # Find percentages like "12.64%" or "~12.64%"
    for match in re.finditer(r"~?(\d+\.\d+)\s*%", expected_lower):
        expected_numbers.append(float(match.group(1)))
    # Find decimals like "0.355" or "~0.355"
    for match in re.finditer(r"~?(\d+\.\d+)", expected_lower):
        num = float(match.group(1))
        if num not in expected_numbers:  # Avoid duplicates
            expected_numbers.append(num)
    
    # If expected answer has specific numbers, verify they appear in chatbot answer
    if expected_numbers:
        actual_numbers = extract_all_numbers_from_text(chatbot_answer)
        # Check if expected numbers are present (within tolerance)
        numbers_found = 0
        for exp_num in expected_numbers:
            for act_num in actual_numbers:
                # For percentages, compare directly
                if abs(act_num - exp_num) < 0.5:  # 0.5% tolerance
                    numbers_found += 1
                    break
                # For decimals, compare with smaller tolerance
                if abs(act_num - exp_num) < 0.01:
                    numbers_found += 1
                    break
        
        # If we have specific numbers, they must be present
        if numbers_found < len(expected_numbers):
            return (False, f"Expected numbers not found. Expected: {expected_numbers}, Found in answer: {actual_numbers[:5]}")
    
    # Check for range requirements (e.g., "rates around 15.5-16.8%")
    range_match = re.search(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", expected_lower)
    if range_match:
        min_val = float(range_match.group(1))
        max_val = float(range_match.group(2))
        # Extract all numbers from chatbot answer
        actual_numbers = extract_all_numbers_from_text(chatbot_answer)
        # Check if any number is in the expected range
        numbers_in_range = [n for n in actual_numbers if min_val <= n <= max_val]
        if not numbers_in_range:
            return (False, f"No numbers found in expected range [{min_val}-{max_val}]. Found: {actual_numbers[:5]}")
    
    # Extract key phrases from expected answer
    key_phrases = []
    
    # Extract numbers and comparisons
    if "higher" in expected_lower or "greater" in expected_lower:
        key_phrases.append("higher")
    if "lower" in expected_lower or "less" in expected_lower:
        key_phrases.append("lower")
    
    # Extract the main entities (e.g., "60 months", "36 months")
    numbers = re.findall(r"\d+", expected_lower)
    if numbers:
        key_phrases.extend(numbers)
    
    # Extract key terms (e.g., "default rate", "interest rate")
    if "default rate" in expected_lower:
        key_phrases.append("default rate")
    if "interest rate" in expected_lower:
        key_phrases.append("interest rate")
    
    # Extract specific entities mentioned (e.g., "small_business", "debt_consolidation")
    # These are critical - if wrong, the answer is wrong
    critical_entities = []
    common_words = [
        "rate", "default", "interest", "loan", "highest", "lowest", "higher", "lower",
        "month", "months", "term", "terms", "has", "have", "should", "typically",
        "around", "versus", "compared", "compare", "vs", "versus"
    ]
    
    entity_patterns = [
        r"([a-z_]+)",  # underscore-separated entities
        r'"([^"]+)"',  # quoted entities
    ]
    for pattern in entity_patterns:
        entities = re.findall(pattern, expected_lower)
        for entity in entities:
            if len(entity) > 3:  # Avoid short words
                # Check if it's a specific entity (not a common word)
                if entity not in common_words:
                    critical_entities.append(entity)
                    if entity not in key_phrases:
                        key_phrases.append(entity)
    
    # For number-based comparisons (e.g., "60 months" vs "36 months"), 
    # prioritize checking if the numbers are present rather than exact phrase matching
    if numbers and len(numbers) >= 2:
        actual_numbers = re.findall(r"\d+", actual_lower)
        numbers_found = sum(1 for num in numbers if num in actual_numbers)
        has_comparison = any(term in actual_lower for term in ["higher", "lower", "greater", "less"])
        
        # If both numbers are present and comparison term is present, this is likely correct
        # Skip strict entity checking for number-based comparisons
        if numbers_found >= 2 and has_comparison:
            # This is a valid comparison - don't fail on entity matching
            pass
        elif critical_entities:
            # Check entities normally with flexible matching (singular/plural)
            critical_found = sum(1 for entity in critical_entities 
                                if entity in actual_lower 
                                or entity.rstrip('s') in actual_lower
                                or actual_lower.replace('-', ' ').replace('_', ' ').find(entity) != -1)
            if critical_found < len(critical_entities):
                return (False, f"Critical entity not found. Expected: {critical_entities}, Found in answer: {[e for e in critical_entities if (e in actual_lower or e.rstrip('s') in actual_lower)]}")
    elif critical_entities:
        # Normal entity check with flexible matching (singular/plural, hyphens, underscores)
        critical_found = sum(1 for entity in critical_entities 
                            if entity in actual_lower 
                            or entity.rstrip('s') in actual_lower
                            or actual_lower.replace('-', ' ').replace('_', ' ').find(entity) != -1)
        if critical_found < len(critical_entities):
            return (False, f"Critical entity not found. Expected: {critical_entities}, Found in answer: {[e for e in critical_entities if (e in actual_lower or e.rstrip('s') in actual_lower)]}")
    
    # Check if chatbot answer contains key phrases
    matches = sum(1 for phrase in key_phrases if phrase in actual_lower)
    match_ratio = matches / len(key_phrases) if key_phrases else 0.0
    
    # Consider it a match if >50% of key phrases are present
    # OR if the key comparison term (higher/lower) and at least one number are present
    has_comparison = any(term in actual_lower for term in ["higher", "lower", "greater", "less"])
    has_number = any(num in actual_lower for num in numbers) if numbers else False
    
    is_match = match_ratio >= 0.5 or (has_comparison and has_number)
    
    explanation = (
        f"Key phrases found: {matches}/{len(key_phrases)} "
        f"({match_ratio*100:.1f}%), "
        f"Has comparison: {has_comparison}, Has number: {has_number}"
    )
    
    return (is_match, explanation)


def extract_all_numbers_from_text(text: str) -> List[float]:
    """Extract all numeric values from text (percentages and decimals)."""
    numbers = []
    # Find percentages first (convert to decimal)
    for match in re.finditer(r"(\d+\.?\d*)\s*%", text):
        numbers.append(float(match.group(1)))
    # Find decimals (but not percentages we already found)
    for match in re.finditer(r"(\d+\.\d+)", text):
        num = float(match.group(1))
        # Avoid duplicates and very small numbers that are likely not the values we want
        if num not in numbers and num > 0.01:
            numbers.append(num)
    return numbers


def evaluate_against_ground_truth(
    question: Dict[str, Any],
    chatbot_answer: str
) -> Dict[str, Any]:
    """
    Evaluate chatbot answer against ground truth.
    
    Returns evaluation result with:
    - matches_ground_truth: bool
    - comparison_type: "numeric" | "text" | "none"
    - explanation: str
    - actual_value: extracted value
    - expected_value: expected value
    """
    ground_truth = question.get("ground_truth", {})
    
    result = {
        "matches_ground_truth": False,
        "comparison_type": "none",
        "explanation": "",
        "actual_value": None,
        "expected_value": None,
        "tolerance_used": None,
    }
    
    # Check for expected_value (numeric comparison)
    if "expected_value" in ground_truth:
        expected_value = ground_truth["expected_value"]
        tolerance = ground_truth.get("tolerance", 0.02)
        
        is_match, explanation, actual_num, expected_num = compare_numeric_answer(
            chatbot_answer, expected_value, tolerance, question.get("question", "")
        )
        
        result["matches_ground_truth"] = is_match
        result["comparison_type"] = "numeric"
        result["explanation"] = explanation
        result["actual_value"] = actual_num
        result["expected_value"] = expected_num
        result["tolerance_used"] = tolerance
        
        return result
    
    # Check for expected_answer (text comparison)
    if "expected_answer" in ground_truth:
        expected_answer = ground_truth["expected_answer"]
        
        is_match, explanation = compare_text_answer(chatbot_answer, expected_answer)
        
        result["matches_ground_truth"] = is_match
        result["comparison_type"] = "text"
        result["explanation"] = explanation
        result["actual_value"] = chatbot_answer[:200]  # First 200 chars
        result["expected_value"] = expected_answer
        
        return result
    
    # Check for expected_behavior (for unanswerable questions)
    if "expected_behavior" in ground_truth:
        expected_behavior = ground_truth["expected_behavior"].lower()
        actual_lower = chatbot_answer.lower()
        
        # Check if answer indicates unanswerable
        unanswerable_keywords = [
            "not available", "not in", "cannot", "unable", "don't have",
            "no information", "not provided", "i can currently answer",
            "can currently answer", "currently answer questions about"
        ]
        
        has_keyword = any(keyword in actual_lower for keyword in unanswerable_keywords)
        
        # Also check if answer is the generic "I can currently answer..." message
        is_generic_message = "i can currently answer questions about aggregate statistics" in actual_lower
        has_keyword = has_keyword or is_generic_message
        
        result["matches_ground_truth"] = has_keyword
        result["comparison_type"] = "behavior"
        result["explanation"] = (
            f"Expected: {expected_behavior}. "
            f"Found unanswerable keywords: {has_keyword}"
        )
        result["actual_value"] = chatbot_answer[:200]
        result["expected_value"] = expected_behavior
        
        return result
    
    # Check for key_metrics 
    if "key_metrics" in ground_truth:
        key_metrics = ground_truth["key_metrics"]
        actual_lower = chatbot_answer.lower()
        question_lower = question.get("question", "").lower()
        
        # Determine which metrics are actually relevant to the question
        # If question asks about "default rate", only check defaultRate
        # If question asks about "interest rate", only check avgInterestRate
        # If question asks about "loan amount", only check avgLoanAmount
        relevant_metrics = key_metrics.copy()
        if "default rate" in question_lower:
            relevant_metrics = [m for m in key_metrics if "default" in m.lower() or "defaultrate" in m.lower()]
        elif "interest rate" in question_lower:
            relevant_metrics = [m for m in key_metrics if "interest" in m.lower() or "interestrate" in m.lower()]
        elif "loan amount" in question_lower or "loan amount" in question_lower:
            relevant_metrics = [m for m in key_metrics if "amount" in m.lower() or "loanamount" in m.lower()]
        elif "loan count" in question_lower or "number of loans" in question_lower:
            relevant_metrics = [m for m in key_metrics if "count" in m.lower() or "loancount" in m.lower()]
        
        # If no specific metric found in question, use all metrics but be more lenient
        if not relevant_metrics:
            relevant_metrics = key_metrics
        
        # Check if answer mentions the relevant metrics
        metrics_found = []
        for metric in relevant_metrics:
            # Convert camelCase to readable format
            metric_readable = metric.replace("avg", "average ").replace("Rate", " rate").replace("Amount", " amount").replace("Count", " count")
            if metric.lower() in actual_lower or metric_readable.lower() in actual_lower:
                metrics_found.append(metric)
        
        # Also check for numbers (answers should have numeric data)
        has_numbers = bool(re.search(r"\d+\.?\d*", chatbot_answer))
        
        # If expected_answer is also provided, use it for stricter checking
        if "expected_answer" in ground_truth:
            expected_answer = ground_truth["expected_answer"]
            is_text_match, text_explanation = compare_text_answer(chatbot_answer, expected_answer)
            # Combine key_metrics check with text comparison
            # Both must pass: relevant metrics mentioned AND text matches (including numbers)
            is_match = (len(metrics_found) > 0 and has_numbers) and is_text_match
            result["matches_ground_truth"] = is_match
            result["comparison_type"] = "key_metrics_with_answer"
            result["explanation"] = (
                f"Relevant metrics found: {len(metrics_found)}/{len(relevant_metrics)} "
                f"(from {len(key_metrics)} total), Has numbers: {has_numbers}, Text match: {is_text_match}. {text_explanation}"
            )
        else:
            # Consider it a match if it mentions at least one relevant metric and has numbers
            # For questions asking about a specific metric, require that metric
            # For general questions, be more lenient
            is_match = len(metrics_found) > 0 and has_numbers
            match_ratio = len(metrics_found) / len(relevant_metrics) if relevant_metrics else 0.0
            result["matches_ground_truth"] = is_match
            result["comparison_type"] = "key_metrics"
            result["explanation"] = (
                f"Relevant metrics found: {len(metrics_found)}/{len(relevant_metrics)} "
                f"(from {len(key_metrics)} total, {match_ratio*100:.1f}%), Has numbers: {has_numbers}"
            )
        
        result["actual_value"] = chatbot_answer[:200]
        result["expected_value"] = ground_truth.get("expected_answer", f"Should mention: {', '.join(key_metrics)}")
        
        return result
    
    # No ground truth to compare against
    result["explanation"] = "No expected_value, expected_answer, or key_metrics in ground_truth"
    return result


def evaluate_question(question: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single question using the chatbot."""
    print(f"\n{'='*60}")
    print(f"Question {question['id']}: {question['question']}")
    
    result = {
        "question_id": question["id"],
        "question": question["question"],
        "question_type": question.get("type", ""),
        "expected_backend": question.get("expected_backend", "unknown"),
        "ground_truth_comparison": {},
        "chatbot_answer": "",
        "errors": [],
        "skipped": False,
    }
    
    # Check if question has ground truth to compare against
    ground_truth = question.get("ground_truth", {})
    has_ground_truth = (
        "expected_value" in ground_truth
        or "expected_answer" in ground_truth
        or "expected_behavior" in ground_truth
        or "key_metrics" in ground_truth
    )
    
    if not has_ground_truth:
        result["skipped"] = True
        result["ground_truth_comparison"] = {
            "comparison_type": "none",
            "explanation": "No ground truth available - question will be answered by LLM without verifiable metrics",
        }
        print(f"    SKIPPED: No ground truth available for evaluation")
        print(f"  Note: This question will be answered by LLM but cannot be automatically evaluated")
        return result
    
    try:
        # Get chatbot answer (this uses LLM routing + LLM answer generation)
        # Debug: Check what route was chosen and what API returned
        from chat_ollama import route_question_with_llm, call_api_ask
        route = route_question_with_llm(question["question"])
        print(f"  Route: {route}")
        
        if route == "cohort":
            # Check API response directly
            api_result = call_api_ask(question["question"])
            print(f"  API Results Count: {len(api_result.get('results', []))}")
            if api_result.get('results'):
                print(f"  First result keys: {list(api_result['results'][0].keys())}")
        
        chatbot_answer = chat_once(question["question"])
        result["chatbot_answer"] = chatbot_answer
        
        print(f"  Chatbot Answer: {chatbot_answer[:300]}...")
        print(f"  Answer Length: {len(chatbot_answer)} characters")
        
        # Compare against ground truth
        gt_comparison = evaluate_against_ground_truth(question, chatbot_answer)
        result["ground_truth_comparison"] = gt_comparison
        
        # Print results
        if gt_comparison["comparison_type"] != "none":
            status = " MATCH" if gt_comparison["matches_ground_truth"] else " NO MATCH"
            print(f"  Ground Truth: {status}")
            print(f"  Type: {gt_comparison['comparison_type']}")
            print(f"  {gt_comparison['explanation']}")
            if gt_comparison.get("actual_value") is not None:
                if isinstance(gt_comparison["actual_value"], float):
                    print(f"  Actual: {gt_comparison['actual_value']:.4f}")
                else:
                    print(f"  Actual: {str(gt_comparison['actual_value'])[:100]}")
            if gt_comparison.get("expected_value") is not None:
                if isinstance(gt_comparison["expected_value"], float):
                    print(f"  Expected: {gt_comparison['expected_value']:.4f}")
                else:
                    print(f"  Expected: {str(gt_comparison['expected_value'])[:100]}")
        else:
            print(f"  No ground truth comparison available")
        
    except Exception as e:
        result["errors"].append(str(e))
        print(f"  ERROR: {e}")
    
    return result


def main():
    """Run evaluation on all questions."""
    print("="*60)
    print("CHATBOT EVALUATION (LLM-Generated Answers)")
    print("="*60)
    print(f"Questions File: {QUESTIONS_FILE}")
    print("\nNote: This uses the full chatbot pipeline (LLM routing + LLM answer generation)")
    print("Make sure Ollama is running: ollama serve")
    
    questions = load_questions()
    print(f"\nLoaded {len(questions)} evaluation questions")
    
    results = []
    for q in questions:
        result = evaluate_question(q)
        results.append(result)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Separate skipped questions from evaluated ones
    skipped_questions = [r for r in results if r.get("skipped", False)]
    evaluated_questions = [r for r in results if not r.get("skipped", False)]
    
    print(f"\nTotal Questions: {len(results)}")
    print(f"Evaluated: {len(evaluated_questions)}")
    print(f"Skipped (no ground truth): {len(skipped_questions)}")
    
    if skipped_questions:
        print(f"\n--- Skipped Questions ---")
        for r in skipped_questions:
            print(f"  {r['question_id']}: {r['question'][:60]}...")
    
    # Count comparisons by type (only for evaluated questions)
    numeric_comparisons = [r for r in evaluated_questions if r["ground_truth_comparison"].get("comparison_type") == "numeric"]
    text_comparisons = [r for r in evaluated_questions if r["ground_truth_comparison"].get("comparison_type") == "text"]
    behavior_comparisons = [r for r in evaluated_questions if r["ground_truth_comparison"].get("comparison_type") == "behavior"]
    key_metrics_comparisons = [r for r in evaluated_questions if r["ground_truth_comparison"].get("comparison_type") == "key_metrics"]
    key_metrics_with_answer_comparisons = [r for r in evaluated_questions if r["ground_truth_comparison"].get("comparison_type") == "key_metrics_with_answer"]
    
    # Count matches
    numeric_matches = sum(1 for r in numeric_comparisons if r["ground_truth_comparison"].get("matches_ground_truth", False))
    text_matches = sum(1 for r in text_comparisons if r["ground_truth_comparison"].get("matches_ground_truth", False))
    behavior_matches = sum(1 for r in behavior_comparisons if r["ground_truth_comparison"].get("matches_ground_truth", False))
    key_metrics_matches = sum(1 for r in key_metrics_comparisons if r["ground_truth_comparison"].get("matches_ground_truth", False))
    key_metrics_with_answer_matches = sum(1 for r in key_metrics_with_answer_comparisons if r["ground_truth_comparison"].get("matches_ground_truth", False))
    
    total_comparisons = len(numeric_comparisons) + len(text_comparisons) + len(behavior_comparisons) + len(key_metrics_comparisons) + len(key_metrics_with_answer_comparisons)
    total_matches = numeric_matches + text_matches + behavior_matches + key_metrics_matches + key_metrics_with_answer_matches
    
    print(f"\n--- Ground Truth Accuracy (Chatbot Answers) ---")
    print(f"Numeric Comparisons: {numeric_matches}/{len(numeric_comparisons)} ({numeric_matches/len(numeric_comparisons)*100:.1f}%)" if numeric_comparisons else "\nNumeric Comparisons: 0")
    print(f"Text Comparisons: {text_matches}/{len(text_comparisons)} ({text_matches/len(text_comparisons)*100:.1f}%)" if text_comparisons else "Text Comparisons: 0")
    print(f"Behavior Comparisons: {behavior_matches}/{len(behavior_comparisons)} ({behavior_matches/len(behavior_comparisons)*100:.1f}%)" if behavior_comparisons else "Behavior Comparisons: 0")
    print(f"Key Metrics Comparisons: {key_metrics_matches}/{len(key_metrics_comparisons)} ({key_metrics_matches/len(key_metrics_comparisons)*100:.1f}%)" if key_metrics_comparisons else "Key Metrics Comparisons: 0")
    print(f"Key Metrics with Answer Comparisons: {key_metrics_with_answer_matches}/{len(key_metrics_with_answer_comparisons)} ({key_metrics_with_answer_matches/len(key_metrics_with_answer_comparisons)*100:.1f}%)" if key_metrics_with_answer_comparisons else "Key Metrics with Answer Comparisons: 0")
    print(f"\nOverall Chatbot Accuracy: {total_matches}/{total_comparisons} ({total_matches/total_comparisons*100:.1f}%)" if total_comparisons > 0 else "\nOverall Chatbot Accuracy: N/A (no comparisons)")
    
    # Show questions with errors
    error_questions = [r for r in results if r.get("errors")]
    if error_questions:
        print(f"\n--- Questions with Errors ({len(error_questions)}) ---")
        for r in error_questions:
            print(f"  {r['question_id']}: {r['question'][:60]}...")
            for error in r["errors"]:
                print(f"    Error: {error}")
    
    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

