"""
Part 3: Evaluation script for the loan intelligence system.

Measures:
1. Retrieval quality (for KG queries)
2. Answer generation quality (LLM responses)
3. End-to-end system performance
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from rdflib import Graph

BASE_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = BASE_DIR / "evaluation" / "evaluation_questions.json"
KG_TTL_PATH = BASE_DIR / "artifacts" / "loan_cohorts.ttl"
API_BASE = "http://127.0.0.1:8000"
OLLAMA_BASE = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3"


def load_questions() -> List[Dict[str, Any]]:
    """Load evaluation questions from JSON."""
    with open(QUESTIONS_FILE) as f:
        data = json.load(f)
    return data["questions"]


def get_ground_truth_from_kg(dimension: str) -> Dict[str, Any]:
    """Query KG directly to get ground truth for cohort questions."""
    g = Graph()
    g.parse(str(KG_TTL_PATH), format="turtle")
    prefix = "http://example.org/loan#"

    query = f"""
    PREFIX ex: <{prefix}>
    SELECT ?key ?loanCount ?defaultRate ?avgInterestRate ?avgLoanAmount
    WHERE {{
      ?cohort a ex:Cohort ;
              ex:cohortType "{dimension}" ;
              ex:loanCount ?loanCount ;
              ex:avgInterestRate ?avgInterestRate ;
              ex:avgLoanAmount ?avgLoanAmount .
      OPTIONAL {{ ?cohort ex:defaultRate ?defaultRate . }}
      OPTIONAL {{
        {'?cohort ex:grade ?key .' if dimension == 'grade' else ''}
        {'?cohort ex:term ?key .' if dimension == 'term' else ''}
        {'?cohort ex:purpose ?key .' if dimension == 'purpose' else ''}
        {'?cohort ex:loanStatus ?key .' if dimension == 'status' else ''}
        {'?cohort ex:homeOwnership ?key .' if dimension == 'home_ownership' else ''}
        {'?cohort ex:incomeBand ?key .' if dimension == 'income_band' else ''}
        {'?cohort ex:state ?key .' if dimension == 'state' else ''}
      }}
    }}
    ORDER BY ?key
    """

    results = {}
    for row in g.query(query):
        key, loan_count, default_rate, avg_ir, avg_amt = row
        key_str = str(key) if key else "unknown"
        results[key_str] = {
            "loanCount": int(loan_count),
            "defaultRate": float(default_rate) if default_rate is not None else None,
            "avgInterestRate": float(avg_ir),
            "avgLoanAmount": float(avg_amt),
        }
    return results


def call_api_ask(question: str) -> Dict[str, Any]:
    """Call /ask endpoint."""
    resp = requests.post(f"{API_BASE}/ask", json={"question": question}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_numbers_from_text(text: str) -> List[float]:
    """Extract numeric values from text (for answer quality checking)."""
    # Find percentages and decimals
    patterns = [
        r"(\d+\.?\d*)\s*%",  # percentages
        r"(\d+\.\d+)",  # decimals
        r"(\d+)",  # integers
    ]
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        numbers.extend([float(m) for m in matches])
    return numbers


def evaluate_retrieval_quality(
    question: Dict[str, Any], api_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate if the system retrieved correct data from KG."""
    if question["expected_backend"] != "cohort":
        return {"score": None, "note": "Not a KG query"}

    dimension = question.get("expected_dimension")
    if not dimension:
        return {"score": None, "note": "No expected dimension"}

    # Get ground truth from KG
    gt = get_ground_truth_from_kg(dimension)
    api_results = api_result.get("results", [])

    # Check if key values match
    matches = 0
    total = 0
    errors = []

    for api_row in api_results:
        key = api_row.get("key")
        if key and key in gt:
            total += 1
            api_dr = api_row.get("defaultRate")
            gt_dr = gt[key].get("defaultRate")
            if api_dr is not None and gt_dr is not None:
                if abs(api_dr - gt_dr) < 0.01:  # Within 1%
                    matches += 1
                else:
                    errors.append(f"{key}: API={api_dr:.3f}, GT={gt_dr:.3f}")

    score = matches / total if total > 0 else 0.0
    return {
        "score": score,
        "matches": matches,
        "total": total,
        "errors": errors,
    }


def evaluate_answer_quality(
    question: Dict[str, Any], api_result: Dict[str, Any], llm_answer: Optional[str]
) -> Dict[str, Any]:
    """Evaluate LLM-generated answer quality."""
    if not llm_answer:
        return {"score": None, "note": "No LLM answer provided"}

    # Simple heuristics:
    # 1. Answer contains numbers (for numeric questions)
    # 2. Answer is coherent (not empty, reasonable length)
    # 3. Answer addresses the question type

    numbers = extract_numbers_from_text(llm_answer)
    has_numbers = len(numbers) > 0
    is_coherent = len(llm_answer.strip()) > 20

    # Check if answer type matches question type
    q_type = question.get("type", "")
    type_match = True
    if q_type == "comparative" and "compare" not in llm_answer.lower():
        type_match = False
    if q_type == "aggregation" and len(numbers) < 2:
        type_match = False

    score = 0.0
    if has_numbers:
        score += 0.4
    if is_coherent:
        score += 0.4
    if type_match:
        score += 0.2

    return {
        "score": score,
        "has_numbers": has_numbers,
        "is_coherent": is_coherent,
        "type_match": type_match,
        "numbers_found": numbers[:5],  # First 5 numbers
    }


def evaluate_routing(question: Dict[str, Any], api_result: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate if question was routed to correct backend."""
    expected = question.get("expected_backend", "unknown")
    actual = api_result.get("interpretation", {}).get("type", "unknown")

    # Map API types to expected
    type_mapping = {
        "cohort": "cohort",
        "predict": "predict",
        "unknown": "unknown",
    }

    actual_mapped = type_mapping.get(actual, "unknown")
    is_correct = actual_mapped == expected or (
        expected == "cohort" and actual_mapped == "cohort"
    )

    return {
        "correct": is_correct,
        "expected": expected,
        "actual": actual_mapped,
    }


def evaluate_end_to_end(question: Dict[str, Any]) -> Dict[str, Any]:
    """Run full evaluation for a single question."""
    print(f"\n{'='*60}")
    print(f"Question {question['id']}: {question['question']}")
    print(f"Type: {question['type']}, Expected backend: {question['expected_backend']}")

    result = {
        "question_id": question["id"],
        "question": question["question"],
        "routing": {},
        "retrieval": {},
        "answer_quality": {},
        "errors": [],
    }

    try:
        # Call API
        api_result = call_api_ask(question["question"])

        # Evaluate routing
        result["routing"] = evaluate_routing(question, api_result)
        
        # Print actual routing result
        actual = result["routing"].get("actual", "unknown")
        correct = result["routing"].get("correct", False)
        status = "" if correct else ""
        print(f"  Expected: {question['expected_backend']}, Actual: {actual} {status}")

        # Evaluate retrieval (for KG queries)
        if question["expected_backend"] == "cohort":
            result["retrieval"] = evaluate_retrieval_quality(question, api_result)

        # Evaluate answer quality (if LLM answer available)
        llm_answer = api_result.get("answer_text", "")
        result["answer_quality"] = evaluate_answer_quality(question, api_result, llm_answer)

    except Exception as e:
        result["errors"].append(str(e))
        print(f"  ERROR: {e}")

    return result


def main():
    """Run evaluation on all questions."""
    questions = load_questions()
    print(f"Loaded {len(questions)} evaluation questions")

    results = []
    for q in questions:
        result = evaluate_end_to_end(q)
        results.append(result)

    # Summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    # Routing accuracy
    routing_correct = sum(1 for r in results if r["routing"].get("correct", False))
    routing_total = len([r for r in results if r["routing"]])
    print(f"\nRouting Accuracy: {routing_correct}/{routing_total} ({routing_correct/routing_total*100:.1f}%)")

    # Save detailed results
    output_file = BASE_DIR / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()

