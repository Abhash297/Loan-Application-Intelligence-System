"""
Utility script to sanity-check the cohort knowledge graph with SPARQL queries.

Run:
    python test_sparql.py
"""

from __future__ import annotations

from pathlib import Path

from rdflib import Graph


BASE_DIR = Path(__file__).resolve().parent
TTL_PATH = BASE_DIR / "loan_cohorts.ttl"
EX_NS = "http://example.org/loan#"


def load_graph() -> Graph:
    if not TTL_PATH.exists():
        raise FileNotFoundError(f"Turtle file not found at {TTL_PATH}")
    g = Graph()
    g.parse(str(TTL_PATH), format="turtle")
    print(f"Loaded graph with {len(g)} triples from {TTL_PATH}")
    return g


def query_by_grade(g: Graph) -> None:
    print("\n=== Default rate by grade ===")
    q = f"""
    PREFIX ex: <{EX_NS}>
    SELECT ?grade ?loanCount ?defaultRate ?avgInterestRate
    WHERE {{
      ?cohort a ex:Cohort ;
              ex:cohortType "grade" ;
              ex:grade ?grade ;
              ex:loanCount ?loanCount ;
              ex:avgInterestRate ?avgInterestRate .
      OPTIONAL {{ ?cohort ex:defaultRate ?defaultRate . }}
    }}
    ORDER BY ?grade
    """
    for row in g.query(q):
        grade, loan_count, default_rate, avg_ir = row
        print(
            f"grade={grade}, loans={int(loan_count)}, "
            f"defaultRate={float(default_rate):.3f}, avgIR={float(avg_ir):.2f}"
        )


def query_by_term(g: Graph) -> None:
    print("\n=== Default rate by term ===")
    q = f"""
    PREFIX ex: <{EX_NS}>
    SELECT ?term ?loanCount ?defaultRate ?avgInterestRate
    WHERE {{
      ?cohort a ex:Cohort ;
              ex:cohortType "term" ;
              ex:term ?term ;
              ex:loanCount ?loanCount ;
              ex:avgInterestRate ?avgInterestRate .
      OPTIONAL {{ ?cohort ex:defaultRate ?defaultRate . }}
    }}
    ORDER BY ?term
    """
    for row in g.query(q):
        term, loan_count, default_rate, avg_ir = row
        print(
            f"term={term}, loans={int(loan_count)}, "
            f"defaultRate={float(default_rate):.3f}, avgIR={float(avg_ir):.2f}"
        )


def query_by_purpose(g: Graph) -> None:
    print("\n=== Default rate by purpose (top 10) ===")
    q = f"""
    PREFIX ex: <{EX_NS}>
    SELECT ?purpose ?loanCount ?defaultRate ?avgInterestRate
    WHERE {{
      ?cohort a ex:Cohort ;
              ex:cohortType "purpose" ;
              ex:purpose ?purpose ;
              ex:loanCount ?loanCount ;
              ex:avgInterestRate ?avgInterestRate .
      OPTIONAL {{ ?cohort ex:defaultRate ?defaultRate . }}
    }}
    ORDER BY DESC(?loanCount)
    """
    for i, row in enumerate(g.query(q)):
        if i >= 10:
            break
        purpose, loan_count, default_rate, avg_ir = row
        print(
            f"purpose={purpose}, loans={int(loan_count)}, "
            f"defaultRate={float(default_rate):.3f}, avgIR={float(avg_ir):.2f}"
        )


def query_by_home_ownership(g: Graph) -> None:
    print("\n=== Default rate by home ownership ===")
    q = f"""
    PREFIX ex: <{EX_NS}>
    SELECT ?homeOwnership ?loanCount ?defaultRate ?avgIncome
    WHERE {{
      ?cohort a ex:Cohort ;
              ex:cohortType "home_ownership" ;
              ex:homeOwnership ?homeOwnership ;
              ex:loanCount ?loanCount .
      OPTIONAL {{ ?cohort ex:defaultRate ?defaultRate . }}
      OPTIONAL {{ ?cohort ex:avgIncome ?avgIncome . }}
    }}
    ORDER BY ?homeOwnership
    """
    for row in g.query(q):
        home_own, loan_count, default_rate, avg_inc = row
        print(
            f"homeOwnership={home_own}, loans={int(loan_count)}, "
            f"defaultRate={float(default_rate):.3f}, avgIncome={float(avg_inc):.2f}"
        )


def query_by_income_band(g: Graph) -> None:
    print("\n=== Default rate by income band ===")
    q = f"""
    PREFIX ex: <{EX_NS}>
    SELECT ?incomeBand ?loanCount ?defaultRate ?avgLoanAmount
    WHERE {{
      ?cohort a ex:Cohort ;
              ex:cohortType "income_band" ;
              ex:incomeBand ?incomeBand ;
              ex:loanCount ?loanCount ;
              ex:avgLoanAmount ?avgLoanAmount .
      OPTIONAL {{ ?cohort ex:defaultRate ?defaultRate . }}
    }}
    ORDER BY ?incomeBand
    """
    for row in g.query(q):
        band, loan_count, default_rate, avg_amt = row
        print(
            f"incomeBand={band}, loans={int(loan_count)}, "
            f"defaultRate={float(default_rate):.3f}, avgLoanAmount={float(avg_amt):.2f}"
        )


def query_by_state(g: Graph) -> None:
    print("\n=== Average interest rate by state (top 10 by loan count) ===")
    q = f"""
    PREFIX ex: <{EX_NS}>
    SELECT ?state ?loanCount ?avgInterestRate
    WHERE {{
      ?cohort a ex:Cohort ;
              ex:cohortType "state" ;
              ex:state ?state ;
              ex:loanCount ?loanCount ;
              ex:avgInterestRate ?avgInterestRate .
    }}
    ORDER BY DESC(?loanCount)
    """
    for i, row in enumerate(g.query(q)):
        if i >= 10:
            break
        state, loan_count, avg_ir = row
        print(
            f"state={state}, loans={int(loan_count)}, "
            f"avgInterestRate={float(avg_ir):.2f}"
        )


def main() -> None:
    g = load_graph()
    query_by_grade(g)
    query_by_term(g)
    query_by_purpose(g)
    query_by_home_ownership(g)
    query_by_income_band(g)
    query_by_state(g)


if __name__ == "__main__":
    main()


