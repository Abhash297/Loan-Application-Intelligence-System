"""
Cohort-based knowledge graph construction for the Lending Club loan dataset.

This module:
1. Computes cohort-level statistics from the cleaned loan dataset.
2. Builds an RDF graph with one node per cohort using rdflib.
3. Saves the graph to a Turtle file for later use by the chatbot / API.

The cohorts follow the design we agreed on:
- By grade
- By term
- By purpose
- By loan status
- By home ownership
- By income band
- By state
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import RDFS, XSD


@dataclass
class CohortConfig:
    name: str  # e.g. "grade"
    column: str  # column in the dataframe
    value_labels: Optional[Dict[str, str]] = None  # optional mapping to nicer labels


def load_cleaned_data(path: Path) -> pd.DataFrame:
    """
    Load the cleaned loan dataset.

    We assume this is the same data used for training the classifier, with a
    binary `target` column (1 = default/bad, 0 = good).
    """
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {path}")

    # Use low_memory=False for safer dtypes; dataset is large so we may want to
    # stream / sample in future, but for now we rely on available memory.
    df = pd.read_csv(path, low_memory=False)

    # Ensure required columns are present
    required = {
        "grade",
        "term",
        "purpose",
        "loan_status",
        "home_ownership",
        "annual_inc",
        "addr_state",
        "loan_amnt",
        "int_rate",
        "dti",
        "target",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in cleaned dataset: {missing}")

    return df


def _to_numeric(series: pd.Series) -> pd.Series:
    """Best-effort conversion to numeric, ignoring non-convertible values."""
    return pd.to_numeric(series, errors="coerce")


def assign_income_band(annual_inc: pd.Series) -> pd.Series:
    """
    Bucket annual income into coarse bands for cohorting.
    """
    inc = _to_numeric(annual_inc).fillna(0)
    bins = [-np.inf, 50000, 75000, 100000, 150000, np.inf]
    labels = ["<50k", "50k-75k", "75k-100k", "100k-150k", ">150k"]
    return pd.cut(inc, bins=bins, labels=labels, right=False)


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def compute_cohorts(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute cohort statistics for multiple dimensions.

    Returns a dict mapping cohort type -> DataFrame with:
    - key columns (e.g. 'grade', 'term', etc.)
    - loanCount
    - defaultRate
    - avgInterestRate
    - avgLoanAmount
    - optional extras (avgDTI, avgIncome)
    """
    df_local = df.copy()

    # Normalize some fields
    df_local["grade"] = df_local["grade"].astype(str).str.strip()
    df_local["term"] = df_local["term"].astype(str).str.strip()
    df_local["purpose"] = df_local["purpose"].astype(str).str.strip()
    df_local["loan_status"] = df_local["loan_status"].astype(str).str.strip()
    df_local["home_ownership"] = df_local["home_ownership"].astype(str).str.strip()
    df_local["addr_state"] = df_local["addr_state"].astype(str).str.strip()

    df_local["loan_amnt"] = _to_numeric(df_local["loan_amnt"])
    df_local["int_rate"] = _to_numeric(df_local["int_rate"])
    df_local["dti"] = _to_numeric(df_local["dti"])
    df_local["annual_inc"] = _to_numeric(df_local["annual_inc"])

    # Income band
    df_local["income_band"] = assign_income_band(df_local["annual_inc"])

    cohorts: Dict[str, pd.DataFrame] = {}

    def make_cohort(group_col: str, cohort_type: str, extra_cols: Optional[List[str]] = None) -> pd.DataFrame:
        cols_for_group = [group_col]
        if extra_cols:
            cols_for_group.extend(extra_cols)

        grouped = df_local.groupby(group_col)
        stats = grouped.agg(
            loanCount=("target", "size"),
            defaultCount=("target", "sum"),
            avgInterestRate=("int_rate", "mean"),
            avgLoanAmount=("loan_amnt", "mean"),
            avgDTI=("dti", "mean"),
            avgIncome=("annual_inc", "mean"),
        ).reset_index()

        stats["defaultRate"] = stats.apply(
            lambda r: _safe_rate(r["defaultCount"], r["loanCount"]), axis=1
        )
        stats["cohortType"] = cohort_type

        # Optionally drop defaultRate where it doesn't make sense (e.g. pure status),
        # but it's also fine to keep it for uniformity.

        return stats

    # 1. By Grade
    cohorts["grade"] = make_cohort("grade", "grade")

    # 2. By Term
    cohorts["term"] = make_cohort("term", "term")

    # 3. By Purpose
    cohorts["purpose"] = make_cohort("purpose", "purpose")

    # 4. By Loan Status
    cohorts["status"] = make_cohort("loan_status", "status")

    # 5. By Home Ownership
    cohorts["home_ownership"] = make_cohort("home_ownership", "home_ownership")

    # 6. By Income Band
    cohorts["income_band"] = make_cohort("income_band", "income_band")

    # 7. By State
    cohorts["state"] = make_cohort("addr_state", "state")

    return cohorts


def build_cohort_graph(cohorts: Dict[str, pd.DataFrame], base_uri: str = "http://example.org/loan#") -> Graph:
    """
    Build an RDF graph with one node per cohort row.
    """
    g = Graph()
    EX = Namespace(base_uri)

    g.bind("ex", EX)
    g.bind("rdfs", RDFS)

    # Define Cohort class
    g.add((EX["Cohort"], RDF.type, RDFS.Class))

    def add_literal(subject: URIRef, predicate: URIRef, value, datatype=None):
        if pd.isna(value):
            return
        if datatype is None:
            lit = Literal(value)
        else:
            lit = Literal(value, datatype=datatype)
        g.add((subject, predicate, lit))

    for cohort_type, df_cohort in cohorts.items():
        for _, row in df_cohort.iterrows():
            key_val = row.iloc[0]
            key_str_raw = str(key_val).replace(" ", "_")
            # Sanitize to a URI-safe fragment (letters, digits, underscore)
            key_str = re.sub(r"[^A-Za-z0-9_]", "_", key_str_raw)
            cohort_uri = EX[f"Cohort_{cohort_type}_{key_str}"]

            g.add((cohort_uri, RDF.type, EX["Cohort"]))

            # cohortType
            add_literal(cohort_uri, EX["cohortType"], cohort_type)

            # Dimension-specific key
            if cohort_type == "grade":
                add_literal(cohort_uri, EX["grade"], key_val)
            elif cohort_type == "term":
                add_literal(cohort_uri, EX["term"], key_val)
            elif cohort_type == "purpose":
                add_literal(cohort_uri, EX["purpose"], key_val)
            elif cohort_type == "status":
                add_literal(cohort_uri, EX["loanStatus"], key_val)
            elif cohort_type == "home_ownership":
                add_literal(cohort_uri, EX["homeOwnership"], key_val)
            elif cohort_type == "income_band":
                add_literal(cohort_uri, EX["incomeBand"], key_val)
            elif cohort_type == "state":
                add_literal(cohort_uri, EX["state"], key_val)

            # Common stats
            add_literal(cohort_uri, EX["loanCount"], int(row["loanCount"]), datatype=XSD.integer)
            add_literal(cohort_uri, EX["defaultRate"], float(row.get("defaultRate", 0.0)), datatype=XSD.decimal)
            add_literal(
                cohort_uri,
                EX["avgInterestRate"],
                float(row.get("avgInterestRate", 0.0)),
                datatype=XSD.decimal,
            )
            add_literal(
                cohort_uri,
                EX["avgLoanAmount"],
                float(row.get("avgLoanAmount", 0.0)),
                datatype=XSD.decimal,
            )
            add_literal(
                cohort_uri,
                EX["avgDTI"],
                float(row.get("avgDTI", 0.0)),
                datatype=XSD.decimal,
            )
            add_literal(
                cohort_uri,
                EX["avgIncome"],
                float(row.get("avgIncome", 0.0)),
                datatype=XSD.decimal,
            )

    return g


def build_and_save_kg(
    data_path: Path,
    output_ttl: Path,
    base_uri: str = "http://example.org/loan#",
) -> Tuple[Dict[str, pd.DataFrame], Graph]:
    """
    Convenience entry point:
    - load data
    - compute cohorts
    - build graph
    - serialize to Turtle
    """
    df = load_cleaned_data(data_path)
    cohorts = compute_cohorts(df)
    g = build_cohort_graph(cohorts, base_uri=base_uri)

    output_ttl.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(output_ttl), format="turtle")

    return cohorts, g


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / "cleaned_loan_dataset.csv"
    output_file = base_dir / "loan_cohorts.ttl"

    print(f"Loading data from {data_file} ...")
    cohorts_dict, graph = build_and_save_kg(data_file, output_file)
    print(f"Built cohorts for types: {list(cohorts_dict.keys())}")
    print(f"Graph has {len(graph)} triples.")
    print(f"Turtle serialization written to: {output_file}")


