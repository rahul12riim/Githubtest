"""
GovTech-Bench synthetic data generator.

Generates ~10,000 fully synthetic government document records across three classes:
  - Unemployment Insurance (UI) claims
  - I-9 Employment Eligibility forms
  - Agency correspondence

All PII is procedurally generated. No real individual, employer, or claimant data is used.

Usage:
    python generation/synthetic_gen.py --seed 42 --output data/
"""

import argparse
import json
import random
import string
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

fake = Faker("en_US")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STATES = ["CA", "TX", "FL", "NY", "IL", "OH", "GA", "WA", "PA", "NJ"]

SEPARATION_REASONS = [
    "lack_of_work", "discharge_no_misconduct", "voluntary_quit",
    "temporary_layoff", "plant_closure", "reduction_in_force",
]

DOC_TYPES = [
    "US_Passport", "Permanent_Resident_Card", "Drivers_License",
    "State_ID", "Employment_Auth_Document", "Foreign_Passport_Visa",
]

AGENCY_SUBTYPES = [
    "determination_letter", "employer_response_request",
    "separation_notice", "appeal_acknowledgment",
]

ADJUDICATION_CODES = ["ADJ-001", "ADJ-002", "ADJ-003", "ADJ-004", "ADJ-005"]

ROUTING_DESTINATIONS = [
    "state_ui_office", "employer_response_unit",
    "appeals_board", "fraud_investigation_unit", "federal_dol_office",
]

URGENCY_TIERS = ["standard", "expedited", "appeal"]

FRAUD_PATTERNS = [
    "synthetic_identity", "wage_mismatch", "employer_collusion", "duplicate_filing"
]


def random_fein():
    return f"{random.randint(10,99)}-{random.randint(1000000,9999999)}"


def random_ssn_last4():
    return str(random.randint(1000, 9999))


def random_date_range(start_year=2020, end_year=2025):
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def random_wages(base=800, scale=3000, fraud=False):
    """Generate 4 quarterly wage amounts. Fraud variant introduces inconsistency."""
    if fraud:
        wages = [round(random.uniform(0, base * 0.3), 2) for _ in range(4)]
        wages[random.randint(0, 3)] = round(random.uniform(base * 5, base * 15), 2)
    else:
        wages = [round(random.uniform(base, base + scale), 2) for _ in range(4)]
    return wages


# ---------------------------------------------------------------------------
# UI Claims
# ---------------------------------------------------------------------------

def generate_ui_claim(claim_id: int, label: str, rng: random.Random) -> dict:
    state = rng.choice(STATES)
    filing_date = random_date_range()
    benefit_week = filing_date + timedelta(weeks=rng.randint(0, 26))
    employer_fein = random_fein()
    separation = rng.choice(SEPARATION_REASONS)
    wba = round(rng.uniform(150, 823), 2)  # weekly benefit amount

    fraud_pattern = None
    wages = random_wages(fraud=(label == "fraudulent"))

    if label == "fraudulent":
        fraud_pattern = rng.choice(FRAUD_PATTERNS)
        if fraud_pattern == "synthetic_identity":
            employer_fein = f"00-{rng.randint(1000000,9999999)}"
        elif fraud_pattern == "wage_mismatch":
            wages = random_wages(fraud=True)
        elif fraud_pattern == "duplicate_filing":
            label = "duplicate"

    return {
        "claim_id": f"UI-{state}-{claim_id:06d}",
        "state": state,
        "claimant_ssn_last4": random_ssn_last4(),
        "claimant_name": fake.name(),
        "employer_fein": employer_fein,
        "employer_name": fake.company(),
        "separation_reason": separation,
        "filing_date": filing_date.isoformat(),
        "benefit_week_ending": benefit_week.isoformat(),
        "weekly_benefit_amount": wba,
        "q1_wages": wages[0],
        "q2_wages": wages[1],
        "q3_wages": wages[2],
        "q4_wages": wages[3],
        "label": label,
        "fraud_pattern": fraud_pattern,
    }


def generate_ui_claims(n: int, rng: random.Random) -> pd.DataFrame:
    label_weights = {
        "valid": 0.55,
        "fraudulent": 0.25,
        "duplicate": 0.12,
        "identity_mismatch": 0.08,
    }
    labels = rng.choices(
        list(label_weights.keys()),
        weights=list(label_weights.values()),
        k=n
    )
    records = [generate_ui_claim(i, label, rng) for i, label in enumerate(labels, 1)]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# I-9 Forms
# ---------------------------------------------------------------------------

def generate_i9(record_id: int, label: str, rng: random.Random) -> dict:
    hire_date = random_date_range(2019, 2025)
    doc_type = rng.choice(DOC_TYPES)
    expiry_delta = rng.randint(-180, 1825)  # can be expired
    expiry_date = hire_date + timedelta(days=expiry_delta)
    ocr_quality = rng.choices(["clean", "degraded", "poor"], weights=[0.5, 0.35, 0.15])[0]

    issuing_authority = rng.choice([
        "U.S. Department of State", "U.S. DHS", "State DMV", "USCIS"
    ])

    doc_number_valid = "".join(rng.choices(string.ascii_uppercase + string.digits, k=9))
    employee_name = fake.name()
    employer_name = fake.company()
    attestation_date = hire_date + timedelta(days=rng.randint(0, 3))

    section1_complete = True
    section2_complete = True

    if label == "missing_field":
        section1_complete = rng.random() > 0.5
        section2_complete = not section1_complete
    if label == "expired_doc":
        expiry_date = hire_date - timedelta(days=rng.randint(1, 365))
    if label == "mismatched_identity":
        employee_name = fake.name()

    return {
        "record_id": f"I9-{record_id:06d}",
        "employee_name": employee_name,
        "hire_date": hire_date.isoformat(),
        "doc_type": doc_type,
        "doc_number": doc_number_valid,
        "doc_expiry_date": expiry_date.isoformat(),
        "issuing_authority": issuing_authority,
        "employer_name": employer_name,
        "employer_attestation_date": attestation_date.isoformat(),
        "section1_complete": section1_complete,
        "section2_complete": section2_complete,
        "ocr_quality": ocr_quality,
        "label": label,
    }


def generate_i9_forms(n: int, rng: random.Random) -> pd.DataFrame:
    label_weights = {
        "compliant": 0.60,
        "expired_doc": 0.18,
        "mismatched_identity": 0.12,
        "missing_field": 0.10,
    }
    labels = rng.choices(
        list(label_weights.keys()),
        weights=list(label_weights.values()),
        k=n
    )
    records = [generate_i9(i, label, rng) for i, label in enumerate(labels, 1)]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Agency Correspondence
# ---------------------------------------------------------------------------

def generate_agency_doc(record_id: int, rng: random.Random) -> dict:
    subtype = rng.choice(AGENCY_SUBTYPES)
    state = rng.choice(STATES)
    issue_date = random_date_range(2020, 2025)
    response_due = issue_date + timedelta(days=rng.choice([10, 14, 21, 30]))
    adjudication_code = rng.choice(ADJUDICATION_CODES)
    routing = rng.choice(ROUTING_DESTINATIONS)
    urgency = rng.choices(
        URGENCY_TIERS, weights=[0.65, 0.25, 0.10]
    )[0]

    employer_name = fake.company()
    claimant_ref = f"CLM-{rng.randint(100000,999999)}"

    body_templates = {
        "determination_letter": (
            f"This letter constitutes the official determination regarding claim reference {claimant_ref}. "
            f"Based on the information provided, the agency has determined eligibility under code {adjudication_code}. "
            f"Any appeal must be filed within 10 days of this notice."
        ),
        "employer_response_request": (
            f"The {state} Department of Labor requests your response regarding separation information for "
            f"claim reference {claimant_ref}. Please submit documentation to {routing.replace('_',' ')} "
            f"by {response_due.isoformat()}."
        ),
        "separation_notice": (
            f"Notice of separation issued for {employer_name}. Claimant reference: {claimant_ref}. "
            f"Separation effective date: {issue_date.isoformat()}. "
            f"Employer is required to provide wage and separation records within 10 business days."
        ),
        "appeal_acknowledgment": (
            f"Your appeal for claim {claimant_ref} has been received and assigned to the "
            f"{routing.replace('_',' ')}. A hearing will be scheduled under urgency classification: {urgency}. "
            f"Adjudication code applied: {adjudication_code}."
        ),
    }

    return {
        "record_id": f"AGY-{state}-{record_id:06d}",
        "state": state,
        "subtype": subtype,
        "issue_date": issue_date.isoformat(),
        "response_due_date": response_due.isoformat(),
        "claimant_reference": claimant_ref,
        "employer_name": employer_name,
        "adjudication_code": adjudication_code,
        "routing_destination": routing,
        "urgency_tier": urgency,
        "body_text": body_templates[subtype],
        "label_routing": routing,
        "label_urgency": urgency,
        "label_adjudication": adjudication_code,
    }


def generate_agency_docs(n: int, rng: random.Random) -> pd.DataFrame:
    records = [generate_agency_doc(i, rng) for i in range(1, n + 1)]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate GovTech-Bench synthetic dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="data/", help="Output directory")
    parser.add_argument("--ui-claims", type=int, default=4000, help="Number of UI claim records")
    parser.add_argument("--i9-forms", type=int, default=3000, help="Number of I-9 form records")
    parser.add_argument("--agency-docs", type=int, default=3000, help="Number of agency doc records")
    parser.add_argument("--format", choices=["parquet", "csv", "both"], default="both")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    Faker.seed(args.seed)
    output = Path(args.output)

    print(f"Generating GovTech-Bench dataset (seed={args.seed})")

    splits = {
        "ui_claims": (generate_ui_claims, args.ui_claims),
        "i9_forms": (generate_i9_forms, args.i9_forms),
        "agency_docs": (generate_agency_docs, args.agency_docs),
    }

    for name, (fn, n) in splits.items():
        print(f"  Generating {n} {name}...")
        if name == "agency_docs":
            df = fn(n, rng)
        else:
            df = fn(n, rng)

        # Train/val/test split: 70/15/15
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        train = df.iloc[:n_train]
        val = df.iloc[n_train:n_train + n_val]
        test = df.iloc[n_train + n_val:]

        dest = output / name
        dest.mkdir(parents=True, exist_ok=True)

        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            if args.format in ("parquet", "both"):
                split_df.to_parquet(dest / f"{split_name}.parquet", index=False)
            if args.format in ("csv", "both"):
                split_df.to_csv(dest / f"{split_name}.csv", index=False)

        print(f"    Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Write metadata
    meta = {
        "seed": args.seed,
        "version": "1.0.0",
        "total_records": args.ui_claims + args.i9_forms + args.agency_docs,
        "document_classes": ["ui_claims", "i9_forms", "agency_docs"],
        "splits": {"train": 0.70, "val": 0.15, "test": 0.15},
        "author": "Rahul Raj",
        "license": "CC BY 4.0",
    }
    with open(output / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Dataset written to: {output.resolve()}")


if __name__ == "__main__":
    main()
