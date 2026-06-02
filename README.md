# GovTech-Bench

**A synthetic benchmark dataset for evaluating AI models on government document processing tasks.**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-govtech--bench-yellow)](https://huggingface.co/datasets/rahulraj/govtech-bench)

---

## Overview

GovTech-Bench is an open benchmark for evaluating large language models and AI systems on the document processing tasks that underpin workforce and employment compliance systems. It covers three document classes and five evaluation tasks drawn from real-world government technology workflows.

The dataset is **fully synthetic** — all personally identifiable information (PII) is procedurally generated. No real claimant, employer, or individual data is included.

**Author:** Rahul Raj — AI and Government Technology practitioner with 17+ years of experience in enterprise AI, workforce systems, and digital identity.

---

## Dataset at a Glance

| Document class | Records | Task types |
|---|---|---|
| Unemployment Insurance (UI) claims | ~4,000 | Classification, fraud detection |
| I-9 Employment Eligibility forms | ~3,000 | NER, OCR quality assessment |
| Agency correspondence | ~3,000 | Adjudication routing, classification |
| **Total** | **~10,000** | **5 task types** |

---

## Document Classes

### 1. Unemployment Insurance Claims (`data/ui_claims/`)

Synthetic UI claim records modeled on state agency form structures across 10 states (CA, TX, FL, NY, IL, OH, GA, WA, PA, NJ).

**Fields:** claimant ID, employer FEIN, wage quarters (4), separation reason, benefit week, weekly benefit amount, state code, filing date

**Labels:**
- `valid` — claim passes all consistency checks
- `fraudulent` — contains one or more fraud patterns
- `duplicate` — duplicate filing for same benefit week
- `identity_mismatch` — claimant identity inconsistent with wage records

**Fraud patterns modeled** (based on GAO pandemic UI fraud taxonomy):
- Synthetic identity (fabricated SSN/employer combination)
- Wage record mismatch (claimed wages inconsistent with employer records)
- Employer collusion (multiple claims from same employer within short window)
- Duplicate filing (same claimant, overlapping benefit weeks)

### 2. I-9 Employment Eligibility Forms (`data/i9_forms/`)

Synthetic I-9 completion records with injected OCR noise at three quality levels.

**Fields:** employee name, document type, document number, expiration date, issuing authority, employer attestation date, section completion flags

**Labels:**
- `compliant` — fully completed, unexpired documents
- `expired_doc` — document past expiration at time of hire
- `mismatched_identity` — name/DOB inconsistency across sections
- `missing_field` — required field absent

**OCR quality levels:** `clean` / `degraded` / `poor` (blur + rotation noise injected)

### 3. Agency Correspondence (`data/agency_docs/`)

Synthetic government agency letters and notices modeled on workforce agency communication formats.

**Subtypes:** determination letters, employer response requests, separation notices, appeals acknowledgments

**Labels:** adjudication code, required response action, urgency tier (`standard` / `expedited` / `appeal`), routing destination

---

## Evaluation Tasks

### Task 1: Document Classification
Classify a document into one of the three document classes.
- **Metric:** Accuracy, macro F1

### Task 2: UI Claim Fraud Detection
Given a UI claim record, predict the label (`valid` / `fraudulent` / `duplicate` / `identity_mismatch`).
- **Metric:** Macro F1, AUC-ROC, false positive rate (FPR)

### Task 3: I-9 Named Entity Recognition
Extract structured fields from I-9 text: employee name, document type, document number, expiration date.
- **Metric:** Entity-level precision, recall, F1 per field type

### Task 4: OCR Quality Assessment
Given a scanned I-9 image representation, classify OCR quality tier.
- **Metric:** Classification accuracy, F1

### Task 5: Adjudication Routing
Given an agency correspondence record, predict the routing destination and urgency tier.
- **Metric:** Routing accuracy, urgency tier F1, mean inference latency (ms)

---

## Quickstart

```bash
pip install govtechbench
```

```python
from govtechbench import load_task, evaluate

# Load a task
dataset = load_task("fraud_detection")

# Run evaluation against your model
results = evaluate(
    task="fraud_detection",
    model_fn=your_model_fn,   # callable: record -> label string
    split="test"
)

print(results)
# {
#   "f1_macro": 0.84,
#   "auc_roc": 0.91,
#   "false_positive_rate": 0.06,
#   "n_samples": 800
# }
```

---

## Repository Structure

```
govtech-bench/
├── README.md
├── pyproject.toml
├── data/
│   ├── ui_claims/          # Parquet + CSV
│   ├── i9_forms/           # Parquet + CSV
│   └── agency_docs/        # Parquet + CSV
├── govtechbench/
│   ├── __init__.py
│   ├── evaluate.py         # Main evaluation runner
│   ├── metrics.py          # All metric implementations
│   ├── loaders.py          # Dataset loading utilities
│   └── tasks/              # Per-task evaluation modules
├── generation/
│   └── synthetic_gen.py    # Reproducible data generation
├── baselines/
│   └── bert_baseline.py
├── leaderboard/
│   └── submit.py
└── tests/
    └── test_metrics.py
```

---

## Generating the Dataset

The full dataset is generated deterministically from a fixed seed:

```bash
python generation/synthetic_gen.py --seed 42 --output data/
```

This produces all three document class files in both `.parquet` and `.csv` formats.

---

## Leaderboard

Submit your model results to the public leaderboard on Hugging Face Spaces:

```bash
python leaderboard/submit.py \
  --task fraud_detection \
  --model-name "your-model-name" \
  --org "your-organization" \
  --results results.json
```

Leaderboard: [https://huggingface.co/spaces/rahulraj/govtech-bench-leaderboard](https://huggingface.co/spaces/rahulraj/govtech-bench-leaderboard)

---

## Citation

```bibtex
@dataset{raj2026govtechbench,
  author    = {Rahul Raj},
  title     = {GovTech-Bench: A Synthetic Benchmark for AI Evaluation on Government Document Processing},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/rahulraj/govtech-bench}
}
```

---

## Domain Background

This benchmark is informed by the author's direct practitioner experience in AI-driven government technology, including:

- Designing and deploying AI systems for large-scale unemployment insurance adjudication and compliance across state and federal agencies
- Building AI-powered document processing pipelines that reduced government document handling time from minutes to seconds
- Working at the intersection of workforce identity, fraud prevention, and enterprise-scale government systems

The fraud typology modeled in GovTech-Bench is drawn from publicly available GAO reports on pandemic UI fraud (GAO-22-104304 and related reports).

---

## License

Dataset: [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
Code: [MIT License](LICENSE)

---

## Contributing

Pull requests welcome. Please open an issue first for significant changes.

**Particularly welcome:**
- Additional state UI form templates
- New fraud pattern variants
- Baseline model implementations
- Translations of agency correspondence into additional languages

---

*Maintained by Rahul Raj — [LinkedIn](https://www.linkedin.com/in/rahulrajmba/) 
