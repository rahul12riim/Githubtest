---
language:
  - en
license: cc-by-4.0
task_categories:
  - text-classification
  - token-classification
task_ids:
  - multi-class-classification
  - named-entity-recognition
pretty_name: GovTech-Bench
size_categories:
  - 10K<n<100K
tags:
  - government
  - workforce
  - unemployment-insurance
  - i9
  - fraud-detection
  - synthetic
  - benchmark
  - ai-evaluation
dataset_info:
  description: >
    GovTech-Bench is a synthetic benchmark dataset for evaluating AI models on
    government document processing tasks, covering unemployment insurance claims,
    I-9 employment eligibility forms, and agency correspondence.
  version: 1.0.0
---

# GovTech-Bench

**A synthetic benchmark dataset for evaluating AI models on government document processing tasks.**

## Dataset Description

GovTech-Bench covers three document classes and five evaluation tasks drawn from real-world government technology workflows in unemployment insurance, employment eligibility verification, and agency correspondence processing.

All records are fully synthetic — no real personally identifiable information (PII) is included.

### Document Classes

| Class | Records | Description |
|---|---|---|
| `ui_claims` | ~4,000 | Unemployment insurance claim records across 10 states |
| `i9_forms` | ~3,000 | I-9 employment eligibility form records with OCR quality labels |
| `agency_docs` | ~3,000 | Agency correspondence: determination letters, response requests, notices |

### Evaluation Tasks

| Task | Input | Target |
|---|---|---|
| `document_classification` | Any record | Document class label |
| `fraud_detection` | UI claim record | valid / fraudulent / duplicate / identity_mismatch |
| `ner` | I-9 record | Extracted fields (name, doc type, doc number, expiry) |
| `ocr_quality` | I-9 record | clean / degraded / poor |
| `adjudication_routing` | Agency doc record | Routing destination + urgency tier |

## Quickstart

```python
from datasets import load_dataset

ds = load_dataset("rahulraj/govtech-bench", "ui_claims", split="test")
```

Or install the evaluation harness:

```bash
pip install govtechbench
```

```python
from govtechbench import load_task, evaluate

results = evaluate(
    task="fraud_detection",
    model_fn=your_model_fn,
    split="test",
    source="huggingface"
)
```

## Domain Background

This benchmark is informed by the author's direct practitioner experience in AI-driven government technology, including:

- Designing and deploying AI systems for large-scale unemployment insurance adjudication and compliance across state and federal agencies
- Building AI-powered document processing pipelines that reduced government document handling time from minutes to seconds
- Working at the intersection of workforce identity, fraud prevention, and enterprise-scale government systems

The fraud typology is drawn from publicly available GAO reports on pandemic UI fraud (GAO-22-104304 and related).

## Data Generation

The dataset is fully reproducible:

```bash
git clone https://github.com/rahul12riim/govtech-bench
pip install -e ".[dev]"
python generation/synthetic_gen.py --seed 42 --output data/
```

## Leaderboard

Submit your results at: [https://huggingface.co/spaces/rahulraj/govtech-bench-leaderboard](https://huggingface.co/spaces/rahulraj/govtech-bench-leaderboard)

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

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Code: MIT

## Author

**Rahul Raj** — AI and Government Technology practitioner with 17+ years of experience in enterprise AI, workforce systems, and digital identity.
