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
    U.S. government document processing tasks, covering unemployment insurance claims,
    I-9 employment eligibility forms, and agency correspondence.
  version: 1.0.0
---

# GovTech-Bench

**A synthetic benchmark dataset for evaluating AI models on U.S. government document processing tasks.**

## Dataset Description

GovTech-Bench covers three document classes and five evaluation tasks drawn from real-world government technology workflows in unemployment insurance, employment eligibility verification, and agency correspondence processing.

All records are fully synthetic — no real personally identifiable information (PII) is included.

### Document Classes

| Class | Records | Description |
|---|---|---|
| `ui_claims` | ~4,000 | Unemployment insurance claim records across 10 U.S. states |
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

# Load fraud detection task
ds = load_dataset("rahulraj/govtech-bench", "ui_claims", split="test")

# Load I-9 forms
ds = load_dataset("rahulraj/govtech-bench", "i9_forms", split="test")
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

This benchmark is informed by direct practitioner experience:

- Architecting AI-driven UI adjudication systems serving 52 U.S. state and federal agencies
- Invention of the Workforce Security Number (WSN), a federated workforce identity framework (US Patent #63/802574)
- Building the Inkless AI document processing platform, reducing processing time from 20 minutes to 2 seconds

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

```bash
python leaderboard/submit.py \
  --task fraud_detection \
  --model-name "your-model" \
  --org "Your Organization" \
  --results results.json
```

## Citation

```bibtex
@dataset{raj2026govtechbench,
  author    = {Rahul Raj},
  title     = {GovTech-Bench: A Synthetic Benchmark for AI Evaluation on U.S. Government Document Processing},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/rahulraj/govtech-bench}
}
```

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Code: MIT

## Author

**Rahul Raj** — Senior Principal Technical Program Manager, Walmart Global Tech | Inventor, Workforce Security Number (US Patent #63/802574) | White House recognized innovator in national workforce technology
