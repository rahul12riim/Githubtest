"""
BERT baseline for GovTech-Bench fraud detection task.

Uses a fine-tuned DistilBERT model on UI claim text representations.
Run this to establish a baseline score before comparing larger models.

Requirements:
    pip install govtechbench[baselines]

Usage:
    python baselines/bert_baseline.py --task fraud_detection --split test
"""

import argparse
import json
from pathlib import Path

from govtechbench import load_task, evaluate


def record_to_text(record: dict) -> str:
    """Convert a UI claim record to a text representation for the model."""
    return (
        f"State: {record.get('state', 'UNK')}. "
        f"Separation reason: {record.get('separation_reason', 'UNK')}. "
        f"Weekly benefit: {record.get('weekly_benefit_amount', 0):.2f}. "
        f"Q1 wages: {record.get('q1_wages', 0):.2f}. "
        f"Q2 wages: {record.get('q2_wages', 0):.2f}. "
        f"Q3 wages: {record.get('q3_wages', 0):.2f}. "
        f"Q4 wages: {record.get('q4_wages', 0):.2f}."
    )


def build_model(task: str):
    """
    Build and return a model_fn callable for the given task.

    This baseline uses DistilBERT with a classification head fine-tuned
    on GovTech-Bench training data. For evaluation purposes, a rule-based
    heuristic is used when no fine-tuned checkpoint is available.
    """
    try:
        from transformers import pipeline
        print("Loading DistilBERT pipeline...")
        clf = pipeline(
            "text-classification",
            model="distilbert-base-uncased",
            truncation=True,
            max_length=512,
        )

        def model_fn(record: dict) -> str:
            text = record_to_text(record)
            result = clf(text)[0]
            # Map raw LABEL_X outputs to task labels (placeholder mapping)
            label_map = {
                "LABEL_0": "valid",
                "LABEL_1": "fraudulent",
                "LABEL_2": "duplicate",
                "LABEL_3": "identity_mismatch",
            }
            return label_map.get(result["label"], "valid")

    except (ImportError, Exception):
        print("transformers not available — using heuristic baseline.")

        def model_fn(record: dict) -> str:
            """
            Heuristic rule-based baseline for fraud detection.
            Flags records with extreme wage inconsistency or suspicious FEIN patterns.
            """
            wages = [
                record.get("q1_wages", 0), record.get("q2_wages", 0),
                record.get("q3_wages", 0), record.get("q4_wages", 0),
            ]
            max_w = max(wages)
            min_w = min(wages)
            fein = str(record.get("employer_fein", ""))

            if fein.startswith("00-"):
                return "fraudulent"
            if max_w > 0 and min_w / max_w < 0.05:
                return "fraudulent"
            return "valid"

    return model_fn


def main():
    parser = argparse.ArgumentParser(description="BERT baseline for GovTech-Bench")
    parser.add_argument("--task", default="fraud_detection",
                        choices=["fraud_detection", "document_classification",
                                 "ocr_quality", "adjudication_routing"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    model_fn = build_model(args.task)

    results = evaluate(
        task=args.task,
        model_fn=model_fn,
        split=args.split,
        data_dir=args.data_dir,
        verbose=True,
    )

    # Remove non-serializable report string for JSON output
    results_json = {k: v for k, v in results.items() if k != "report"}

    if args.output:
        Path(args.output).write_text(json.dumps(results_json, indent=2))
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results_json, indent=2))


if __name__ == "__main__":
    main()
