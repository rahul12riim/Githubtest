"""
Submit evaluation results to the GovTech-Bench public leaderboard.

The leaderboard is hosted on Hugging Face Spaces:
https://huggingface.co/spaces/rahulraj/govtech-bench-leaderboard

Usage:
    python leaderboard/submit.py \
        --task fraud_detection \
        --model-name "your-model-name" \
        --org "Your Organization" \
        --results results.json

Submissions are reviewed before appearing on the public leaderboard.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


LEADERBOARD_URL = "https://huggingface.co/spaces/rahulraj/govtech-bench-leaderboard"
SUBMISSION_API = "https://huggingface.co/spaces/rahulraj/govtech-bench-leaderboard/api/submit"

REQUIRED_FIELDS = {
    "fraud_detection": ["f1_macro", "false_positive_rate", "n_samples"],
    "document_classification": ["f1_macro", "accuracy", "n_samples"],
    "ner": ["__macro_f1__", "n_samples"],
    "ocr_quality": ["f1_macro", "accuracy", "n_samples"],
    "adjudication_routing": ["f1_macro", "accuracy", "n_samples"],
}


def validate_results(task: str, results: dict) -> list[str]:
    required = REQUIRED_FIELDS.get(task, [])
    missing = [f for f in required if f not in results]
    return missing


def submit(task, model_name, org, results_path, model_url=None, notes=None):
    results_file = Path(results_path)
    if not results_file.exists():
        print(f"ERROR: results file not found: {results_path}")
        sys.exit(1)

    with open(results_file) as f:
        results = json.load(f)

    missing = validate_results(task, results)
    if missing:
        print(f"ERROR: results file is missing required fields: {missing}")
        print(f"Required for task '{task}': {REQUIRED_FIELDS[task]}")
        sys.exit(1)

    payload = {
        "task": task,
        "model_name": model_name,
        "organization": org,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "model_url": model_url,
        "notes": notes,
        "results": results,
        "benchmark_version": "1.0.0",
    }

    print("\nSubmission payload:")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))
    print(f"\nResults summary:")
    for k, v in results.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")

    print(f"\nTo submit, visit: {LEADERBOARD_URL}")
    print("Upload this payload as a JSON file in the submission form.")
    print("\nPayload saved to: submission_payload.json")
    Path("submission_payload.json").write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Submit results to GovTech-Bench leaderboard")
    parser.add_argument("--task", required=True,
                        choices=list(REQUIRED_FIELDS.keys()),
                        help="Benchmark task name")
    parser.add_argument("--model-name", required=True,
                        help="Model or system name for leaderboard display")
    parser.add_argument("--org", required=True,
                        help="Organization or institution name")
    parser.add_argument("--results", required=True,
                        help="Path to JSON file containing evaluation results")
    parser.add_argument("--model-url", default=None,
                        help="Optional URL to model card or paper")
    parser.add_argument("--notes", default=None,
                        help="Optional notes about the submission")
    args = parser.parse_args()

    submit(
        task=args.task,
        model_name=args.model_name,
        org=args.org,
        results_path=args.results,
        model_url=args.model_url,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
