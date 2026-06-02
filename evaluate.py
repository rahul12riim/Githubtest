"""
Main evaluation runner for GovTech-Bench.

Usage:
    from govtechbench import load_task, evaluate

    dataset = load_task("fraud_detection", split="test")
    results = evaluate(task="fraud_detection", model_fn=my_model, split="test")
"""

from __future__ import annotations

from typing import Callable, Literal

from rich.console import Console
from rich.table import Table

from govtechbench.loaders import load_task
from govtechbench.metrics import (
    classification_metrics,
    fraud_detection_metrics,
    ner_metrics,
    LatencyTracker,
)

console = Console()

TASK_LABEL_COLUMNS = {
    "document_classification": "document_class",
    "fraud_detection": "label",
    "ner": None,                  # handled separately
    "ocr_quality": "ocr_quality",
    "adjudication_routing": "label_routing",
}


def evaluate(
    task: str,
    model_fn: Callable,
    split: Literal["train", "val", "test"] = "test",
    source: Literal["local", "huggingface"] = "local",
    data_dir: str = "data/",
    verbose: bool = True,
) -> dict:
    """
    Evaluate a model function against a GovTech-Bench task.

    Args:
        task: task name (see TASK_LABEL_COLUMNS keys)
        model_fn: callable that accepts a single record dict and returns a
                  label string (or dict for NER)
        split: data split to evaluate on
        source: 'local' or 'huggingface'
        data_dir: local data directory path
        verbose: print results table to stdout

    Returns:
        dict of metric name -> value
    """
    df = load_task(task, split=split, source=source, data_dir=data_dir)
    records = df.to_dict(orient="records")

    tracker = LatencyTracker()
    predictions = []

    console.print(f"[bold]Evaluating[/bold] task=[cyan]{task}[/cyan] "
                  f"split=[cyan]{split}[/cyan] n=[cyan]{len(records)}[/cyan]")

    for record in records:
        pred = tracker.record(model_fn, record)
        predictions.append(pred)

    results = _compute_metrics(task, records, predictions, df)
    results.update(tracker.summary)

    if verbose:
        _print_results(task, results)

    return results


def _compute_metrics(task, records, predictions, df) -> dict:
    if task == "ner":
        ner_fields = ["employee_name", "doc_type", "doc_number", "doc_expiry_date"]
        y_true = [{f: r.get(f) for f in ner_fields} for r in records]
        return ner_metrics(y_true, predictions, fields=ner_fields)

    label_col = TASK_LABEL_COLUMNS[task]
    y_true = [str(r[label_col]) for r in records]
    y_pred = [str(p) for p in predictions]

    if task == "fraud_detection":
        return fraud_detection_metrics(y_true, y_pred)
    else:
        return classification_metrics(y_true, y_pred)


def _print_results(task: str, results: dict):
    table = Table(title=f"GovTech-Bench — {task}", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    skip = {"report"}
    for k, v in results.items():
        if k in skip:
            continue
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                table.add_row(f"  {k} / {sub_k}", str(sub_v))
        else:
            table.add_row(k, str(v))

    console.print(table)
    if "report" in results:
        console.print(results["report"])
