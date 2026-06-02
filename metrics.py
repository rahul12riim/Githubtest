"""
Evaluation metrics for GovTech-Bench tasks.

Implements: F1 (macro/weighted), AUC-ROC, false positive rate,
entity-level NER metrics, and latency tracking.
"""

from __future__ import annotations

import time
from typing import Callable, List

import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    classification_report,
)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] | None = None,
) -> dict:
    """Macro F1, accuracy, and per-class breakdown."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "n_samples": len(y_true),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Fraud detection metrics
# ---------------------------------------------------------------------------

def fraud_detection_metrics(
    y_true: List[str],
    y_pred: List[str],
    y_prob: List[float] | None = None,
) -> dict:
    """F1 macro, FPR, and optionally AUC-ROC for binary fraud detection."""
    metrics = classification_metrics(y_true, y_pred)

    # False positive rate: predicted fraud when actually valid
    valid_mask = np.array(y_true) == "valid"
    pred_fraud = np.array(y_pred) != "valid"
    fp = int((valid_mask & pred_fraud).sum())
    tn = int((valid_mask & ~pred_fraud).sum())
    metrics["false_positive_rate"] = round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4)

    if y_prob is not None:
        binary_true = [0 if t == "valid" else 1 for t in y_true]
        try:
            metrics["auc_roc"] = round(roc_auc_score(binary_true, y_prob), 4)
        except ValueError:
            metrics["auc_roc"] = None

    return metrics


# ---------------------------------------------------------------------------
# NER metrics
# ---------------------------------------------------------------------------

def ner_metrics(
    y_true: List[dict],
    y_pred: List[dict],
    fields: List[str] | None = None,
) -> dict:
    """
    Entity-level precision, recall, F1 per field type.

    Args:
        y_true: list of dicts {field_name: value_or_None}
        y_pred: list of dicts {field_name: extracted_value_or_None}
        fields: field names to evaluate; defaults to keys of first y_true item
    """
    if fields is None:
        fields = list(y_true[0].keys()) if y_true else []

    results = {}
    for field in fields:
        tp = fp = fn = 0
        for gold, pred in zip(y_true, y_pred):
            gold_val = gold.get(field)
            pred_val = pred.get(field)
            if gold_val and pred_val:
                if str(gold_val).strip().lower() == str(pred_val).strip().lower():
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif gold_val and not pred_val:
                fn += 1
            elif not gold_val and pred_val:
                fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        results[field] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    macro_f1 = np.mean([v["f1"] for v in results.values()]) if results else 0.0
    results["__macro_f1__"] = round(float(macro_f1), 4)
    return results


# ---------------------------------------------------------------------------
# Latency tracking
# ---------------------------------------------------------------------------

class LatencyTracker:
    """Context manager that records wall-clock inference time."""

    def __init__(self):
        self._times: List[float] = []
        self._start: float | None = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        if self._start is not None:
            self._times.append((time.perf_counter() - self._start) * 1000)

    def record(self, fn: Callable, *args, **kwargs):
        """Call fn(*args, **kwargs) and record its latency."""
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        self._times.append((time.perf_counter() - start) * 1000)
        return result

    @property
    def mean_ms(self) -> float:
        return round(float(np.mean(self._times)), 2) if self._times else 0.0

    @property
    def p95_ms(self) -> float:
        return round(float(np.percentile(self._times, 95)), 2) if self._times else 0.0

    @property
    def summary(self) -> dict:
        return {
            "mean_latency_ms": self.mean_ms,
            "p95_latency_ms": self.p95_ms,
            "n_calls": len(self._times),
        }
