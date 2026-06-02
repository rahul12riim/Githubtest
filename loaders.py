"""
Dataset loading utilities for GovTech-Bench.

Supports loading from:
  - Local parquet/CSV files
  - Hugging Face Hub (datasets library)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

TASK_TO_CLASS = {
    "document_classification": None,          # all classes
    "fraud_detection": "ui_claims",
    "ner": "i9_forms",
    "ocr_quality": "i9_forms",
    "adjudication_routing": "agency_docs",
}

HF_REPO_ID = "rahulraj/govtech-bench"


def load_task(
    task: str,
    split: Literal["train", "val", "test"] = "test",
    source: Literal["local", "huggingface"] = "local",
    data_dir: str | Path = "data/",
) -> pd.DataFrame:
    """
    Load a GovTech-Bench task split.

    Args:
        task: one of document_classification | fraud_detection | ner |
              ocr_quality | adjudication_routing
        split: train | val | test
        source: local (reads from data_dir) or huggingface (downloads from Hub)
        data_dir: path to local data directory (ignored if source='huggingface')

    Returns:
        DataFrame with records for the requested task and split.
    """
    if task not in TASK_TO_CLASS:
        raise ValueError(
            f"Unknown task '{task}'. Choose from: {list(TASK_TO_CLASS.keys())}"
        )

    doc_class = TASK_TO_CLASS[task]

    if source == "huggingface":
        return _load_from_hub(doc_class, split)
    else:
        return _load_local(doc_class, split, Path(data_dir))


def _load_local(
    doc_class: str | None,
    split: str,
    data_dir: Path,
) -> pd.DataFrame:
    if doc_class is None:
        # document_classification: load all three classes and tag them
        frames = []
        for cls in ["ui_claims", "i9_forms", "agency_docs"]:
            df = _read_split(data_dir / cls, split)
            df["document_class"] = cls
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
    else:
        df = _read_split(data_dir / doc_class, split)
        df["document_class"] = doc_class
        return df


def _read_split(class_dir: Path, split: str) -> pd.DataFrame:
    parquet_path = class_dir / f"{split}.parquet"
    csv_path = class_dir / f"{split}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No data found for split '{split}' in {class_dir}. "
            "Run `python generation/synthetic_gen.py` first, or use source='huggingface'."
        )


def _load_from_hub(doc_class: str | None, split: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the `datasets` library: pip install datasets")

    config = doc_class if doc_class else "all"
    ds = load_dataset(HF_REPO_ID, config, split=split, trust_remote_code=False)
    return ds.to_pandas()
