"""GovTech-Bench: AI benchmark for U.S. government document processing."""

from govtechbench.loaders import load_task
from govtechbench.evaluate import evaluate

__version__ = "1.0.0"
__author__ = "Rahul Raj"
__all__ = ["load_task", "evaluate"]
