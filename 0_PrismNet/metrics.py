"""
Classification metrics utilities for linear probing.

Provides a unified function to compute accuracy, balanced accuracy, quadratic
weighted kappa, optional AUROC (binary or multi-class), and the scikit-learn
classification report. Includes a helper to pretty-print numeric metrics.
"""

from typing import Optional, Dict, Any, Union, List
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
    cohen_kappa_score,
    classification_report,
)

def get_eval_metrics(
    targets_all: Union[List[int], np.ndarray],
    preds_all: Union[List[int], np.ndarray],
    probs_all: Optional[Union[List[float], np.ndarray]] = None,
    get_report: bool = True,
    prefix: str = "",
    roc_kwargs: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Compute common classification metrics for predictions.

    Args:
        targets_all: True target values.
        preds_all: Predicted discrete labels.
        probs_all: Predicted probabilities or scores for AUROC (optional).
        get_report: Whether to include sklearn classification report.
        prefix: Optional key prefix for returned metrics.
        roc_kwargs: Extra kwargs passed to sklearn roc_auc_score.

    Returns:
        dict mapping metric names to values (and report if requested).

    """
    bacc = balanced_accuracy_score(targets_all, preds_all)
    kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
    acc = accuracy_score(targets_all, preds_all)
    cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0)

    eval_metrics = {
        f"{prefix}acc": acc,
        f"{prefix}bacc": bacc,
        f"{prefix}kappa": kappa,
        f"{prefix}weighted_f1": cls_rep["weighted avg"]["f1-score"],
    }

    if get_report:
        eval_metrics[f"{prefix}report"] = cls_rep

    if probs_all is not None:
        roc_auc = roc_auc_score(targets_all, probs_all, **roc_kwargs)
        eval_metrics[f"{prefix}auroc"] = roc_auc

    return eval_metrics

def print_metrics(eval_metrics):
    """Pretty-print numeric metrics (skip long classification report)."""
    for k, v in eval_metrics.items():
        if "report" in k:
            continue
        
        print(f"Test {k}: {v:.3f}")