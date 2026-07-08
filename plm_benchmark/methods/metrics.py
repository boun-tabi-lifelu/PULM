"""Shared scoring used by train / eval / test for every task type."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, mean_squared_error

from plm_benchmark.tasks import TaskSpec


def score_predictions(preds: np.ndarray, labels: np.ndarray, spec: TaskSpec) -> float:
    if spec.task_type in ("classification", "ppi"):
        return float(accuracy_score(labels, np.argmax(preds, axis=1)))
    if spec.task_type == "multilabel":
        # Per-label accuracy on logits, thresholded at 0 (== sigmoid>0.5). This is
        # exactly torchmetrics Accuracy(task="multilabel"): its macro average equals
        # the element-wise mean because every label has the same N samples. Pure
        # NumPy so no torchmetrics dependency.
        logits = np.asarray(preds, dtype=np.float64)
        target = np.asarray(labels).astype(int)
        pred_bin = (logits > 0.0).astype(int)
        return float((pred_bin == target).mean())
    if spec.metric == "mse":
        return float(mean_squared_error(labels, np.squeeze(preds).astype(float)))
    return float(
        stats.spearmanr(np.squeeze(preds).astype(float), np.asarray(labels, dtype=float)).correlation
    )


def metrics_fn(spec: TaskSpec):
    def compute(eval_pred):
        preds, labels = eval_pred
        return {spec.metric: score_predictions(preds, labels, spec)}

    return compute
