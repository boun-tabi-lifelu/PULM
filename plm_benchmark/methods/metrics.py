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
        # Match PETA exactly: torchmetrics Accuracy(task="multilabel") on logits.
        import torch
        from torchmetrics import Accuracy

        logits = torch.as_tensor(np.asarray(preds), dtype=torch.float32)
        target = torch.as_tensor(np.asarray(labels)).int()
        metric = Accuracy(task="multilabel", num_labels=spec.num_labels)
        return float(metric(logits, target))
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
