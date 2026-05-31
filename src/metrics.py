"""
Metric functions for plasma surrogate evaluation.

This module implements the legacy evaluation protocol used in the paper:

1. Compute field-wise RMSE at each time step.
2. Aggregate time-wise RMSE values using RMS:
       sqrt(mean(RMSE_t^2))
3. Average the field-wise scores over [v, p, rho].
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch


FIELD_NAMES = ["v", "p", "rho"]


def rmse_np(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1.0e-12,
) -> float:

    return float(np.sqrt(np.mean((pred - target) ** 2) + eps))


def rmse_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-12,
) -> torch.Tensor:


    return torch.sqrt(torch.mean((pred - target) ** 2) + eps)


def fieldwise_rmse_np(
    pred: np.ndarray,
    target: np.ndarray,
    field_names: List[str] | None = None,
    eps: float = 1.0e-12,
) -> Dict[str, float]:


    if field_names is None:
        field_names = FIELD_NAMES

    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape. "
            f"Got {pred.shape} and {target.shape}."
        )

    if pred.ndim != 2 or pred.shape[1] != len(field_names):
        raise ValueError(
            f"pred and target must have shape (N, {len(field_names)}). "
            f"Got {pred.shape}."
        )

    result = {}

    for idx, name in enumerate(field_names):
        result[f"rmse_{name}"] = rmse_np(
            pred=pred[:, idx],
            target=target[:, idx],
            eps=eps,
        )

    return result


def rms_aggregate_np(
    values: Iterable[float],
    eps: float = 1.0e-12,
) -> float:

    values = np.asarray(list(values), dtype=np.float64)

    if values.size == 0:
        return float("nan")

    return float(np.sqrt(np.mean(values ** 2) + eps))


def legacy_condition_score(
    timewise_field_rmse: Dict[str, List[float]],
    field_names: List[str] | None = None,
    eps: float = 1.0e-12,
) -> Dict[str, float]:


    if field_names is None:
        field_names = FIELD_NAMES

    field_scores = {}

    for name in field_names:
        field_scores[name] = rms_aggregate_np(
            timewise_field_rmse.get(name, []),
            eps=eps,
        )

    final_score = float(
        np.nanmean([field_scores[name] for name in field_names])
    )

    result = {
        "SLRMSE": final_score,
    }

    for name in field_names:
        result[f"SLRMSE_{name}"] = field_scores[name]

    return result


def global_rmse_np(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1.0e-12,
) -> float:


    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape. "
            f"Got {pred.shape} and {target.shape}."
        )

    return float(np.sqrt(np.mean((pred - target) ** 2) + eps))