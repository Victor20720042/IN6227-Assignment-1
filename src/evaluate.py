"""Evaluation metrics, restricted to what Lecture 4 covers.

Includes the two statistical procedures from the end of that lecture -- a
confidence interval for a single accuracy, and an interval for the difference
between two models -- because "which model is better" needs an answer that
survives sampling noise, not just two point estimates.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Z values for two-sided intervals, as tabulated in Lecture 4.
Z_TABLE = {0.90: 1.65, 0.95: 1.96, 0.98: 2.33, 0.99: 2.58}


@dataclass
class Metrics:
    n: int
    accuracy: float
    error_rate: float
    precision: float
    recall: float
    specificity: float
    false_positive_rate: float
    f1: float
    roc_auc: float
    pr_auc: float
    tp: int
    fp: int
    fn: int
    tn: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_metrics(y_true, y_pred, y_score=None) -> Metrics:
    """Confusion-matrix derived metrics, with the positive class as class 1."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = int(tn + fp + fn + tp)

    accuracy = _safe_div(tp + tn, n)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)

    return Metrics(
        n=n,
        accuracy=accuracy,
        error_rate=1.0 - accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        false_positive_rate=1.0 - specificity,
        f1=_safe_div(2 * precision * recall, precision + recall),
        roc_auc=float(roc_auc_score(y_true, y_score)) if y_score is not None else float("nan"),
        pr_auc=(
            float(average_precision_score(y_true, y_score))
            if y_score is not None
            else float("nan")
        ),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
    )


def confusion_frame(y_true, y_pred, positive_label: str = "yes", negative_label: str = "no"):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=pd.Index([f"actual {negative_label}", f"actual {positive_label}"]),
        columns=pd.Index([f"predicted {negative_label}", f"predicted {positive_label}"]),
    )


def accuracy_confidence_interval(
    accuracy: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Interval for the true accuracy, treating each prediction as a Bernoulli trial.

    Solving the normal approximation for p rather than centring the interval on
    the observed accuracy, which is the form Lecture 4 derives.
    """
    z = Z_TABLE.get(confidence)
    if z is None:
        raise ValueError(f"confidence must be one of {sorted(Z_TABLE)}")
    if n <= 0:
        return (float("nan"), float("nan"))

    denominator = 2 * (n + z**2)
    centre = 2 * n * accuracy + z**2
    spread = z * math.sqrt(z**2 + 4 * n * accuracy - 4 * n * accuracy**2)
    return ((centre - spread) / denominator, (centre + spread) / denominator)


@dataclass
class ModelComparison:
    error_a: float
    error_b: float
    difference: float
    std_error: float
    z: float
    interval: tuple[float, float]
    significant: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["interval_low"], payload["interval_high"] = payload.pop("interval")
        return payload


def compare_error_rates(
    error_a: float,
    n_a: int,
    error_b: float,
    n_b: int,
    confidence: float = 0.95,
    *,
    shared_test_set: bool = True,
) -> ModelComparison:
    """Is the gap between two error rates larger than sampling noise?

    The derivation in Lecture 4 assumes the two models were measured on
    independent test sets, so that the variances simply add. Scoring both models
    on the same held-out set breaks that assumption -- their errors are positively
    correlated, which makes the variance of the difference smaller than this
    estimate. The interval is therefore conservative, and that is recorded in the
    note rather than silently ignored.
    """
    z = Z_TABLE.get(confidence)
    if z is None:
        raise ValueError(f"confidence must be one of {sorted(Z_TABLE)}")

    difference = error_a - error_b
    variance = error_a * (1 - error_a) / n_a + error_b * (1 - error_b) / n_b
    std_error = math.sqrt(variance)
    low, high = difference - z * std_error, difference + z * std_error

    note = (
        "Both models scored on the same held-out set, so the independence "
        "assumption behind this interval is violated; treat it as conservative."
        if shared_test_set
        else "Independent test sets, matching the assumption in the derivation."
    )

    return ModelComparison(
        error_a=error_a,
        error_b=error_b,
        difference=difference,
        std_error=std_error,
        z=z,
        interval=(low, high),
        significant=not (low <= 0.0 <= high),
        note=note,
    )


def total_misclassification_cost(y_true, y_pred, cost_fn: float = 1.0, cost_fp: float = 1.0):
    """Weighted error under a cost matrix, for when the two mistakes differ in price."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(cost_fn * fn + cost_fp * fp)


def cost_table(
    predictions: dict[str, tuple], ratios: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
) -> pd.DataFrame:
    """Total cost per model as a missed positive gets more expensive than a false alarm.

    Accuracy implicitly fixes the ratio at 1:1 and, on a skewed problem, that is
    the one setting where doing nothing looks competitive. Varying it shows
    whether a classifier is doing real work.
    """
    rows = []
    for name, (y_true, y_pred) in predictions.items():
        row = {"model": name}
        for ratio in ratios:
            cost = total_misclassification_cost(y_true, y_pred, cost_fn=ratio, cost_fp=1.0)
            row[f"cost_fn:fp={ratio:g}:1"] = cost
        rows.append(row)
    return pd.DataFrame(rows)


def best_f1_threshold(y_true, y_score) -> tuple[float, float]:
    """Operating point that maximises F1, read off the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(
            (precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0
        )
    # precision_recall_curve returns one more point than thresholds.
    best = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best]), float(f1[best])


def curve_points(y_true, y_score) -> dict[str, np.ndarray]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}
