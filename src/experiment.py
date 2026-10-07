"""Shared tuning and scoring machinery.

Hyperparameters are chosen by stratified k-fold cross-validation on the training
file; the held-out file is scored once per fitted model. Both run_experiment.py
and run_sensitivity.py go through here so that a cleaning comparison and a model
comparison are measured the same way.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from . import config, evaluate
from .config import CleaningConfig
from .models import ModelSpec

SCORING = "f1"

QUICK_GRIDS: dict[str, dict[str, list]] = {
    "decision_tree": {
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [6, 10, None],
        "clf__min_samples_leaf": [1, 50],
        "clf__class_weight": [None, "balanced"],
    },
    "naive_bayes_binned": {
        "prep__num__bin__n_bins": [5, 10],
        "prep__num__bin__strategy": ["quantile", "uniform"],
        "clf__alpha": [0.1, 1.0],
    },
    "naive_bayes_gaussian": {
        "prep__num__transform__kind": ["identity", "log1p"],
        "clf__alpha": [0.1, 1.0],
    },
}


@dataclass
class Fitted:
    key: str
    label: str
    estimator: object
    best_params: dict
    cv_score: float
    cv_std: float
    fit_seconds: float
    metrics: evaluate.Metrics
    confusion: pd.DataFrame
    y_score: np.ndarray | None = None
    curves: dict = field(default_factory=dict)


def positive_scores(estimator, X) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return None


def tune_and_fit(
    spec: ModelSpec,
    dataset,
    cfg: CleaningConfig,
    cv,
    quick: bool = False,
    *,
    save_cv_results: bool = True,
    cv_results_suffix: str = "",
) -> Fitted:
    grid = QUICK_GRIDS.get(spec.key, spec.param_grid) if quick else spec.param_grid
    pipeline = spec.build(dataset.numeric_features, dataset.categorical_features, cfg)

    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    started = time.perf_counter()
    search.fit(dataset.X_train, dataset.y_train)
    elapsed = time.perf_counter() - started

    estimator = search.best_estimator_
    y_pred = estimator.predict(dataset.X_test)
    y_score = positive_scores(estimator, dataset.X_test)

    metrics = evaluate.compute_metrics(dataset.y_test, y_pred, y_score)
    confusion = evaluate.confusion_frame(dataset.y_test, y_pred)
    curves = evaluate.curve_points(dataset.y_test, y_score) if y_score is not None else {}

    if save_cv_results:
        config.ensure_output_dirs()
        name = f"cv_results_{spec.key}{cv_results_suffix}.csv"
        pd.DataFrame(search.cv_results_).to_csv(config.TABLE_DIR / name, index=False)

    return Fitted(
        key=spec.key,
        label=spec.label,
        estimator=estimator,
        best_params=search.best_params_,
        cv_score=float(search.best_score_),
        cv_std=float(search.cv_results_["std_test_score"][search.best_index_]),
        fit_seconds=elapsed,
        metrics=metrics,
        confusion=confusion,
        y_score=y_score,
        curves=curves,
    )
