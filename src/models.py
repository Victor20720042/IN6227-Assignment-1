"""Model registry.

Every classifier is described by the pipeline it needs and the grid it is tuned
over, so switching the comparison to another pair is a matter of changing which
keys are enabled -- no other file has to change. Decision tree and naive Bayes
are enabled; the ensemble and distance-based entries are wired up and left off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import CleaningConfig, RANDOM_STATE
from .preprocess import (
    _to_int,
    build_discretised_preprocessor,
    build_distance_preprocessor,
    build_mixed_nb_preprocessor,
    build_tree_preprocessor,
)


class MixedNaiveBayes(BaseEstimator, ClassifierMixin):
    """Naive Bayes over a continuous block and a discrete block.

    Conditional independence given the class is what makes this legal: the joint
    likelihood is a product over attributes, so a Gaussian density can supply the
    factors for the numeric attributes while multinomial counts supply the rest.
    The two sub-models each return log P(y) + sum log P(x_i | y), so one copy of
    the class log-prior is subtracted to avoid counting it twice.

    Parameters
    ----------
    n_numeric:
        Number of leading columns treated as continuous. The preprocessor places
        the numeric block first.
    alpha:
        Laplace / Lidstone smoothing for the categorical block. Without it a
        level that never co-occurs with a class in training drives the whole
        product to zero.
    """

    def __init__(self, n_numeric: int, alpha: float = 1.0, var_smoothing: float = 1e-9):
        self.n_numeric = n_numeric
        self.alpha = alpha
        self.var_smoothing = var_smoothing

    def _split_blocks(self, X):
        arr = np.asarray(X, dtype=float)
        return arr[:, : self.n_numeric], arr[:, self.n_numeric :]

    def fit(self, X, y):
        numeric_block, categorical_block = self._split_blocks(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        self.gaussian_ = None
        if numeric_block.shape[1]:
            self.gaussian_ = GaussianNB(var_smoothing=self.var_smoothing).fit(numeric_block, y)

        self.categorical_ = None
        if categorical_block.shape[1]:
            self.categorical_ = CategoricalNB(alpha=self.alpha, force_alpha=True).fit(
                categorical_block.astype(int), y
            )

        counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        self.class_log_prior_ = np.log(counts / counts.sum())
        return self

    def _joint_log_likelihood(self, X):
        numeric_block, categorical_block = self._split_blocks(X)
        total = np.tile(self.class_log_prior_, (X.shape[0], 1))
        if self.gaussian_ is not None:
            total = total + (
                self.gaussian_._joint_log_likelihood(numeric_block) - self.class_log_prior_
            )
        if self.categorical_ is not None:
            total = total + (
                self.categorical_._joint_log_likelihood(categorical_block.astype(int))
                - self.class_log_prior_
            )
        return total

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        log_norm = np.logaddexp.reduce(jll, axis=1, keepdims=True)
        return jll - log_norm

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]


@dataclass
class ModelSpec:
    key: str
    label: str
    build: Callable[[list[str], list[str], CleaningConfig], Pipeline]
    param_grid: dict[str, list] = field(default_factory=dict)
    enabled: bool = False
    lecture_note: str = ""


def _decision_tree(numeric, categorical, cfg) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_tree_preprocessor(numeric, categorical, cfg)),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )


def _categorical_nb(numeric, categorical, cfg) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_discretised_preprocessor(numeric, categorical, cfg)),
            ("to_int", FunctionTransformer(_to_int)),
            ("clf", CategoricalNB(force_alpha=True)),
        ]
    )


def _mixed_nb(numeric, categorical, cfg) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_mixed_nb_preprocessor(numeric, categorical, cfg)),
            ("clf", MixedNaiveBayes(n_numeric=len(numeric))),
        ]
    )


def _random_forest(numeric, categorical, cfg) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_tree_preprocessor(numeric, categorical, cfg)),
            (
                "clf",
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            ),
        ]
    )


def _knn(numeric, categorical, cfg) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_distance_preprocessor(numeric, categorical, cfg)),
            ("clf", KNeighborsClassifier(n_jobs=-1)),
        ]
    )


def _linear_svm(numeric, categorical, cfg) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_distance_preprocessor(numeric, categorical, cfg)),
            ("clf", LinearSVC(random_state=RANDOM_STATE, dual="auto")),
        ]
    )


def _rbf_svm(numeric, categorical, cfg) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_distance_preprocessor(numeric, categorical, cfg)),
            ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE)),
        ]
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "decision_tree": ModelSpec(
        key="decision_tree",
        label="Decision Tree (CART)",
        build=_decision_tree,
        param_grid={
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [None, 6, 10, 14],
            "clf__min_samples_leaf": [1, 20, 100],
            "clf__ccp_alpha": [0.0, 1e-4, 1e-3],
            "clf__class_weight": [None, "balanced"],
        },
        enabled=True,
        lecture_note=(
            "Lecture 4: Gini vs entropy impurity; max_depth/min_samples_leaf are "
            "pre-pruning stopping rules, ccp_alpha is cost-complexity post-pruning; "
            "class_weight is the cost-matrix lever."
        ),
    ),
    "naive_bayes_binned": ModelSpec(
        key="naive_bayes_binned",
        label="Naive Bayes (discretised)",
        build=_categorical_nb,
        param_grid={
            "prep__num__bin__n_bins": [5, 10, 20],
            "prep__num__bin__strategy": ["quantile", "uniform", "kmeans"],
            "clf__alpha": [0.01, 0.1, 1.0, 10.0],
        },
        enabled=True,
        lecture_note=(
            "Lecture 5 option 1 for continuous attributes: discretise into bins. "
            "Bin strategies are the equal-width / equal-frequency / k-means methods "
            "from Lecture 2; alpha is Laplace smoothing against zero probabilities."
        ),
    ),
    "naive_bayes_gaussian": ModelSpec(
        key="naive_bayes_gaussian",
        label="Naive Bayes (Gaussian + categorical)",
        build=_mixed_nb,
        param_grid={
            "prep__num__transform__kind": ["identity", "log1p"],
            "clf__alpha": [0.01, 0.1, 1.0, 10.0],
        },
        enabled=True,
        lecture_note=(
            "Lecture 5 option 2: assume a normal distribution per class for the "
            "continuous attributes. log1p tests whether correcting right skew "
            "makes that assumption more defensible."
        ),
    ),
    # --- Kept wired up but disabled. Enable with --models on the command line. ---
    "random_forest": ModelSpec(
        key="random_forest",
        label="Random Forest",
        build=_random_forest,
        param_grid={
            "clf__n_estimators": [200, 400],
            "clf__max_features": ["sqrt", 0.5],
            "clf__min_samples_leaf": [1, 5, 20],
            "clf__class_weight": [None, "balanced"],
        },
        lecture_note=(
            "Lecture 5: bagging plus random attribute subsets. Bagging pays off "
            "when the base learner is unstable, which an unpruned tree is."
        ),
    ),
    "knn": ModelSpec(
        key="knn",
        label="k-Nearest Neighbours",
        build=_knn,
        param_grid={
            "clf__n_neighbors": [5, 15, 31],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
        lecture_note=(
            "Lecture 5: lazy learner; needs the standardisation and encoding from "
            "Lecture 2, and one-hot encoding pushes it into the sparse regime "
            "where the curse of dimensionality bites."
        ),
    ),
    "linear_svm": ModelSpec(
        key="linear_svm",
        label="Linear SVM",
        build=_linear_svm,
        param_grid={
            "clf__C": [0.01, 0.1, 1.0],
            "clf__class_weight": [None, "balanced"],
        },
        lecture_note="Lecture 5: maximum-margin hyperplane; C trades margin against slack.",
    ),
    "rbf_svm": ModelSpec(
        key="rbf_svm",
        label="RBF SVM",
        build=_rbf_svm,
        param_grid={
            "clf__C": [1.0],
            "clf__gamma": ["scale"],
        },
        lecture_note=(
            "Lecture 5: kernel trick for non-linear boundaries. Training is "
            "roughly quadratic in the sample count, so expect this to be slow on "
            "30k rows -- subsample before enabling it."
        ),
    ),
    # A rule-based learner (RIPPER, Lecture 5) has no scikit-learn implementation.
    # It would need an extra dependency such as `wittgenstein`; left out on purpose
    # rather than approximated by something that is not actually RIPPER.
}


def enabled_models() -> list[str]:
    return [key for key, spec in MODEL_REGISTRY.items() if spec.enabled]
