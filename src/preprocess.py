"""Preprocessing blocks, one profile per model family.

Lecture 2 makes the point that preprocessing is model-dependent: tree learners
need almost none, distance-based learners need scaling and encoding, and a naive
Bayes over continuous attributes needs either binning or a distributional
assumption. Each builder below encodes one of those positions so the choice is
visible rather than buried in a single monolithic transformer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)

from .config import CleaningConfig


class RareCategoryEncoder(BaseEstimator, TransformerMixin):
    """Map nominal levels to integer codes, with code 0 reserved.

    Code 0 absorbs both levels that were too infrequent to estimate anything
    from and levels never seen during fitting; real levels take 1..k. Reserving
    the *lowest* code rather than the highest matters for the categorical naive
    Bayes: it sizes its count table from the largest code it sees while fitting,
    so a reserved top code that happens to be unused in a training fold would
    leave the table too small for a later unseen level. Nothing is guaranteed to
    occupy code 0, and nothing needs to.
    """

    def __init__(self, min_count: int = 50, rare_label: str = "Rare"):
        self.min_count = min_count
        self.rare_label = rare_label

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.columns_ = list(frame.columns)
        self.mapping_ = {}
        self.levels_ = {}
        for col in self.columns_:
            counts = frame[col].astype(object).value_counts(dropna=True)
            kept = sorted(
                (str(value) for value, count in counts.items() if count >= self.min_count)
            )
            self.mapping_[col] = {value: code for code, value in enumerate(kept, start=1)}
            self.levels_[col] = [self.rare_label, *kept]
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        encoded = {}
        for col in self.columns_:
            mapping = self.mapping_[col]
            encoded[col] = (
                frame[col].astype(object).map(lambda value: mapping.get(str(value), 0)).astype(int)
            )
        return pd.DataFrame(encoded, index=frame.index)

    def n_categories(self) -> dict[str, int]:
        return {col: len(levels) for col, levels in self.levels_.items()}

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Collapse infrequent nominal levels into a single string bucket.

    Used ahead of one-hot encoding, where an unseen level can be safely ignored
    and integer codes would be meaningless anyway.
    """

    def __init__(self, min_count: int = 50, rare_label: str = "Rare"):
        self.min_count = min_count
        self.rare_label = rare_label

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.columns_ = list(frame.columns)
        self.vocabulary_ = {}
        for col in self.columns_:
            counts = frame[col].astype(object).value_counts(dropna=True)
            self.vocabulary_[col] = {
                value for value, count in counts.items() if count >= self.min_count
            }
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        for col in self.columns_:
            values = frame[col].astype(object)
            frame[col] = values.where(values.isin(self.vocabulary_[col]), self.rare_label)
        return frame

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def _to_int(X):
    """CategoricalNB expects non-negative integer codes."""
    return np.asarray(X, dtype=float).astype(int)


class NumericTransform(BaseEstimator, TransformerMixin):
    """Optional monotone attribute transformation, as a tunable step.

    `log1p` is the log transform from Lecture 2, applied to pull in the right
    tails that make a per-class normal assumption hard to defend. The shift that
    keeps the logarithm finite is learned during fitting, so test rows are
    transformed with the training shift rather than their own minimum.
    """

    def __init__(self, kind: str = "identity"):
        self.kind = kind

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.columns_ = list(frame.columns)
        self.shift_ = frame.astype(float).min(axis=0)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)[self.columns_].astype(float)
        if self.kind == "identity":
            return frame
        if self.kind == "log1p":
            return np.log1p((frame - self.shift_).clip(lower=0.0))
        raise ValueError(f"unknown transform kind: {self.kind!r}")

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def categorical_imputer(cfg: CleaningConfig) -> SimpleImputer:
    if cfg.categorical_missing == "constant":
        return SimpleImputer(strategy="constant", fill_value=cfg.categorical_missing_label)
    return SimpleImputer(strategy=cfg.categorical_missing)


def categorical_branch(cfg: CleaningConfig) -> Pipeline:
    branch = Pipeline(
        [
            ("impute", categorical_imputer(cfg)),
            (
                "encode",
                RareCategoryEncoder(
                    min_count=cfg.rare_category_min_count, rare_label=cfg.rare_category_label
                ),
            ),
        ]
    )
    return branch.set_output(transform="pandas")


def numeric_branch(cfg: CleaningConfig, *, tunable_transform: bool = False) -> Pipeline:
    steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy=cfg.numeric_missing))]
    if tunable_transform:
        steps.append(("transform", NumericTransform()))
    branch = Pipeline(steps)
    return branch.set_output(transform="pandas")


def build_tree_preprocessor(
    numeric: list[str], categorical: list[str], cfg: CleaningConfig
) -> ColumnTransformer:
    """Trees split on one attribute at a time, so monotone rescaling is pointless.

    Nominal levels get integer codes rather than one-hot columns: a CART
    implementation only ever tests `code <= threshold`, and one-hot would force
    every split to be one-level-versus-rest.
    """
    ct = ColumnTransformer(
        [
            ("num", numeric_branch(cfg), numeric),
            ("cat", categorical_branch(cfg), categorical),
        ],
        verbose_feature_names_out=False,
    )
    return ct.set_output(transform="pandas")


def build_discretised_preprocessor(
    numeric: list[str],
    categorical: list[str],
    cfg: CleaningConfig,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> ColumnTransformer:
    """Everything becomes a discrete attribute, so one categorical NB covers all of it.

    This is the first of the two options Lecture 5 gives for continuous
    attributes under naive Bayes: discretise the range into bins.
    """
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy=cfg.numeric_missing)),
            (
                "bin",
                KBinsDiscretizer(
                    n_bins=n_bins, encode="ordinal", strategy=strategy, quantile_method="linear"
                ),
            ),
        ]
    ).set_output(transform="pandas")

    ct = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_branch(cfg), categorical),
        ],
        verbose_feature_names_out=False,
    )
    return ct.set_output(transform="pandas")


def build_mixed_nb_preprocessor(
    numeric: list[str], categorical: list[str], cfg: CleaningConfig
) -> ColumnTransformer:
    """Numeric attributes stay continuous for a Gaussian likelihood.

    The second option from Lecture 5: assume a normal distribution per class and
    estimate its parameters. Numeric columns come first so the classifier knows
    where the continuous block ends.
    """
    ct = ColumnTransformer(
        [
            ("num", numeric_branch(cfg, tunable_transform=True), numeric),
            ("cat", categorical_branch(cfg), categorical),
        ],
        verbose_feature_names_out=False,
    )
    return ct.set_output(transform="pandas")


def build_distance_preprocessor(
    numeric: list[str], categorical: list[str], cfg: CleaningConfig
) -> ColumnTransformer:
    """Standardise and one-hot encode, for the distance-based models kept in reserve.

    Lecture 2 is explicit that an unscaled attribute with a wide range dominates
    a Euclidean distance, and that nominal attributes have to be encoded before
    a distance is even defined.
    """
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy=cfg.numeric_missing)),
            ("scale", StandardScaler()),
        ]
    ).set_output(transform="pandas")

    categorical_pipe = Pipeline(
        [
            ("impute", categorical_imputer(cfg)),
            (
                "group_rare",
                RareCategoryGrouper(
                    min_count=cfg.rare_category_min_count, rare_label=cfg.rare_category_label
                ),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    ).set_output(transform="pandas")

    ct = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        verbose_feature_names_out=False,
    )
    return ct.set_output(transform="pandas")
