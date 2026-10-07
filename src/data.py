"""Loading, cleaning, and attribute typing.

Imputation deliberately lives in the model pipelines, not here, so that any
statistic used to fill a value is estimated on training folds only. What happens
here is limited to decisions that carry no information across the split:
normalising whitespace, re-labelling placeholder tokens, and dropping rows whose
class label is unknown (an unlabelled row cannot train or score a classifier).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .config import CleaningConfig


@dataclass
class Dataset:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    numeric_features: list[str]
    categorical_features: list[str]
    report: dict[str, object]

    @property
    def feature_names(self) -> list[str]:
        return self.numeric_features + self.categorical_features


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(config.TRAIN_CSV), pd.read_csv(config.TEST_CSV)


def attribute_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into ratio/interval (numeric) and nominal (categorical)."""
    features = [c for c in df.columns if c != config.TARGET]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def _strip_strings(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    for col in categorical:
        df[col] = df[col].astype(object).where(df[col].isna(), df[col].astype(str).str.strip())
    return df


def _blank_out_placeholders(
    df: pd.DataFrame, categorical: list[str], cfg: CleaningConfig
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert placeholder tokens such as 'Unknown' into genuine NaN."""
    hits: dict[str, int] = {}
    if not cfg.unknown_as_missing:
        return df, hits
    for col in categorical:
        mask = df[col].isin(cfg.unknown_tokens)
        if mask.any():
            hits[col] = int(mask.sum())
            df.loc[mask, col] = np.nan
    return df, hits


def _blank_out_zeros(
    df: pd.DataFrame, cfg: CleaningConfig
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert exact zeros to NaN for the columns nominated in the config."""
    hits: dict[str, int] = {}
    for col in cfg.zero_as_missing:
        if col not in df.columns:
            continue
        mask = df[col] == 0
        if mask.any():
            hits[col] = int(mask.sum())
            df.loc[mask, col] = np.nan
    return df, hits


def prepare(cfg: CleaningConfig | None = None) -> Dataset:
    cfg = cfg or config.DEFAULT_CLEANING
    train_raw, test_raw = load_raw()
    report: dict[str, object] = {
        "raw_shapes": {"train": train_raw.shape, "test": test_raw.shape},
    }

    numeric, categorical = attribute_types(train_raw)

    frames = {}
    for name, raw in (("train", train_raw), ("test", test_raw)):
        df = raw.copy()
        df = _strip_strings(df, categorical + [config.TARGET])
        df, placeholder_hits = _blank_out_placeholders(df, categorical, cfg)
        df, zero_hits = _blank_out_zeros(df, cfg)

        unlabelled = int(df[config.TARGET].isna().sum())
        df = df[df[config.TARGET].notna()].reset_index(drop=True)

        report[f"{name}_unlabelled_rows_dropped"] = unlabelled
        report[f"{name}_placeholders_blanked"] = placeholder_hits
        report[f"{name}_zeros_blanked"] = zero_hits
        frames[name] = df

    train, test = frames["train"], frames["test"]

    if cfg.drop_features:
        keep = [c for c in train.columns if c not in cfg.drop_features]
        train, test = train[keep], test[keep]
        numeric = [c for c in numeric if c not in cfg.drop_features]
        categorical = [c for c in categorical if c not in cfg.drop_features]
        report["dropped_features"] = list(cfg.drop_features)

    if cfg.drop_incomplete_train_rows:
        before = len(train)
        train = train.dropna().reset_index(drop=True)
        report["train_incomplete_rows_dropped"] = before - len(train)

    y_train = _encode_target(train[config.TARGET])
    y_test = _encode_target(test[config.TARGET])

    X_train = train[numeric + categorical].copy()
    X_test = test[numeric + categorical].copy()
    for col in categorical:
        X_train[col] = X_train[col].astype(object)
        X_test[col] = X_test[col].astype(object)

    report["class_balance"] = {
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
    }
    report["missing_per_feature_train"] = X_train.isna().sum().to_dict()

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        numeric_features=numeric,
        categorical_features=categorical,
        report=report,
    )


def _encode_target(series: pd.Series) -> np.ndarray:
    """Positive class -> 1 so that precision/recall/F1 describe the minority class."""
    return (series.astype(str).str.strip() == config.POSITIVE_CLASS).astype(int).to_numpy()
