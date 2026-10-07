"""Stage 1: exploration and data quality assessment.

Prints the diagnostics the report needs and writes the corresponding figures and
tables to outputs/. Run this before run_experiment.py -- the preprocessing
choices in src/config.py are meant to be argued from what this produces.

    python run_eda.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, data, plots

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

SECTION = "=" * 78


def heading(title: str) -> None:
    print(f"\n{SECTION}\n{title}\n{SECTION}")


def attribute_overview(X: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    """Attribute typing plus the data-quality flags from Lecture 2."""
    rows = []
    for column in X.columns:
        series = X[column]
        is_numeric = column in numeric
        rows.append(
            {
                "attribute": column,
                "measurement_scale": "ratio/interval" if is_numeric else "nominal",
                "dtype": str(series.dtype),
                "n_unique": int(series.nunique(dropna=True)),
                "n_missing": int(series.isna().sum()),
                "pct_missing": round(float(series.isna().mean() * 100), 3),
                "dominant_level_share": round(
                    float(series.value_counts(normalize=True, dropna=True).iloc[0]), 3
                ),
            }
        )
    return pd.DataFrame(rows)


def placeholder_audit(X: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """Levels that look like a missing-value placeholder rather than a real category."""
    suspects = {"Unknown", "unknown", "?", "NA", "N/A", "None", "none", "Other", ""}
    rows = []
    for column in categorical:
        counts = X[column].astype(object).value_counts(dropna=True)
        for level, count in counts.items():
            if str(level).strip() in suspects:
                rows.append(
                    {
                        "attribute": column,
                        "level": level,
                        "count": int(count),
                        "pct": round(float(count / len(X) * 100), 3),
                    }
                )
    return pd.DataFrame(rows)


def skew_and_tails(X: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    """Location, spread, and how far the tail runs past the box."""
    rows = []
    for column in numeric:
        series = X[column].dropna()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        rows.append(
            {
                "attribute": column,
                "mean": round(float(series.mean()), 3),
                "median": round(float(series.median()), 3),
                "std": round(float(series.std()), 3),
                "min": round(float(series.min()), 3),
                "max": round(float(series.max()), 3),
                "pearson_skew": round(float(3 * (series.mean() - series.median()) / series.std()), 3),
                "pct_above_upper_fence": round(float((series > upper_fence).mean() * 100), 3),
                "pct_exactly_zero": round(float((series == 0).mean() * 100), 3),
            }
        )
    return pd.DataFrame(rows)


def zero_patterns(X: pd.DataFrame, y: np.ndarray, numeric: list[str]) -> pd.DataFrame:
    """Do zeros co-occur across attributes?

    A zero that shows up in the same rows across several attributes is more
    likely a placeholder for "not recorded" than a measured value, which changes
    whether it should be imputed or left alone.
    """
    zero_mask = X[numeric] == 0
    combinations = zero_mask.apply(
        lambda row: "+".join(sorted(col for col in numeric if row[col])), axis=1
    )
    frame = pd.DataFrame({"zero_attributes": combinations, "y": y})
    grouped = frame[frame["zero_attributes"] != ""].groupby("zero_attributes")["y"].agg(
        ["size", "mean"]
    )
    grouped = grouped.rename(columns={"size": "rows", "mean": "positive_rate"})
    grouped["pct_of_data"] = (grouped["rows"] / len(X) * 100).round(3)
    grouped["positive_rate"] = grouped["positive_rate"].round(3)
    return grouped.sort_values("rows", ascending=False).reset_index()


def redundancy(X: pd.DataFrame, numeric: list[str], threshold: float = 0.6) -> pd.DataFrame:
    """Attribute pairs correlated strongly enough to count as redundant."""
    corr = X[numeric].corr()
    rows = []
    for i, first in enumerate(numeric):
        for second in numeric[i + 1 :]:
            r = corr.loc[first, second]
            if abs(r) > threshold:
                rows.append({"attribute_a": first, "attribute_b": second, "pearson_r": round(float(r), 3)})
    return pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False)


def target_association(
    X: pd.DataFrame, y: np.ndarray, numeric: list[str], categorical: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_rows = (
        X[numeric]
        .corrwith(pd.Series(y, index=X.index))
        .round(3)
        .sort_values(key=abs, ascending=False)
        .rename_axis("attribute")
        .reset_index(name="point_biserial_r")
    )

    categorical_rows = []
    for column in categorical:
        frame = pd.DataFrame({"level": X[column].astype(object), "y": y})
        grouped = frame.groupby("level")["y"].agg(["mean", "size"])
        eligible = grouped[grouped["size"] >= 30]
        if len(eligible) > 1:
            categorical_rows.append(
                {
                    "attribute": column,
                    "n_levels": int(X[column].nunique()),
                    "n_levels_below_50_rows": int((grouped["size"] < 50).sum()),
                    "positive_rate_spread": round(
                        float(eligible["mean"].max() - eligible["mean"].min()), 3
                    ),
                }
            )
    categorical_frame = pd.DataFrame(categorical_rows).sort_values(
        "positive_rate_spread", ascending=False
    )
    return numeric_rows, categorical_frame


def main() -> None:
    config.ensure_output_dirs()
    dataset = data.prepare()
    X, y = dataset.X_train, dataset.y_train
    numeric, categorical = dataset.numeric_features, dataset.categorical_features

    heading("SHAPE AND CLASS BALANCE")
    print(f"raw shapes            : {dataset.report['raw_shapes']}")
    print(f"rows with no label    : train={dataset.report['train_unlabelled_rows_dropped']}, "
          f"test={dataset.report['test_unlabelled_rows_dropped']}  (dropped: an unlabelled "
          f"row can neither train nor score a classifier)")
    print(f"usable rows           : train={len(X)}, test={len(dataset.X_test)}")
    balance = dataset.report["class_balance"]
    print(f"P(label=yes)          : train={balance['train_positive_rate']:.4f}, "
          f"test={balance['test_positive_rate']:.4f}")
    print(f"majority-class accuracy: {1 - balance['test_positive_rate']:.4f}  "
          f"<- any model must beat this to be interesting")
    print(f"duplicate rows        : train={int(X.duplicated().sum())}, "
          f"test={int(dataset.X_test.duplicated().sum())}")

    heading("ATTRIBUTE TYPES AND DATA QUALITY")
    overview = attribute_overview(X, numeric)
    print(overview.to_string(index=False))

    heading("PLACEHOLDER LEVELS (missing values wearing a category label)")
    placeholders = placeholder_audit(X, categorical)
    print(placeholders.to_string(index=False) if len(placeholders) else "none found")

    heading("LOCATION, SPREAD, SKEW, TAIL MASS")
    stats = skew_and_tails(X, numeric)
    print(stats.to_string(index=False))

    heading("ZERO CO-OCCURRENCE PATTERNS (candidate coded missing values)")
    zeros = zero_patterns(X, y, numeric)
    print(zeros.to_string(index=False) if len(zeros) else "no zeros present")

    heading("REDUNDANT ATTRIBUTE PAIRS (|r| > 0.6)")
    redundant = redundancy(X, numeric)
    print(redundant.to_string(index=False) if len(redundant) else "none")

    heading("ASSOCIATION WITH THE CLASS LABEL")
    numeric_assoc, categorical_assoc = target_association(X, y, numeric, categorical)
    print("numeric attributes:")
    print(numeric_assoc.to_string(index=False))
    print("\nnominal attributes:")
    print(categorical_assoc.to_string(index=False))

    heading("FIGURES")
    written = [
        plots.class_balance(y, dataset.y_test),
        plots.missingness(X),
        plots.numeric_histograms(X, numeric),
        plots.numeric_boxplots_by_class(X, y, numeric),
        plots.correlation_matrix(X, numeric),
        plots.categorical_positive_rate(X, y, categorical),
    ]
    for path in written:
        print(f"  {path.relative_to(config.PROJECT_ROOT)}")

    heading("TABLES")
    tables = {
        "eda_attribute_overview": overview,
        "eda_placeholder_levels": placeholders,
        "eda_numeric_summary": stats,
        "eda_zero_patterns": zeros,
        "eda_redundant_pairs": redundant,
        "eda_numeric_association": numeric_assoc,
        "eda_categorical_association": categorical_assoc,
    }
    for name, frame in tables.items():
        path = config.TABLE_DIR / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"  {path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
