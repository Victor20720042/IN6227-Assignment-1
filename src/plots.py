"""Figures, using the visualisation vocabulary from Lecture 3 plus the two
diagnostic curves from Lecture 4 (complexity vs error, learning curve).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

FIG_DPI = 160


def _save(fig, name: str) -> Path:
    config.ensure_output_dirs()
    path = config.FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def class_balance(y_train: np.ndarray, y_test: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    rates = pd.DataFrame(
        {
            "train": [1 - y_train.mean(), y_train.mean()],
            "test": [1 - y_test.mean(), y_test.mean()],
        },
        index=["no", "yes"],
    )
    rates.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"], edgecolor="black", linewidth=0.4)
    ax.set_ylabel("proportion of rows")
    ax.set_xlabel("class label")
    ax.set_title("Class balance")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=1)
    ax.set_ylim(0, 0.9)
    ax.tick_params(axis="x", rotation=0)
    return _save(fig, "eda_class_balance.png")


def numeric_histograms(X: pd.DataFrame, numeric: list[str]) -> Path:
    n = len(numeric)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.3 * rows))
    for ax, column in zip(np.ravel(axes), numeric):
        values = X[column].dropna()
        ax.hist(values, bins=40, color="#4c72b0", edgecolor="white", linewidth=0.3)
        ax.axvline(values.mean(), color="#c44e52", linewidth=1.0, label="mean")
        ax.axvline(values.median(), color="#55a868", linewidth=1.0, label="median")
        ax.set_title(column, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    handles, labels = np.ravel(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8)
    fig.suptitle("Numeric attribute distributions (mean vs median shows the skew)", fontsize=10)
    return _save(fig, "eda_numeric_histograms.png")


def numeric_boxplots_by_class(
    X: pd.DataFrame, y: np.ndarray, numeric: list[str]
) -> Path:
    n = len(numeric)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
    for ax, column in zip(np.ravel(axes), numeric):
        groups = [X.loc[y == 0, column].dropna(), X.loc[y == 1, column].dropna()]
        bp = ax.boxplot(groups, tick_labels=["no", "yes"], patch_artist=True, widths=0.55)
        for patch, colour in zip(bp["boxes"], ["#4c72b0", "#dd8452"]):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)
        ax.set_title(column, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle("Numeric attributes by class", fontsize=10)
    return _save(fig, "eda_numeric_boxplots.png")


def correlation_matrix(X: pd.DataFrame, numeric: list[str]) -> Path:
    corr = X[numeric].corr()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric)), numeric, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(numeric)), numeric, fontsize=8)
    for i in range(len(numeric)):
        for j in range(len(numeric)):
            value = corr.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) > 0.55 else "black",
            )
    fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Numeric attribute correlation", fontsize=10)
    return _save(fig, "eda_correlation_matrix.png")


def categorical_positive_rate(
    X: pd.DataFrame, y: np.ndarray, categorical: list[str], min_count: int = 30
) -> Path:
    n = len(categorical)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.6 * rows))
    base_rate = y.mean()
    for ax, column in zip(np.ravel(axes), categorical):
        frame = pd.DataFrame({"level": X[column].astype(object), "y": y})
        grouped = frame.groupby("level")["y"].agg(["mean", "size"])
        grouped = grouped[grouped["size"] >= min_count].sort_values("mean")
        ax.barh(grouped.index.astype(str), grouped["mean"], color="#4c72b0")
        ax.axvline(base_rate, color="#c44e52", linestyle="--", linewidth=1.0)
        ax.set_title(f"{column} ({X[column].nunique()} levels)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 1)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle(
        f"P(label = yes) per level; dashed line is the base rate {base_rate:.3f}", fontsize=10
    )
    return _save(fig, "eda_categorical_positive_rate.png")


def missingness(X: pd.DataFrame) -> Path:
    counts = X.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(counts.index.astype(str), counts.to_numpy(), color="#937860")
    ax.set_xticks(range(len(counts)), counts.index.astype(str), rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("missing rows")
    ax.set_title("Explicit missing values per attribute (training set)", fontsize=10)
    ax.bar_label(ax.containers[0], fontsize=7)
    return _save(fig, "eda_missingness.png")


def confusion_heatmaps(matrices: dict[str, pd.DataFrame]) -> Path:
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.2))
    axes = np.atleast_1d(axes)
    for ax, (name, frame) in zip(axes, matrices.items()):
        ax.imshow(frame.to_numpy(), cmap="Blues")
        ax.set_xticks(range(frame.shape[1]), frame.columns, fontsize=8)
        ax.set_yticks(range(frame.shape[0]), frame.index, fontsize=8, rotation=90, va="center")
        total = frame.to_numpy().sum()
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                count = frame.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{count}\n{count / total:.1%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if count > total * 0.35 else "black",
                )
        ax.set_title(name, fontsize=9)
    fig.suptitle("Confusion matrices on the held-out test set", fontsize=10)
    return _save(fig, "results_confusion_matrices.png")


def roc_and_pr(curves: dict[str, dict[str, np.ndarray]], base_rate: float) -> Path:
    with plt.rc_context(
        {
            "font.family": "Times New Roman",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    ):
        fig, (ax_roc, ax_pr) = plt.subplots(2, 1, figsize=(4.6, 7.2))
        for name, data in curves.items():
            ax_roc.plot(data["fpr"], data["tpr"], linewidth=1.4, label=name)
            ax_pr.plot(data["recall"], data["precision"], linewidth=1.4, label=name)

        ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random guessing")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC")
        ax_roc.legend(frameon=False)
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)

        ax_pr.axhline(base_rate, color="black", linestyle="--", linewidth=0.8, label="base rate")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall")
        ax_pr.legend(frameon=False)
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1)
        return _save(fig, "results_roc_pr.png")


def complexity_curve(
    frame: pd.DataFrame,
    x_column: str,
    title: str,
    filename: str,
    tick_label_column: str | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(frame[x_column], frame["train_error"], "o-", label="training error", linewidth=1.3)
    ax.plot(frame[x_column], frame["cv_error"], "s-", label="cross-validation error", linewidth=1.3)
    if tick_label_column is not None:
        ax.set_xticks(frame[x_column], frame[tick_label_column], fontsize=8)
    ax.set_xlabel(x_column)
    ax.set_ylabel("error rate")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, filename)


def learning_curves(frames: dict[str, pd.DataFrame]) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for name, frame in frames.items():
        ax.plot(frame["train_size"], frame["cv_f1"], "o-", linewidth=1.3, label=name)
    ax.set_xlabel("training rows used")
    ax.set_ylabel("cross-validated F1")
    ax.set_title("Learning curves", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, "results_learning_curves.png")
