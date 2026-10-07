"""Stage 3: does each cleaning decision actually matter?

Several choices in the cleaning stage have a defensible argument on both sides.
Rather than pick one and assert it, this script re-runs the whole procedure --
including hyperparameter selection -- under each alternative and reports what
changed. A decision that moves nothing is worth recording as such; a decision
that moves the result needs the argument spelled out in the report.

    python run_sensitivity.py
    python run_sensitivity.py --quick --scenarios baseline drop_index_weight

The yardstick for "did anything change" is the cross-validation standard
deviation of the baseline: a shift smaller than the noise across folds is not
evidence of anything.
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config, data
from src.config import CleaningConfig
from src.experiment import tune_and_fit
from src.models import MODEL_REGISTRY

MODELS = ["decision_tree", "naive_bayes_binned"]

# Each scenario is one cleaning decision flipped away from the baseline, with the
# question it is meant to answer.
SCENARIOS: dict[str, tuple[CleaningConfig, str]] = {
    "baseline": (
        CleaningConfig(),
        "current defaults: zeros kept as values, 'Unknown' kept as a level, nothing dropped",
    ),
    "drop_index_weight": (
        CleaningConfig(drop_features=("index_weight",)),
        "index_weight correlates -0.008 with the label and reads like a sampling weight",
    ),
    "drop_collinear": (
        CleaningConfig(drop_features=("stability_index", "composite_rank")),
        "three numeric attributes correlate above 0.83; keep one, drop the other two",
    ),
    # Blanking 'Unknown' and then filling with a fresh constant label only renames
    # the level, so it is kept here as the control that shows the rename changes
    # nothing. The two scenarios after it are the ones that actually remove the
    # group: merge it into the majority level, or drop the rows outright.
    "unknown_relabelled": (
        CleaningConfig(unknown_as_missing=True),
        "control: 'Unknown' -> NaN -> refilled as its own 'Missing' level (a rename)",
    ),
    "unknown_to_mode": (
        CleaningConfig(unknown_as_missing=True, categorical_missing="most_frequent"),
        "merge 'Unknown' into the most frequent level, destroying it as a group",
    ),
    "unknown_rows_dropped": (
        CleaningConfig(unknown_as_missing=True, drop_incomplete_train_rows=True),
        "eliminate the training rows whose level was 'Unknown'",
    ),
    "zeros_as_missing": (
        CleaningConfig(
            zero_as_missing=(
                "performance_score",
                "stability_index",
                "activity_duration",
                "load_ratio",
            )
        ),
        "the co-occurring exact zeros read as a code for 'not recorded'",
    ),
    "drop_incomplete_rows": (
        CleaningConfig(drop_incomplete_train_rows=True),
        "eliminate training rows with any missing value instead of imputing",
    ),
    "rare_min_5": (
        CleaningConfig(rare_category_min_count=5),
        "barely group rare levels: almost every level kept separate",
    ),
    "rare_min_500": (
        CleaningConfig(rare_category_min_count=500),
        "group aggressively: only high-frequency levels survive",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scenarios", nargs="+", default=None, choices=sorted(SCENARIOS))
    parser.add_argument("--models", nargs="+", default=MODELS, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--quick", action="store_true", help="use reduced parameter grids")
    parser.add_argument("--folds", type=int, default=config.CV_FOLDS)
    return parser.parse_args()


def describe_effect(cfg: CleaningConfig, dataset) -> str:
    """One line on how much data the scenario actually touched."""
    parts = []
    blanked = dataset.report.get("train_placeholders_blanked") or {}
    if blanked:
        parts.append(f"placeholders->NaN: {sum(blanked.values())}")
    zeros = dataset.report.get("train_zeros_blanked") or {}
    if zeros:
        parts.append(f"zeros->NaN: {sum(zeros.values())}")
    dropped_rows = dataset.report.get("train_incomplete_rows_dropped")
    if dropped_rows:
        parts.append(f"rows dropped: {dropped_rows}")
    if cfg.drop_features:
        parts.append(f"features dropped: {', '.join(cfg.drop_features)}")
    parts.append(f"train rows: {len(dataset.X_train)}")
    parts.append(f"features: {len(dataset.feature_names)}")
    return "; ".join(parts)


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()
    names = args.scenarios or list(SCENARIOS)
    if "baseline" not in names:
        names = ["baseline", *names]

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=config.RANDOM_STATE)
    rows = []

    for name in names:
        cfg, question = SCENARIOS[name]
        dataset = data.prepare(cfg)
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        print(f"  rationale: {question}")
        print(f"  effect   : {describe_effect(cfg, dataset)}")

        for key in args.models:
            spec = MODEL_REGISTRY[key]
            fitted = tune_and_fit(
                spec,
                dataset,
                cfg,
                cv,
                args.quick,
                save_cv_results=False,
            )
            rows.append(
                {
                    "scenario": name,
                    "model": spec.label,
                    "n_train": len(dataset.X_train),
                    "n_features": len(dataset.feature_names),
                    "cv_f1": fitted.cv_score,
                    "cv_f1_std": fitted.cv_std,
                    "accuracy": fitted.metrics.accuracy,
                    "f1": fitted.metrics.f1,
                    "precision": fitted.metrics.precision,
                    "recall": fitted.metrics.recall,
                    "roc_auc": fitted.metrics.roc_auc,
                    "pr_auc": fitted.metrics.pr_auc,
                    "best_params": fitted.best_params,
                }
            )
            print(
                f"    {spec.label:28s} cv_f1={fitted.cv_score:.4f}  "
                f"test f1={fitted.metrics.f1:.4f}  acc={fitted.metrics.accuracy:.4f}  "
                f"pr_auc={fitted.metrics.pr_auc:.4f}"
            )

    frame = pd.DataFrame(rows)
    report = build_comparison(frame)

    print(f"\n{'=' * 78}\nSENSITIVITY TO CLEANING DECISIONS\n{'=' * 78}")
    for model, block in report.groupby("model", sort=False):
        noise = float(block.loc[block["scenario"] == "baseline", "cv_f1_std"].iloc[0])
        print(f"\n{model}   (baseline fold-to-fold CV F1 sd = {noise:.4f})")
        columns = [
            "scenario",
            "n_train",
            "cv_f1",
            "d_cv_f1",
            "f1",
            "d_f1",
            "accuracy",
            "d_accuracy",
            "pr_auc",
            "d_pr_auc",
            "verdict",
        ]
        print(block[columns].to_string(index=False))

    report.to_csv(config.TABLE_DIR / "sensitivity_summary.csv", index=False)
    frame.to_csv(config.TABLE_DIR / "sensitivity_raw.csv", index=False)
    print(f"\nwritten to {(config.TABLE_DIR / 'sensitivity_summary.csv').relative_to(config.PROJECT_ROOT)}")


def build_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Deltas against the baseline, judged against fold-to-fold noise.

    A scenario that removes rows is not comparable on cross-validation score: the
    folds are drawn from a different population, so a rise can mean the
    validation rows got easier rather than the model got better. Those scenarios
    are flagged, and the held-out test set -- identical across every scenario --
    is the only common yardstick for them.
    """
    output = []
    for model, block in frame.groupby("model", sort=False):
        base = block[block["scenario"] == "baseline"].iloc[0]
        noise = float(base["cv_f1_std"])
        for _, row in block.iterrows():
            deltas = {
                "d_cv_f1": row["cv_f1"] - base["cv_f1"],
                "d_f1": row["f1"] - base["f1"],
                "d_accuracy": row["accuracy"] - base["accuracy"],
                "d_pr_auc": row["pr_auc"] - base["pr_auc"],
            }
            if row["scenario"] == "baseline":
                verdict = "-"
            elif row["n_train"] != base["n_train"]:
                verdict = "cv not comparable (rows removed)"
            elif abs(deltas["d_cv_f1"]) < noise:
                verdict = "within fold noise"
            else:
                verdict = "better" if deltas["d_cv_f1"] > 0 else "worse"
            output.append(
                {
                    "model": model,
                    "scenario": row["scenario"],
                    "n_train": int(row["n_train"]),
                    "n_features": int(row["n_features"]),
                    "cv_f1": round(row["cv_f1"], 4),
                    "cv_f1_std": round(row["cv_f1_std"], 4),
                    "f1": round(row["f1"], 4),
                    "accuracy": round(row["accuracy"], 4),
                    "pr_auc": round(row["pr_auc"], 4),
                    **{k: round(v, 4) for k, v in deltas.items()},
                    "verdict": verdict,
                    "best_params": row["best_params"],
                }
            )
    return pd.DataFrame(output)


if __name__ == "__main__":
    main()
