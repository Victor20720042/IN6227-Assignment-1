"""Stage 2: tuning, held-out evaluation, and model comparison.

    python run_experiment.py                      # decision tree vs the two naive Bayes variants
    python run_experiment.py --models decision_tree random_forest
    python run_experiment.py --quick              # smaller grids while iterating
    python run_experiment.py --unknown-as-missing # sensitivity check on the 'Unknown' level

Hyperparameters are selected by stratified k-fold cross-validation on the
training file. The supplied test file is touched once, at the end, so it stays a
genuine estimate of generalisation error rather than another tuning signal.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve

from src import config, data, evaluate, plots
from src.config import CleaningConfig
from src.experiment import SCORING, Fitted, tune_and_fit
from src.models import MODEL_REGISTRY, enabled_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=sorted(MODEL_REGISTRY),
        help="which registry entries to run (default: the enabled ones)",
    )
    parser.add_argument("--quick", action="store_true", help="use reduced grids")
    parser.add_argument("--folds", type=int, default=config.CV_FOLDS)
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        choices=sorted(evaluate.Z_TABLE),
        help="confidence level for the accuracy and model-difference intervals",
    )
    parser.add_argument(
        "--unknown-as-missing",
        action="store_true",
        help="treat the literal 'Unknown' level as a missing value",
    )
    parser.add_argument(
        "--drop-features",
        nargs="*",
        default=[],
        help="attributes to exclude before modelling",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="skip the pruning study and learning curves",
    )
    return parser.parse_args()


def baseline_fits(dataset) -> list[Fitted]:
    """The two trivial classifiers a real model has to beat.

    Reporting these alongside the models is what stops accuracy from being read
    as evidence on its own: the majority-class rule scores 0.76 here while
    detecting not a single positive case.
    """
    from sklearn.dummy import DummyClassifier

    baselines = {
        "Baseline: always 'no'": DummyClassifier(strategy="most_frequent"),
        "Baseline: random (stratified)": DummyClassifier(
            strategy="stratified", random_state=config.RANDOM_STATE
        ),
    }

    fitted = []
    for label, estimator in baselines.items():
        estimator.fit(dataset.X_train, dataset.y_train)
        y_pred = estimator.predict(dataset.X_test)
        y_score = estimator.predict_proba(dataset.X_test)[:, 1]
        fitted.append(
            Fitted(
                key=label.split(":")[1].strip().replace(" ", "_").replace("'", ""),
                label=label,
                estimator=estimator,
                best_params={},
                cv_score=float("nan"),
                cv_std=float("nan"),
                fit_seconds=0.0,
                metrics=evaluate.compute_metrics(dataset.y_test, y_pred, y_score),
                confusion=evaluate.confusion_frame(dataset.y_test, y_pred),
                y_score=y_score,
            )
        )
    return fitted


def pruning_study(dataset, cfg: CleaningConfig, cv) -> dict[str, pd.DataFrame]:
    """Training error against generalisation error as the tree is allowed to grow.

    Pre-pruning is varied through max_depth and post-pruning through ccp_alpha, so
    the two strategies from Lecture 4 can be read off the same axis: error rate.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    from src.preprocess import build_tree_preprocessor

    def evaluate_setting(**tree_kwargs) -> tuple[float, float]:
        pipeline = Pipeline(
            [
                ("prep", build_tree_preprocessor(dataset.numeric_features, dataset.categorical_features, cfg)),
                ("clf", DecisionTreeClassifier(random_state=config.RANDOM_STATE, **tree_kwargs)),
            ]
        )
        scores = cross_validate(
            pipeline,
            dataset.X_train,
            dataset.y_train,
            cv=cv,
            scoring="accuracy",
            return_train_score=True,
            n_jobs=-1,
        )
        return (
            1.0 - float(np.mean(scores["train_score"])),
            1.0 - float(np.mean(scores["test_score"])),
        )

    depth_rows = []
    for depth in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, None]:
        train_error, cv_error = evaluate_setting(max_depth=depth)
        depth_rows.append(
            {
                "max_depth": 30 if depth is None else depth,
                "max_depth_label": "unlimited" if depth is None else str(depth),
                "train_error": train_error,
                "cv_error": cv_error,
            }
        )

    alpha_rows = []
    for alpha in [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        train_error, cv_error = evaluate_setting(ccp_alpha=alpha)
        alpha_rows.append({"ccp_alpha": alpha, "train_error": train_error, "cv_error": cv_error})

    return {
        "depth": pd.DataFrame(depth_rows),
        "ccp_alpha": pd.DataFrame(alpha_rows),
    }


def learning_curve_study(fitted: list[Fitted], dataset, cv) -> dict[str, pd.DataFrame]:
    frames = {}
    fractions = np.linspace(0.1, 1.0, 6)
    for item in fitted:
        sizes, _, test_scores = learning_curve(
            item.estimator,
            dataset.X_train,
            dataset.y_train,
            cv=cv,
            scoring=SCORING,
            train_sizes=fractions,
            n_jobs=-1,
        )
        frames[item.label] = pd.DataFrame(
            {"train_size": sizes, "cv_f1": test_scores.mean(axis=1)}
        )
    return frames


def summarise(fitted: list[Fitted], dataset, confidence: float) -> pd.DataFrame:
    rows = []
    for item in fitted:
        metrics = item.metrics.as_dict()
        low, high = evaluate.accuracy_confidence_interval(
            item.metrics.accuracy, item.metrics.n, confidence
        )
        if item.y_score is not None:
            threshold, threshold_f1 = evaluate.best_f1_threshold(dataset.y_test, item.y_score)
        else:
            threshold, threshold_f1 = float("nan"), float("nan")
        rows.append(
            {
                "model": item.label,
                "cv_f1": round(item.cv_score, 4),
                "cv_f1_std": round(item.cv_std, 4),
                "accuracy": round(metrics["accuracy"], 4),
                f"accuracy_ci_{int(confidence * 100)}": f"[{low:.4f}, {high:.4f}]",
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
                "roc_auc": round(metrics["roc_auc"], 4),
                "pr_auc": round(metrics["pr_auc"], 4),
                "best_f1_threshold": round(threshold, 4),
                "f1_at_best_threshold": round(threshold_f1, 4),
                "tuning_seconds": round(item.fit_seconds, 1),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()

    cfg = CleaningConfig(
        unknown_as_missing=args.unknown_as_missing,
        drop_features=tuple(args.drop_features),
    )
    dataset = data.prepare(cfg)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=config.RANDOM_STATE)

    keys = args.models or enabled_models()
    print(f"models      : {', '.join(keys)}")
    print(f"train / test: {len(dataset.X_train)} / {len(dataset.X_test)} rows")
    print(f"positive rate: train={dataset.y_train.mean():.4f}, test={dataset.y_test.mean():.4f}")
    print(f"cleaning    : unknown_as_missing={cfg.unknown_as_missing}, "
          f"rare_level_min_count={cfg.rare_category_min_count}, "
          f"dropped={list(cfg.drop_features) or 'none'}")

    fitted: list[Fitted] = []
    for key in keys:
        spec = MODEL_REGISTRY[key]
        print(f"\n--- {spec.label} ---")
        print(f"    {spec.lecture_note}")
        item = tune_and_fit(spec, dataset, cfg, cv, args.quick)
        fitted.append(item)
        print(f"    best CV {SCORING}: {item.cv_score:.4f} (+/- {item.cv_std:.4f})  "
              f"[{item.fit_seconds:.1f}s]")
        print(f"    best params    : {item.best_params}")
        print(f"    held-out       : acc={item.metrics.accuracy:.4f}  f1={item.metrics.f1:.4f}  "
              f"pr_auc={item.metrics.pr_auc:.4f}")

    baselines = baseline_fits(dataset)

    summary = summarise(baselines + fitted, dataset, args.confidence)
    print("\n" + "=" * 78)
    print("HELD-OUT TEST RESULTS (trivial baselines first)")
    print("=" * 78)
    print(summary.to_string(index=False))

    costs = evaluate.cost_table(
        {
            item.label: (dataset.y_test, item.estimator.predict(dataset.X_test))
            for item in baselines + fitted
        }
    )
    print("\n" + "=" * 78)
    print("TOTAL MISCLASSIFICATION COST as a missed positive gets more expensive")
    print("=" * 78)
    print(costs.to_string(index=False))
    costs.to_csv(config.TABLE_DIR / "results_cost_matrix.csv", index=False)

    comparisons = []
    for i, first in enumerate(baselines[:1] + fitted):
        for second in (baselines[:1] + fitted)[i + 1 :]:
            result = evaluate.compare_error_rates(
                first.metrics.error_rate,
                first.metrics.n,
                second.metrics.error_rate,
                second.metrics.n,
                confidence=args.confidence,
                shared_test_set=True,
            )
            payload = {"model_a": first.label, "model_b": second.label, **result.as_dict()}
            comparisons.append(payload)
            verdict = "significant" if result.significant else "not significant"
            print(
                f"\n{first.label} vs {second.label}: "
                f"error {result.error_a:.4f} vs {result.error_b:.4f}, "
                f"d={result.difference:+.4f}, "
                f"{int(args.confidence * 100)}% interval "
                f"[{result.interval[0]:+.4f}, {result.interval[1]:+.4f}] -> {verdict}"
            )
            print(f"    {result.note}")

    summary.to_csv(config.TABLE_DIR / "results_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(config.TABLE_DIR / "results_significance.csv", index=False)
    for item in fitted:
        item.confusion.to_csv(config.TABLE_DIR / f"confusion_{item.key}.csv")

    with open(config.TABLE_DIR / "best_params.json", "w", encoding="utf-8") as handle:
        json.dump(
            {item.key: {str(k): str(v) for k, v in item.best_params.items()} for item in fitted},
            handle,
            indent=2,
        )

    figures = [
        plots.confusion_heatmaps(
            {item.label: item.confusion for item in baselines[:1] + fitted}
        ),
        plots.roc_and_pr(
            {item.label: item.curves for item in fitted if item.curves},
            base_rate=float(dataset.y_test.mean()),
        ),
    ]

    if not args.skip_diagnostics:
        print("\nrunning pruning study and learning curves ...")
        study = pruning_study(dataset, cfg, cv)
        study["depth"].to_csv(config.TABLE_DIR / "tree_depth_study.csv", index=False)
        study["ccp_alpha"].to_csv(config.TABLE_DIR / "tree_ccp_alpha_study.csv", index=False)
        figures.append(
            plots.complexity_curve(
                study["depth"],
                "max_depth",
                "Pre-pruning: tree depth against error",
                "results_tree_depth.png",
                tick_label_column="max_depth_label",
            )
        )
        figures.append(
            plots.complexity_curve(
                study["ccp_alpha"],
                "ccp_alpha",
                "Post-pruning: cost-complexity penalty against error",
                "results_tree_ccp_alpha.png",
            )
        )
        curves = learning_curve_study(fitted, dataset, cv)
        for label, frame in curves.items():
            frame.to_csv(
                config.TABLE_DIR / f"learning_curve_{label.replace(' ', '_').replace('/', '-')}.csv",
                index=False,
            )
        figures.append(plots.learning_curves(curves))

    print("\nfigures written:")
    for path in figures:
        print(f"  {path.relative_to(config.PROJECT_ROOT)}")
    print(f"tables written to {config.TABLE_DIR.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
