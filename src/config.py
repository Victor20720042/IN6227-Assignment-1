"""Central configuration: paths, target definition, and the cleaning decisions
that the report has to justify.

Every field in CleaningConfig corresponds to a choice discussed in Lecture 2
(data quality / preprocessing), so the report can point at a concrete switch
and explain why it was set the way it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dataset"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

TARGET = "label"
POSITIVE_CLASS = "yes"

RANDOM_STATE = 42
CV_FOLDS = 5


@dataclass
class CleaningConfig:
    # Lecture 2 lists four strategies for missing values. "impute" = estimate
    # the value; "drop" = eliminate the data object. Dropping is only applied to
    # the training set, never to test rows, so the evaluation set stays fixed.
    numeric_missing: str = "median"
    drop_incomplete_train_rows: bool = False

    # "constant" gives missing nominal values their own level, which preserves
    # them as a group the model can act on. "most_frequent" merges them into the
    # majority level, which is what it takes to actually test whether that group
    # carries anything -- filling with a fresh label only renames it.
    categorical_missing: str = "constant"
    categorical_missing_label: str = "Missing"

    # The literal string "Unknown" appears ~5.7% of the time in two columns.
    # Treating it as a real level keeps whatever signal it carries; treating it
    # as missing follows the usual reading of a placeholder. Toggle to compare.
    unknown_as_missing: bool = False
    unknown_tokens: tuple[str, ...] = ("Unknown", "?", "N/A", "NA")

    # Exact zeros that co-occur across attributes look like a code for "not
    # recorded" rather than a measurement. Listing a column here converts its
    # zeros to NaN so the imputer handles them; leaving it out keeps them as
    # values. The zero block is strongly associated with one class, so this is
    # not a free choice -- it trades a plausible reading of the data against a
    # pattern the classifier can use.
    zero_as_missing: tuple[str, ...] = field(default_factory=tuple)

    # Rare-level grouping. High-cardinality nominal attributes (region has 21
    # levels) inflate multi-way split gain, which is the bias Lecture 4 warns
    # about; grouping also gives unseen test levels somewhere to land.
    rare_category_min_count: int = 50
    rare_category_label: str = "Rare"

    # Features excluded before modelling, with the reason recorded for the report.
    drop_features: tuple[str, ...] = field(default_factory=tuple)


DEFAULT_CLEANING = CleaningConfig()


def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, FIGURE_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)
