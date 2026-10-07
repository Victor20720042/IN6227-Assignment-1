# IN6227 Data Mining — Assignment 1 (Variant 1)

Decision tree versus naive Bayes on the supplied binary classification dataset.
Every modelling decision is restricted to material covered in Lectures 1–5, and
the code is organised so that each choice sits in one place and can be argued
for in the report.

## Running it

```bash
pip install -r requirements.txt

python run_eda.py          # exploration, data-quality audit, EDA figures
python run_experiment.py   # tuning, held-out evaluation, model comparison
```

Useful flags on `run_experiment.py`:

| Flag | Effect |
| --- | --- |
| `--models decision_tree random_forest` | run other registry entries instead of the defaults |
| `--quick` | reduced parameter grids while iterating |
| `--unknown-as-missing` | sensitivity check: treat the literal `Unknown` level as missing |
| `--drop-features index_weight` | exclude attributes before modelling |
| `--folds 10` | change the number of cross-validation folds |
| `--skip-diagnostics` | skip the pruning study and learning curves |

Outputs land in `outputs/figures` (PNG) and `outputs/tables` (CSV, plus
`best_params.json`).

## Layout

```
src/config.py       paths, target definition, and the cleaning switches
src/data.py         loading, label handling, attribute typing
src/preprocess.py   one preprocessing profile per model family
src/models.py       model registry + the mixed-type naive Bayes
src/evaluate.py     confusion-matrix metrics, accuracy CI, two-model test
src/plots.py        EDA and results figures
run_eda.py          stage 1: exploration
run_experiment.py   stage 2: tuning and evaluation
```

## Design notes

**Two models, three configurations.** The comparison the report makes is decision
tree versus naive Bayes. Naive Bayes appears twice because Lecture 5 gives two
ways to handle continuous attributes — discretise them, or assume a normal
distribution per class — and which one to use is itself a decision worth
settling empirically rather than by assertion.

**The test file is held back.** Hyperparameters are chosen by stratified k-fold
cross-validation on `train.csv`. `test.csv` is scored once, at the end.

**Imputation lives inside the pipelines.** Any quantity estimated from data (the
median used to fill a numeric gap, the level frequencies used to decide what
counts as rare) is fitted per fold, so no fold sees a statistic computed from
its own rows.

**Preprocessing is per model family, not global.** Trees get integer-coded
levels and no scaling; the discretised naive Bayes gets binned numerics; the
distance-based models kept in reserve get standardisation and one-hot encoding.

**Adding a model.** Add one `ModelSpec` to `MODEL_REGISTRY` naming its pipeline
builder and grid. `random_forest`, `knn`, `linear_svm`, and `rbf_svm` are already
wired up and switched off; enable them with `--models`. RBF SVM will be slow at
this sample size. A rule-based learner (RIPPER) is deliberately absent — there is
no scikit-learn implementation, and substituting something that is not RIPPER
would misrepresent it.
