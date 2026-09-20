# ICU Length of Stay Prediction

Build a clinical cohort and evaluate whether ICU stays last at least two days, with patients kept separate across training and testing.

**Author:** Heyang Ma · Independent UCLA graduate course project, revised for this portfolio.
**Tools:** SQL / BigQuery, Python, Patient-level evaluation. **Scope:** 94,444 ICU stays.

## Question

How well do demographics, laboratory measurements, and recorded vital signs classify ICU stays lasting at least two days?

## What I did

I built the cohort in BigQuery and evaluated logistic regression and random forest on a patient-grouped holdout. The features include demographics, laboratory measurements joined at the patient level, and the earliest recorded vital sign within each ICU stay.

## Main finding

In a revised patient-grouped holdout, random forest ROC-AUC was 0.612 and logistic regression ROC-AUC was 0.581. The test set contained 47,151 stays from 32,678 patients. These are new fixed-parameter benchmarks, not the original tuned-model results.

![ROC curves from the revised patient-grouped retrospective holdout.](results/grouped-holdout-roc.png)

_ROC curves from the revised patient-grouped retrospective holdout._

## Important limitations

This is retrospective classification, not admission-time prediction. Vital signs are the earliest recorded anywhere in the ICU stay, while laboratory measurements have no bounded lookback and are joined by patient rather than admission. No external or temporal validation was performed.

## Code and reproducibility

Install Python dependencies with `python -m pip install -r requirements.txt`.
Read [data access and input requirements](DATA_ACCESS.md), then run from this repository:

```sh
python train.py --data data/mimiciv_icu_cohort.parquet --output results
```

XGBoost remains available as an optional comparison. Install `requirements-xgboost.txt` and add `--include-xgboost`; the committed benchmarks do not include that optional run.


The executable analysis is [train.py](train.py). See [REPORT.md](REPORT.md) for model details and interpretation, [DATA_ACCESS.md](DATA_ACCESS.md) for inputs, and [REVISION_NOTES.md](REVISION_NOTES.md) for the distinction between the course project and portfolio revision. The committed `results/` files are generated summaries from the revision.

## Checks

Run `python -m unittest test_analysis.py`. These tests cover the corrected failure cases; they do not replace statistical validation.
