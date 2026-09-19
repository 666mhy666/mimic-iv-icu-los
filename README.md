# ICU Length of Stay Prediction

Build a clinical cohort and evaluate whether ICU stays last at least two days, with patients kept separate across training and testing.

**Author:** Heyang Ma · Independent UCLA graduate course project, revised for this portfolio.
**Tools:** SQL / BigQuery, Python, Patient-level evaluation. **Scope:** 94,444 ICU stays.

## Question and result

How well do demographics, laboratory measurements, and recorded vital signs classify ICU stays lasting at least two days?

In a revised patient-grouped holdout, random forest ROC-AUC was 0.612 and logistic regression ROC-AUC was 0.581. The test set contained 47,151 stays from 32,678 patients. These are new fixed-parameter benchmarks, not the original tuned-model results.

![Main result](results/grouped-holdout-roc.png)

## What the analysis does

The executable analysis is [train.py](train.py). [Methods and interpretation](REPORT.md) explains the scope; [revision notes](REVISION_NOTES.md) distinguish the original analysis from the portfolio revision.

## Run locally

Install Python dependencies with `python -m pip install -r requirements.txt`.
Read [data access and input requirements](DATA_ACCESS.md), then run from this repository:

```sh
python train.py --data data/mimiciv_icu_cohort.parquet --output results
```

## Results and limits

This is retrospective classification. The source extraction takes the earliest available vital sign anywhere in the ICU stay, rather than enforcing a prediction-time window. Labs have no bounded lookback and are linked by patient rather than admission. These features do not support a prospective admission-time prediction claim. The source uses anchor age rather than admission-specific age and groups categories before splitting. No external or temporal validation was performed.

The committed `results/` files are generated summaries from the portfolio revision. Source records, credentials, fitted models, and original notebook outputs are excluded.

## Checks

Run `python -m unittest test_analysis.py`. These tests cover the corrected failure cases; they do not replace statistical validation.
