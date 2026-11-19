# Classifying ICU Length of Stay with MIMIC-IV

I implemented the same ICU length of stay workflow in R and Python. In both, I prepared the data, preprocessed the inputs, fit logistic regression, random forest and boosting models, and evaluated them. The R version also included stacking and Shiny. In Python, I used BigQuery to build a cohort of 94,444 stays, classified whether a stay lasted at least two days, and built a Dash app for the logistic regression, random forest and XGBoost models.

[Read the case study](https://666mhy666.github.io/projects/mimic-iv-icu-los/)

## Repository structure

| Start here | What's inside |
| --- | --- |
| [cohort-extraction.ipynb](cohort-extraction.ipynb) | Original MIMIC-IV 3.1 BigQuery queries, joins, adult cohort criteria and outcome definition |
| [notebooks/course-modeling.ipynb](notebooks/course-modeling.ipynb) | Original Python row split, preprocessing, model tuning, evaluation and local model saves |
| [Dash source](app/dash_course_app.py) · [App setup](app/README.md) | Choose models, submit feature values for predictions, or evaluate a labeled file |
| [train.py](train.py) | Patient-separated evaluation with logistic regression, random forest, XGBoost and stacking |
| [results/](results/) | Saved coursework metrics and tuning figures, revised metrics, ROC figure and overlap audit |
| [REPORT.md](REPORT.md) · [REVISION_NOTES.md](REVISION_NOTES.md) | Feature timing, model settings and what changed |

## Results

The original Python test ROC-AUC was 0.570 for logistic regression, 0.624 for random forest and 0.629 for XGBoost. These scores come from tuned models tested on 47,222 held-out stays in a stratified split by stay. See the [saved coursework metrics](results/course-python-metrics.csv).

The September 2026 revision keeps each patient's stays together. Its November expansion compares four models on the same 47,151 test stays from 32,678 patients. The stack learns from predictions made on training patients held out from each of five folds. The expanded benchmark omits marital status because the archived extraction grouped it using full-cohort frequencies. The original and revised analyses use different model fits and test sets. See the [four-model metrics](results/grouped-holdout-metrics.csv), [ROC figure](results/grouped-holdout-roc.png) and [split audit](results/cohort-audit.json).

This is retrospective classification. The earliest recorded vitals could come from anywhere in the stay, and labs stored before ICU entry were matched by patient without a bounded lookback. There was no external or temporal validation.

## Run the public analysis

Install the pinned Python dependencies and follow [DATA_ACCESS.md](DATA_ACCESS.md) to prepare an authorized local cohort. Run from the repository root:

```sh
python -m pip install -r requirements-xgboost.txt
python train.py --data data/mimiciv_icu_cohort.parquet --output data/benchmark-results --include-stacking
python -m unittest test_analysis.py
```

[run_analysis.ipynb](run_analysis.ipynb) launches the same command. Use `--include-xgboost` instead of `--include-stacking` for the three individual models on the expanded feature set. With neither flag, the script reproduces the [earlier two-model benchmark](results/september-2026-lr-rf/grouped-holdout-metrics.csv) with its original features.

For the original Python tuning notebook and Dash interface, see [course environment and app setup](app/README.md).
