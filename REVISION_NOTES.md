# Revision notes

## Original coursework

The R and Python implementations covered the same project, including the core preparation, modeling and evaluation workflow. The R implementation also included stacking and R Shiny. The original R files are unavailable, so its history comes from my account. The surviving Python coursework includes the BigQuery cohort, randomized tuning of logistic regression, random forest and XGBoost, and Dash.

## September 2026 portfolio revision

The September 2026 revision replaced the row split with a patient split and added pipeline scaling and handling for unseen categories. The original split shared 10,058 patients across sets. The revised split shared zero. The first revision evaluated logistic regression and random forest with the archived features. Those [results remain available](results/september-2026-lr-rf/grouped-holdout-metrics.csv).

## November 2026 expansion

Added fixed-setting XGBoost and a logistic stack built from five patient-separated out-of-fold predictions on the training data. All four models use the original September holdout assignment and share a feature set that excludes marital status. The cohort export grouped that variable using full-cohort frequencies, and the raw categories are unavailable in the export. See [REPORT.md](REPORT.md) for the full design. The revised model outputs and the original coursework results are stored separately.

Added a copy of the original Python tuning workflow with cleared outputs and local input and model paths. Preserved the saved aggregate coursework metrics and tuning figures. Added the Dash source with local model loading, predictor-column filtering and loopback-only serving. The results tables identify which run produced each number. The original restricted inputs, model files, row previews and course reports remain private.

The portfolio revision was prepared with coding-assistant support.
