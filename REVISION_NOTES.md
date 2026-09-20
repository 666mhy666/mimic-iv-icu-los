# Revision notes

## Original coursework

The original coursework built the MIMIC-IV cohort, tuned logistic regression, random forest, and XGBoost models, and included a Dash interface.

## September 2026 portfolio revision

The September 2026 revision replaced the row-wise split with a patient-grouped split, fitted preprocessing only on training data, and added scaling and unknown-category handling. The original split shared 10,058 patients across sets; the revised split shares zero. Published numerical benchmarks come from the revised fixed-parameter logistic regression and random forest analysis; the original XGBoost tuning and Dash interface were not rerun.

The public revision was prepared with coding-assistant support to improve reproducibility and interpretation. Published numerical findings refer to the revised scripts and generated outputs unless stated otherwise. Original graded reports, instructor prompts, source data, student identifiers, and notebook outputs are not included.
