# Revision notes

The starting point was Heyang Ma’s independent UCLA course analysis. The public version was revised with coding-assistant support in September 2026 to improve reproducibility and interpretation. It is academic work, not employment or clinical deployment.

Replaced a row-wise split with a patient-grouped split: the original split shared 10,058 patients across sets; the revision shares zero. Added numeric scaling and unknown-category handling. Preprocessing is fitted only on training data. Original XGBoost tuning and the original Dash interface are not claimed as rerun.

Published numerical findings refer to the revised scripts and their generated outputs. Original graded reports, instructor prompts, source data, student identifiers, and notebook outputs are not included. The original files are preserved privately.
