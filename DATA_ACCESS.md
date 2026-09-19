# Data access

Requires credentialed MIMIC-IV access and a locally prepared cohort. Keep the cohort and any fitted models outside the public repository. The extraction notebook documents the original MIMIC-IV 3.1 queries; it does not repair the feature-time limitations above. BigQuery execution can incur charges.

## Input contract

See NUMERIC/CATEGORICAL feature lists in train.py; subject_id, stay_id, los, and los_long are also required. los_long must equal los >= 2.

Source context: [https://physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/)

Keep inputs in a local `data/` directory. Input data and model files are excluded from the public release. Course access is not assumed to confer redistribution permission.
