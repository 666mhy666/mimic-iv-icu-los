# Data access

Request credentialed access through [MIMIC-IV v3.1 on PhysioNet](https://physionet.org/content/mimiciv/3.1/) and follow the BigQuery access instructions there. Use a local Python environment with an authorized Google Cloud project. Install `requirements-extraction.txt`, then open [cohort-extraction.ipynb](cohort-extraction.ipynb) from the repository root. Its export path is `data/mimiciv_icu_cohort.parquet`. BigQuery execution can incur charges.

The extraction preserves the coursework definitions. See [REPORT.md](REPORT.md) for the lab and vital sign timing. Keep authorized data, notebook outputs and trained models in your controlled environment. PhysioNet treats derived datasets and trained models as sensitive.

## Input contract

See NUMERIC/CATEGORICAL feature lists in train.py; subject_id, stay_id, los, and los_long are also required. los_long must equal los >= 2.

Keep inputs in a local `data/` directory. Input data and model files are excluded from the public release. Course access is not assumed to confer redistribution permission.
