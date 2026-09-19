"""Patient-disjoint retrospective ICU LOS benchmark using a local approved cohort.

This revision does not re-extract time-bounded features. Earliest recorded ICU
vitals in the source cohort can occur later in the stay; see the limitations.
"""

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUMERIC = [
    "anchor_age",
    "wbc",
    "potassium",
    "sodium",
    "creatinine",
    "bicarbonate",
    "hematocrit",
    "chloride",
    "glucose",
    "heart_rate",
    "temperature_fahrenheit",
    "respiratory_rate",
    "non_invasive_blood_pressure_systolic",
    "non_invasive_blood_pressure_diastolic",
]
CATEGORICAL = ["gender", "marital_status", "race"]
FEATURES = NUMERIC + CATEGORICAL


def patient_split(d, seed=203):
    if d.subject_id.isna().any():
        raise ValueError("Missing patient identifier")
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed).split(
            d, groups=d.subject_id
        )
    )
    if set(d.iloc[train].subject_id) & set(d.iloc[test].subject_id):
        raise AssertionError("Patient overlap")
    return train, test


def preprocessing():
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="mean")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )


def run(a):
    required = set(FEATURES + ["subject_id", "stay_id", "los", "los_long"])
    import pyarrow.parquet as pq

    # Read only model/validation columns, ignoring unrelated BigQuery dbdate metadata.
    d = pq.read_table(a.data, columns=sorted(required)).to_pandas(ignore_metadata=True)
    if not required.issubset(d):
        raise ValueError(f"Missing columns: {sorted(required-set(d))}")
    if d.stay_id.isna().any() or d.stay_id.duplicated().any():
        raise ValueError("stay_id must be complete and unique")
    if d.los.isna().any() or not np.array_equal(d.los_long.astype(bool), d.los.ge(2)):
        raise ValueError("Label must be LOS >= 2 days")
    d = d.copy()
    d[CATEGORICAL] = d[CATEGORICAL].replace({None: np.nan})
    train, test = patient_split(d, a.seed)
    y = d.los_long.astype(int)
    for ix in [train, test]:
        if y.iloc[ix].nunique() != 2:
            raise ValueError("Split must contain both labels")
    old_train, old_test = train_test_split(
        d, test_size=0.5, random_state=203, stratify=y
    )
    audit = {
        "cohort_stays": len(d),
        "patients": int(d.subject_id.nunique()),
        "training_stays": len(train),
        "test_stays": len(test),
        "training_patients": int(d.iloc[train].subject_id.nunique()),
        "test_patients": int(d.iloc[test].subject_id.nunique()),
        "patient_overlap": 0,
        "original_row_split_overlapping_patients": len(
            set(old_train.subject_id) & set(old_test.subject_id)
        ),
        "positive_label": "LOS >= 2 days",
        "train_prevalence": float(y.iloc[train].mean()),
        "test_prevalence": float(y.iloc[test].mean()),
        "seed": a.seed,
        "evaluation": "Retrospective grouped holdout; no validated prediction-time window.",
    }
    models = {
        "Logistic regression": LogisticRegression(
            C=1, max_iter=2000, solver="lbfgs", random_state=a.seed
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=2,
            random_state=a.seed,
        ),
    }
    if a.include_xgboost:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            n_jobs=2,
            random_state=a.seed,
            eval_metric="logloss",
        )
    a.output.mkdir(parents=True, exist_ok=True)
    results = []
    fig, ax = plt.subplots(figsize=(6, 4.6), layout="constrained")
    for name, model in models.items():
        pipeline = Pipeline([("features", preprocessing()), ("model", model)])
        pipeline.fit(d.iloc[train][FEATURES], y.iloc[train])
        prob = pipeline.predict_proba(d.iloc[test][FEATURES])[:, 1]
        pred = prob >= 0.5
        results.append(
            dict(
                model=name,
                roc_auc=roc_auc_score(y.iloc[test], prob),
                accuracy=accuracy_score(y.iloc[test], pred),
                precision=precision_score(y.iloc[test], pred, zero_division=0),
                recall=recall_score(y.iloc[test], pred, zero_division=0),
                f1=f1_score(y.iloc[test], pred, zero_division=0),
            )
        )
        fpr, tpr, _ = roc_curve(y.iloc[test], prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC {results[-1]["roc_auc"]:.3f})')
        if a.model_dir:
            import joblib

            a.model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                pipeline, a.model_dir / (name.lower().replace(" ", "-") + ".joblib")
            )
    ax.plot([0, 1], [0, 1], ls="--", color="#888")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Patient-disjoint retrospective holdout",
    )
    ax.legend(fontsize=9)
    fig.savefig(a.output / "grouped-holdout-roc.png", dpi=170)
    plt.close(fig)
    pd.DataFrame(results).to_csv(a.output / "grouped-holdout-metrics.csv", index=False)
    (a.output / "cohort-audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=203)
    p.add_argument("--include-xgboost", action="store_true")
    p.add_argument("--model-dir", type=Path)
    run(p.parse_args())
