"""Patient-disjoint retrospective ICU LOS benchmark using a local approved cohort.

This revision does not re-extract time-bounded features. Earliest recorded ICU
vitals in the source cohort can occur later in the stay; see the limitations.
"""

from pathlib import Path
import argparse
import json
from importlib.metadata import version
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
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
GROUPED_CATEGORICAL = ["gender", "race"]


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


def preprocessing(categorical=CATEGORICAL):
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
                categorical,
            ),
        ]
    )


def model_pipelines(seed, include_xgboost, categorical=CATEGORICAL):
    models = {
        "Logistic regression": LogisticRegression(
            C=1, max_iter=2000, solver="lbfgs", random_state=seed
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=2, n_jobs=2,
            random_state=seed,
        ),
    }
    if include_xgboost:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            n_jobs=2, random_state=seed, eval_metric="logloss",
        )
    return {
        name: Pipeline([("features", preprocessing(categorical)), ("model", model)])
        for name, model in models.items()
    }


def stacking_oof(pipelines, features, labels, groups, seed, n_splits=5):
    """Make one out-of-fold probability per stay and base model."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probabilities = np.full((len(features), len(pipelines)), np.nan)
    seen = np.zeros(len(features), dtype=int)
    folds = []
    for fit, valid in splitter.split(features, labels, groups):
        overlap = len(set(groups.iloc[fit]) & set(groups.iloc[valid]))
        if overlap or labels.iloc[fit].nunique() != 2 or not len(valid):
            raise ValueError("Each stacking fold needs disjoint patients and both fitting labels")
        for column, pipeline in enumerate(pipelines.values()):
            estimator = clone(pipeline).fit(features.iloc[fit], labels.iloc[fit])
            probabilities[valid, column] = estimator.predict_proba(features.iloc[valid])[:, 1]
        seen[valid] += 1
        folds.append({
            "fitting_stays": len(fit),
            "validation_stays": len(valid),
            "patient_overlap": overlap,
        })
    if not np.all(seen == 1) or not np.isfinite(probabilities).all():
        raise ValueError("Stacking requires one finite out-of-fold prediction per model and stay")
    return probabilities, folds


def run(a):
    include_xgboost = a.include_xgboost or a.include_stacking
    # The archived extraction groups marital status using full-cohort counts.
    # Exclude it from the expanded benchmark because raw values are unavailable.
    categorical = GROUPED_CATEGORICAL if include_xgboost else CATEGORICAL
    features = NUMERIC + categorical
    required = set(features + ["subject_id", "stay_id", "los", "los_long"])
    import pyarrow.parquet as pq

    # Read only model/validation columns, ignoring unrelated BigQuery dbdate metadata.
    d = pq.read_table(a.data, columns=sorted(required)).to_pandas(ignore_metadata=True)
    if not required.issubset(d):
        raise ValueError(f"Missing columns: {sorted(required-set(d))}")
    if d.stay_id.isna().any() or d.stay_id.duplicated().any():
        raise ValueError("stay_id must be complete and unique")
    if d.los.isna().any() or not d.los_long.isin([0, 1]).all() or not np.array_equal(
        d.los_long.astype(bool), d.los.ge(2)
    ):
        raise ValueError("Label must be LOS >= 2 days")
    d = d.copy()
    d[categorical] = d[categorical].replace({None: np.nan})
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
    audit["predictors"] = features
    if include_xgboost:
        audit["excluded_predictor"] = "marital_status was grouped by full-cohort frequencies in the archived extraction"
    pipelines = model_pipelines(a.seed, include_xgboost, categorical)
    train_features, train_labels = d.iloc[train][features], y.iloc[train]
    stack = None
    if a.include_stacking:
        out_of_fold, folds = stacking_oof(
            pipelines, train_features, train_labels, d.iloc[train].subject_id, a.seed
        )
        stack = LogisticRegression(C=1, solver="lbfgs", max_iter=2000, random_state=a.seed)
        stack.fit(out_of_fold, train_labels)
        audit["stacking"] = {
            "splitter": "StratifiedGroupKFold",
            "group": "subject_id",
            "base_models": list(pipelines),
            "meta_model": "LogisticRegression(C=1, solver=lbfgs, max_iter=2000)",
            "inputs": "One out-of-fold positive-class probability per base model and training stay",
            "out_of_fold_stays": len(out_of_fold),
            "folds": folds,
        }
    for pipeline in pipelines.values():
        pipeline.fit(train_features, train_labels)

    test_features, test_labels = d.iloc[test][features], y.iloc[test]
    probabilities = {
        name: pipeline.predict_proba(test_features)[:, 1]
        for name, pipeline in pipelines.items()
    }
    artifacts = dict(pipelines)
    if stack is not None:
        probabilities["Stacking"] = stack.predict_proba(
            np.column_stack(list(probabilities.values()))
        )[:, 1]
        artifacts["Stacking"] = {
            "base_models": list(pipelines.items()),
            "meta_model": stack,
        }
    audit["models"] = list(probabilities)
    audit["threshold"] = 0.5
    audit["packages"] = {
        name: version(name)
        for name in ("numpy", "pandas", "scikit-learn", "pyarrow", "matplotlib")
    }
    if include_xgboost:
        audit["packages"]["xgboost"] = version("xgboost")

    a.output.mkdir(parents=True, exist_ok=True)
    results = []
    fig, ax = plt.subplots(figsize=(6, 4.6), layout="constrained")
    for name, prob in probabilities.items():
        pred = prob >= 0.5
        results.append(
            dict(
                model=name,
                roc_auc=roc_auc_score(test_labels, prob),
                accuracy=accuracy_score(test_labels, pred),
                precision=precision_score(test_labels, pred, zero_division=0),
                recall=recall_score(test_labels, pred, zero_division=0),
                f1=f1_score(test_labels, pred, zero_division=0),
            )
        )
        fpr, tpr, _ = roc_curve(test_labels, prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC {results[-1]["roc_auc"]:.3f})')
        if a.model_dir:
            import joblib

            a.model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                artifacts[name], a.model_dir / (name.lower().replace(" ", "-") + ".joblib")
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
    p.add_argument(
        "--include-stacking", action="store_true",
        help="Include XGBoost and the five-fold patient-grouped stack.",
    )
    p.add_argument("--model-dir", type=Path)
    run(p.parse_args())
