"""Regression checks for patient leakage and training-only preprocessing."""

import unittest
from importlib.util import find_spec
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from train import (
    CATEGORICAL, NUMERIC, model_pipelines, patient_split, preprocessing, stacking_oof,
)


class PatientOverlapCheck(BaseEstimator):
    """Synthetic estimator that fails if any fitted patient is scored."""

    def fit(self, features, labels):
        self.fitting_patients_ = set(features["synthetic_patient"])
        return self

    def predict_proba(self, features):
        if self.fitting_patients_.intersection(features["synthetic_patient"]):
            raise AssertionError("Stacking scored a patient seen by this fitted model")
        p = features["expected_probability"].to_numpy()
        return np.column_stack((1 - p, p))


class EvaluationChecks(unittest.TestCase):
    def test_repeated_patients_stay_in_one_partition(self):
        cohort = pd.DataFrame({"subject_id": np.repeat(np.arange(20), 3)})
        train, test = patient_split(cohort)
        self.assertFalse(
            set(cohort.iloc[train].subject_id) & set(cohort.iloc[test].subject_id)
        )
        self.assertEqual(set(train) | set(test), set(range(len(cohort))))

    def test_missing_patient_identifier_is_rejected(self):
        with self.assertRaises(ValueError):
            patient_split(pd.DataFrame({"subject_id": [1, 2, np.nan]}))

    def test_unseen_categories_do_not_change_fitted_imputation(self):
        training = pd.DataFrame({name: [1.0, 3.0, np.nan] for name in NUMERIC})
        for name in CATEGORICAL:
            training[name] = ["A", "B", "A"]
        transformer = preprocessing().fit(training)
        test = training.iloc[:1].copy()
        test[NUMERIC] = 1000.0
        test[CATEGORICAL] = "unseen"
        result = transformer.transform(test)
        self.assertEqual(result.shape[0], 1)
        imputer = transformer.named_transformers_["numeric"].named_steps["impute"]
        np.testing.assert_allclose(imputer.statistics_, 2.0)

    def test_stacking_scores_each_stay_only_with_unseen_patients(self):
        patients = pd.Series(np.repeat(np.arange(30), 2))
        labels = patients % 2
        expected = np.linspace(0.1, 0.9, len(patients))
        features = pd.DataFrame({
            "synthetic_patient": patients,
            "expected_probability": expected,
        })
        pipelines = {"first": PatientOverlapCheck(), "second": PatientOverlapCheck()}
        probabilities, folds = stacking_oof(pipelines, features, labels, patients, 203)
        np.testing.assert_allclose(probabilities, np.column_stack((expected, expected)))
        self.assertEqual(len(folds), 5)
        self.assertEqual(sum(f["validation_stays"] for f in folds), len(features))
        self.assertTrue(all(f["patient_overlap"] == 0 for f in folds))

    @unittest.skipUnless(find_spec("xgboost"), "Install requirements-xgboost.txt")
    def test_xgboost_pipeline_scores_unseen_categories(self):
        features = pd.DataFrame({name: np.arange(24, dtype=float) for name in NUMERIC})
        for name in CATEGORICAL:
            features[name] = ["A", "B"] * 12
        features.loc[0, NUMERIC[0]] = np.nan
        labels = np.tile([0, 1], 12)
        pipeline = model_pipelines(203, include_xgboost=True)["XGBoost"]
        pipeline.fit(features, labels)
        sample = features.iloc[:3].copy()
        sample[CATEGORICAL] = "unseen"
        probabilities = pipeline.predict_proba(sample)
        self.assertEqual(probabilities.shape, (3, 2))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
