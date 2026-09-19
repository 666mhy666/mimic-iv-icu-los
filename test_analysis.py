"""Regression checks for patient leakage and training-only preprocessing."""

import unittest
import numpy as np
import pandas as pd
from train import patient_split, preprocessing, NUMERIC, CATEGORICAL


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


if __name__ == "__main__":
    unittest.main()
