from unittest import TestCase

from ga.utils.cohort import make_kfold_stratified_test_train_splits
from ga.utils.phenopacket import parse_phenopackets


class TestGA(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # read in pt_df fixture
        import pandas as pd
        cls.pt_df = pd.read_csv("tests/data/test_pt_df.csv")

    def test_reality(self):
        self.assertIsNone(None)

    # test for data leakage
    def test_no_leakage(self):
        # make sure that the test and train sets don't have any overlapping patients
        train_test_kfolds = make_kfold_stratified_test_train_splits(
            pt_df=self.pt_df, pt_id_col="person_id", pt_label_col="patient_label"
        )
        for item in train_test_kfolds:
            train = item["train"]
            test = item["test"]
            train_patients = set(train["person_id"].unique())
            test_patients = set(test["person_id"].unique())
            self.assertEqual(len(train_patients.intersection(test_patients)), 0)
