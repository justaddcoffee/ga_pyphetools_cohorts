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

    def test_parse_phenopackets(self):
        # test/data is a directory of with 1 phenopacket
        p = parse_phenopackets("tests/data/")
        self.assertIsInstance(p, dict)
        self.assertEqual(len(list(p.keys())), 2)
        self.assertCountEqual(p.keys(), {"phenotype_data", "all_data"})
        self.assertTrue("Craniometaphyseal dysplasia" in p["phenotype_data"].keys())
        self.assertTrue("Craniometaphyseal dysplasia" in p["all_data"].keys())

        self.assertTrue(
            "PMID_20186813_6-year-old_male_patient"
            in p["phenotype_data"]["Craniometaphyseal dysplasia"].keys()
        )
        self.assertTrue(
            "PMID_20186813_6-year-old_male_patient"
            == p["all_data"]["Craniometaphyseal dysplasia"][0]["id"]
        )

        self.assertEqual(
            len(
                p["phenotype_data"]["Craniometaphyseal dysplasia"][
                    "PMID_20186813_6-year-old_male_patient"
                ]
            ),
            7,
        )
        self.assertCountEqual(
            p["phenotype_data"]["Craniometaphyseal dysplasia"][
                "PMID_20186813_6-year-old_male_patient"
            ],
            [
                ("HP:0000407", "Sensorineural hearing impairment", "observed"),
                ("HP:0000572", "Visual loss", "observed"),
                ("HP:0011856", "Pica", "observed"),
                ("HP:0100255", "Metaphyseal dysplasia", "observed"),
                ("HP:0100774", "Hyperostosis", "observed"),
                ("HP:0003196", "Short nose", "observed"),
                ("HP:0011856", "Pica", "observed"),
            ],
        )

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
