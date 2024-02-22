from unittest import TestCase

from ga.utils.cohort import make_kfold_stratified_test_train_splits
from ga.utils.phenopacket import parse_phenopackets


class TestGA(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_parse_one_phenopacket(self):
        # test/data is a directory of with 1 phenopacket
        p = parse_phenopackets("tests/data/one_ppkt_Craniometaphyseal_dysplasia/")
        self.assertIsInstance(p, dict)
        self.assertEqual(len(list(p.keys())), 2)
        self.assertCountEqual(p.keys(), {"phenotype_data", "all_data"})
        self.assertTrue("Craniometaphyseal dysplasia" in p["phenotype_data"].keys())
        self.assertTrue("Craniometaphyseal dysplasia" in p["all_data"].keys())

        # should only be 1 case in p:
        self.assertEqual(len(list(p["phenotype_data"].keys())), 1)
        self.assertEqual(len(list(p["all_data"].keys())), 1)

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

    def test_parse_phenopackets_tools_cohort(self):
        p = parse_phenopackets("tests/data/phenopacket_tools_cohort")
        expected_disease_names = ['Greig cephalopolysyndactyly syndrome',
                                  'Polydactyly, postaxial, types A1 and B',
                                  'Pallister-Hall syndrome',
                                  'Marfan syndrome',
                                  'Marfan lipodystrophy syndrome']
        self.assertCountEqual(list(p.get('all_data').keys()),
                              expected_disease_names)
        self.assertCountEqual(list(p.get('phenotype_data').keys()),
                              expected_disease_names)
        # check one set of PMIDs (cases) for a disease
        d = p.get('phenotype_data').get('Polydactyly, postaxial, types A1 and B')
        self.assertCountEqual(
            list(d.keys()),
            ['PMID_22428873_P5', 'PMID_22428873_P4', 'PMID_22428873_P3',
             'PMID_22428873_P2', 'PMID_22428873_P1', 'PMID_22428873_P6'])
        # check one phenotype for a case
        case = d.get('PMID_22428873_P5')
        self.assertEqual(len(case), 13)
        self.assertTrue(('HP:0001162', 'Postaxial hand polydactyly', 'observed')
                        in case)

    def test_parse_synthetic_data_cohort(self):
        pass
