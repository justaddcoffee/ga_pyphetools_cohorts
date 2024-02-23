from unittest import TestCase

from ga.utils.cohort import make_kfold_stratified_test_train_splits
from ga.utils.phenopacket import parse_phenopackets


class TestPhenopacketParsing(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_parse_one_phenopacket(self):
        # test/data is a directory of with 1 phenopacket
        p = parse_phenopackets("tests/data/one_ppkt_Craniometaphyseal_dysplasia/")
        self.assertIsInstance(p, dict)
        self.assertEqual(len(list(p.keys())), 1)
        self.assertCountEqual(p.keys(), {"by_disease"})
        self.assertTrue("Craniometaphyseal dysplasia" in p["by_disease"].keys())

        # should only be 1 case in p:
        self.assertEqual(len(list(p["by_disease"].keys())), 1)

        case = p["by_disease"]["Craniometaphyseal dysplasia"][0]
        self.assertEqual(case.get("id"), "PMID_20186813_6-year-old_male_patient")
        self.assertEqual(case.get("subject"),
                         {'id': '6-year-old male patient',
                          'timeAtLastEncounter': {'age': {'iso8601duration': 'P12Y'}}})

        phenotypic_features = case.get("phenotypicFeatures")
        self.assertEqual(len(phenotypic_features), 7)

        self.assertCountEqual(
            phenotypic_features,
            [
                {'type': {'excluded': True, 'id': 'HP:0003196', 'label': 'Short nose'}},
                {'type': {'id': 'HP:0000407',
                          'label': 'Sensorineural hearing impairment'}},
                {'type': {'id': 'HP:0000572', 'label': 'Visual loss'}},
                {'type': {'id': 'HP:0011856', 'label': 'Pica'}},
                {'type': {'id': 'HP:0100255', 'label': 'Metaphyseal dysplasia'}},
                {'type': {'id': 'HP:0100774', 'label': 'Hyperostosis'}},
                {'type': {'id': 'HP:0011856', 'label': 'Pica'}}])

        self.assertCountEqual(case.get('parsedPhenotypicFeatures'),
                              [('HP:0003196', 'Short nose', 'excluded'),
                               ('HP:0000407', 'Sensorineural hearing impairment',
                                'observed'),
                               ('HP:0000572', 'Visual loss', 'observed'),
                               ('HP:0011856', 'Pica', 'observed'),
                               ('HP:0100255', 'Metaphyseal dysplasia', 'observed'),
                               ('HP:0100774', 'Hyperostosis', 'observed'),
                               ('HP:0011856', 'Pica', 'observed')])

    def test_parse_phenopackets_tools_cohort(self):
        p = parse_phenopackets("tests/data/phenopacket_tools_cohort")
        expected_disease_names = ['Greig cephalopolysyndactyly syndrome',
                                  'Polydactyly, postaxial, types A1 and B',
                                  'Pallister-Hall syndrome',
                                  'Marfan syndrome',
                                  'Marfan lipodystrophy syndrome']
        self.assertCountEqual(list(p.get('by_disease').keys()), expected_disease_names)
        # check one set of PMIDs (cases) for a disease
        d = p.get('by_disease').get('Polydactyly, postaxial, types A1 and B')
        self.assertCountEqual([c['id'] for c in d],
                              ['PMID_22428873_P5', 'PMID_22428873_P4',
                               'PMID_22428873_P3',
                               'PMID_22428873_P2', 'PMID_22428873_P1',
                               'PMID_22428873_P6'])
        # check one phenotype for a case
        case = [c for c in d if c['id'] == 'PMID_22428873_P5'][0]
        self.assertEqual(len(case['phenotypicFeatures']), 13)
        self.assertTrue(('HP:0001162', 'Postaxial hand polydactyly', 'observed')
                        in case['parsedPhenotypicFeatures'])

    def test_parse_synthetic_data_cohort(self):
        p = parse_phenopackets("tests/data/synthetic_data_cohort")
        self.assertCountEqual(list(p.get('by_disease').keys()),
                              ['Marfan syndrome', 'Cystic fibrosis'])

        marfan_cases = p['by_disease']['Marfan syndrome']
        self.assertCountEqual(
            [case['subject']['id'] for case in marfan_cases],
            ['OMIM:154700_Marfan syndrome_5', 'OMIM:154700_Marfan syndrome_9',
             'OMIM:154700_Marfan syndrome_8', 'OMIM:154700_Marfan syndrome_4',
             'OMIM:154700_Marfan syndrome_3', 'OMIM:154700_Marfan syndrome_2',
             'OMIM:154700_Marfan syndrome_10', 'OMIM:154700_Marfan syndrome_1',
             'OMIM:154700_Marfan syndrome_7', 'OMIM:154700_Marfan syndrome_6'])
        one_case = [case for case in marfan_cases if case['subject']['id'] == 'OMIM:154700_Marfan syndrome_8'][0]
        self.assertEqual(len(one_case['phenotypicFeatures']), 27)
        self.assertCountEqual(
            one_case['parsedPhenotypicFeatures'],
            [('HP:0001377', 'Limited elbow extension', 'observed'),
             ('HP:0001519', 'Disproportionate tall stature', 'observed'),
             ('HP:0012385', 'Camptodactyly', 'observed'),
             ('HP:0003088', 'Premature osteoarthritis', 'observed'),
             ('HP:0000272', 'Malar flattening', 'observed'),
             ('HP:0000518', 'Cataract', 'observed'),
             ('HP:0004970', 'Ascending tubular aorta aneurysm', 'observed'),
             ('HP:0100775', 'Dural ectasia', 'observed'),
             ('HP:0001166', 'Arachnodactyly', 'observed'),
             ('HP:0000218', 'High palate', 'observed'),
             ('HP:0001840', 'Metatarsus adductus', 'observed'),
             ('HP:0001382', 'Joint hypermobility', 'observed'),
             ('HP:0003179', 'Protrusio acetabuli', 'observed'),
             ('HP:0003199', 'Decreased muscle mass', 'observed'),
             ('HP:0000545', 'Myopia', 'observed'),
             ('HP:0012372', 'Abnormal eye morphology', 'observed'),
             ('HP:0009805', 'Low-output congestive heart failure', 'observed'),
             ('HP:0000678', 'Dental crowding', 'observed'), (
             'HP:0002011', 'Morphological central nervous system abnormality',
             'observed'),
             ('HP:0012773', 'Reduced upper to lower segment ratio', 'observed'),
             ('HP:0001659', 'Aortic regurgitation', 'observed'),
             ('HP:0030799', 'Scaphocephaly', 'observed'),
             ('HP:0005136', 'Mitral annular calcification', 'observed'),
             ('HP:0001786', 'Narrow foot', 'observed'),
             ('HP:0003758', 'Reduced subcutaneous adipose tissue', 'observed'),
             ('HP:0001765', 'Hammertoe', 'observed'),
             ('HP:0011805', 'Abnormal skeletal muscle morphology', 'observed')])
        pass
