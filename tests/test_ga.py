import pandas as pd
from unittest import TestCase
from ga.ga import parse_phenopackets, run_genetic_algorithm, make_cohort


class TestGA(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_reality(self):
        self.assertIsNone(None)

    def test_parse_phenopackets(self):
        p = parse_phenopackets('data/')
        self.assertIsInstance(p, dict)
        self.assertEquals(len(list(p.keys())), 2)
        self.assertCountEqual(p.keys(), {'phenotype_data', 'all_data'})
        self.assertTrue('Craniometaphyseal dysplasia' in p['phenotype_data'].keys())
        self.assertTrue('Craniometaphyseal dysplasia' in p['all_data'].keys())

        self.assertTrue('PMID_20186813_6-year-old_male_patient' in p['phenotype_data']['Craniometaphyseal dysplasia'].keys())
        self.assertTrue('PMID_20186813_6-year-old_male_patient' == p['all_data']['Craniometaphyseal dysplasia'][0]['id'])

        self.assertEquals(len(p['phenotype_data']['Craniometaphyseal dysplasia']['PMID_20186813_6-year-old_male_patient']), 7)
        self.assertCountEqual(p['phenotype_data']['Craniometaphyseal dysplasia']['PMID_20186813_6-year-old_male_patient'],
                          [ ('HP:0000407', 'Sensorineural hearing impairment', 'observed'),
                                    ('HP:0000572', 'Visual loss', 'observed'),
                                    ('HP:0011856', 'Pica', 'observed'),
                                    ('HP:0100255', 'Metaphyseal dysplasia', 'observed'),
                                    ('HP:0100774', 'Hyperostosis', 'observed'),
                                    ('HP:0003196', 'Short nose', 'observed'),
                                    ('HP:0011856', 'Pica', 'observed')])

        pass

