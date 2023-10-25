import os
from ga.ga import parse_phenopackets, run_genetic_algorithm, make_cohort, make_test_train_split


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get phenopacket data (can't add this repo as a dependency)
    if not os.path.exists('phenopacket-store'):
        os.system("git clone https://github.com/monarch-initiative/phenopacket-store.git")

    phenopackets_path = 'phenopacket-store/phenopackets/'
    data = parse_phenopackets(phenopackets_path)

    # make a cohort to analyze
    disease = 'Marfan syndrome'

    negatives = list(data['phenotype_data'].keys())
    negatives.remove(disease)
    negatives.remove('Marfan lipodystrophy syndrome')
    negatives.sort()

    # make pandas dataframe
    pt_df = make_cohort(data['phenotype_data'], disease, negatives)

    # test/train split
    pt_test_train_df = make_test_train_split(pt_df)

    pass


