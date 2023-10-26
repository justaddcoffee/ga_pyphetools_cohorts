import os
import tempfile

import wget

from ga.ga import parse_phenopackets, run_genetic_algorithm, make_cohort, \
    make_test_train_splits, make_hpo_closures

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
    num_splits = 5
    pt_test_train_df = make_test_train_splits(pt_df=pt_df, num_splits=num_splits, seed=42)

    # download HPO graphq
    spo: list = make_hpo_closures()


    pass

    # run genetic algorithm on each kfold split
    for i in range(num_splits):
        run_genetic_algorithm(
            pt_train_df=pt_test_train_df[i]['train'],
            pt_test_df=pt_test_train_df[i]['test'],
        )

    pass


