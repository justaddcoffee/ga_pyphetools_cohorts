import os
import warnings

from semsimian import Semsimian


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
    pt_df.rename(columns={'excluded': 'negated'}, inplace=True)

    # test/train split
    num_splits = 5
    pt_test_train_df = make_test_train_splits(pt_df=pt_df, num_splits=num_splits, seed=42)

    # make "spo" (subject predicate object closures) for semsimian
    spo: list = make_hpo_closures(
        url='https://kg-hub.berkeleybop.io/kg-obo/hp/2023-04-05/hp_kgx_tsv.tar.gz',
        root_node_to_use='HP:0000001'
    )

    # check how many pt HPO terms we have that aren't in the spo
    all_pt_hpo_terms = set(pt_df['hpo_term_id'].unique())
    all_spo_hpo_terms = set([s[0] for s in spo] + [s[2] for s in spo])

    # find all_pt_hpo_terms that aren't in all_spo_hpo_terms
    pt_hpo_terms_not_in_spo = all_pt_hpo_terms.difference(all_spo_hpo_terms)
    warnings.warn(f"There are {str(len(pt_hpo_terms_not_in_spo))} "
                  f"({str(round(100*len(pt_hpo_terms_not_in_spo) / len(all_pt_hpo_terms), 2))}%) patient HPO terms are not in the closures I'm using: "
                  f"{' '.join(pt_hpo_terms_not_in_spo)}\n"
                  f"These are possibly obsolete terms, or terms that are not in the induced subgraph of the `root_node_to_use` arg passed to make_hpo_closures(). "
                  f"These terms will have 0 semantic similarity to other terms")
    s = Semsimian(spo=spo)

    # run genetic algorithm on each kfold split
    for i in range(num_splits):
        run_genetic_algorithm(
            semsimian=s,
            spo=spo,
            pt_train_df=pt_test_train_df[i]['train'],
            pt_test_df=pt_test_train_df[i]['test'],
        )

    pass


