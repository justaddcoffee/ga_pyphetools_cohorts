import os
import warnings
from semsimian import Semsimian


from ga.ga import parse_phenopackets, run_genetic_algorithm, make_cohort, \
    make_test_train_splits, make_hpo_closures, make_hpo_labels_df


# Press the green button in the gutter to run the script.
def run_smoke_test():
    # this termset sim seems to be 0.0, which seems fishy
    pt_test_tuples = [('HP:0002650', 1.0, False),
                      ('HP:0000098', 1.0, False),
                      ('HP:0001166', 1.0, False), # <-- this term causes a panic - remove this and it returns 0.0
                      ('HP:0001083', 1.0, False),
                      ('HP:0000545', 1.0, False),
                      ('HP:0002616', 1.0, False)]
    profile_test_tuples = [('HP:0033127', 0.7594267694796112, True),
                           ('HP:0033677', 0.2590903171508303, False),
                           ('HP:0010730', 0.7373312314046617, False),
                           ('HP:0005206', 0.16651076083997507, False),
                           ('HP:0033729', 0.30911732402073555, False)]
    test_sim = s.termset_pairwise_similarity_weighted_negated(
        subject_dat=pt_test_tuples,
        object_dat=profile_test_tuples)
    return test_sim


if __name__ == '__main__':
    ################################################################
    # things we might want to change/set at runtime
    ################################################################
    phenopackets_path = 'phenopacket-store/phenopackets/'
    data = parse_phenopackets(phenopackets_path)
    hpo_url = 'https://kg-hub.berkeleybop.io/kg-obo/hp/2023-04-05/hp_kgx_tsv.tar.gz'
    hpo_root_node_to_use = 'HP:0000001'
    # make a cohort to analyze
    disease = 'Marfan syndrome'
    diseases_to_remove_from_negatives = ['Marfan lipodystrophy syndrome']
    phenopackets_store_gh_url = "https://github.com/monarch-initiative/phenopacket-store.git"
    num_kfold_splits = 5
    include_self_in_closure = True
    remove_pt_terms_not_in_spo = False
    debug = True

    ################################################################

    # get phenopacket data (can't add this repo as a dependency)
    if not os.path.exists('phenopacket-store'):
        os.system(f"git clone {phenopackets_store_gh_url}")

    # make cohort
    negatives = list(data['phenotype_data'].keys())
    negatives.remove(disease)
    for r in diseases_to_remove_from_negatives:
        negatives.remove(r)
    negatives.sort()
    pt_df = make_cohort(data['phenotype_data'], disease, negatives)
    pt_df.rename(columns={'excluded': 'negated'}, inplace=True)
    # all pt phenotypes are weighted equally
    pt_df['weight'] = 1.0

    # make "spo" (subject predicate object closures) for semsimian
    spo: list = make_hpo_closures(
        url=hpo_url,
        root_node_to_use=hpo_root_node_to_use,
        include_self_in_closure=include_self_in_closure,
    )

    node_labels = make_hpo_labels_df(url=hpo_url)

    # check how many pt HPO terms we have that aren't in the spo
    all_pt_hpo_terms = set(pt_df['hpo_term_id'].unique())
    all_spo_hpo_terms = set([s[0] for s in spo] + [s[2] for s in spo])

    # find all_pt_hpo_terms that aren't in all_spo_hpo_terms
    pt_hpo_terms_not_in_spo = all_pt_hpo_terms.difference(all_spo_hpo_terms)
    warnings.warn(f"There are {str(len(pt_hpo_terms_not_in_spo))} "
                  f"({str(round(100*len(pt_hpo_terms_not_in_spo) / len(all_pt_hpo_terms), 2))}%) patient HPO terms are not in the closures we're using using: "
                  f"{' '.join(pt_hpo_terms_not_in_spo)}\n"
                  f"These are possibly obsolete terms, or terms that are not in the induced subgraph of the `root_node_to_use` arg passed to make_hpo_closures(). "
                  f"These terms will have 0 semantic similarity to other terms, and may cause a semsimian panic")
    # get rid of these terms
    if remove_pt_terms_not_in_spo:
        warnings.warn(f"Removing {str(len(pt_hpo_terms_not_in_spo))} patient HPO terms that are not in the closures I'm using")
        pt_df = pt_df[~pt_df['hpo_term_id'].isin(pt_hpo_terms_not_in_spo)]

    # test/train split
    pt_test_train_df = make_test_train_splits(pt_df=pt_df, num_splits=num_kfold_splits, seed=42)

    s = Semsimian(spo=spo)

    # this is giving sim of 0.0 which doesn't seem right
    if run_smoke_test() == 0.0:
        warnings.warn("!!!!!!!!!!!!!!!\nwhy is this similarity 0.0\n!!!!!!!!!!!!!!!!!!!!!!")

    # run genetic algorithm on each kfold split
    for i in range(num_kfold_splits):
        run_genetic_algorithm(
            semsimian=s,
            pt_train_df=pt_test_train_df[i]['train'],
            pt_test_df=pt_test_train_df[i]['test'],
            node_labels=node_labels,
            debug = debug
        )

    pass


