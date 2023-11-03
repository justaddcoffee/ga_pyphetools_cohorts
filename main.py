import os
import warnings

import networkx as nx
from semsimian import Semsimian


from ga.ga import parse_phenopackets, run_genetic_algorithm, make_cohort, \
    make_test_train_splits, make_hpo_closures_and_graph, make_hpo_labels_df
from tqdm import tqdm


# Press the green button in the gutter to run the script.
def run_smoke_test():
    # this termset sim seems to be 0.0, which seems fishy
    pt_test_tuples = [('HP:0002650', 1.0, False),
                      ('HP:0000098', 1.0, False),
                      ('HP:0001166', 1.0, False),
                      ('HP:0001083', 1.0, False),
                      ('HP:0000545', 1.0, False),
                      ('HP:0002616', 1.0, False)]
    profile_test_tuples = [('HP:0033127', 0.7594267694796112, True),
                           ('HP:0033677', 0.2590903171508303, False),
                           ('HP:0010730', 0.7373312314046617, False),
                           ('HP:0005206', 0.16651076083997507, False),
                           ('HP:0033729', 0.30911732402073555, False)]

    # version of these variables with only the term
    pt_test_terms = ['HP:0002650', 'HP:0000098', 'HP:0001166', 'HP:0001083', 'HP:0000545', 'HP:0002616']
    profile_test_terms = ['HP:0033127', 'HP:0033677', 'HP:0010730', 'HP:0005206', 'HP:0033729']

    test_sim = s.termset_pairwise_similarity_weighted_negated(
        subject_dat=pt_test_tuples,
        object_dat=profile_test_tuples)
    return test_sim


if __name__ == '__main__':
    ################################################################
    # things we might want to change/set at runtime
    ################################################################
    phenopackets_path = os.path.join('phenopacket-store', 'phenopackets')
    data = parse_phenopackets(phenopackets_path)
    hpo_url = 'https://kg-hub.berkeleybop.io/kg-obo/hp/2023-04-05/hp_kgx_tsv.tar.gz'
    hpo_root_node_to_use = 'HP:0000001'
    # make a cohort to analyze
    disease = 'Marfan syndrome'
    diseases_to_remove_from_negatives =  ['Marfan lipodystrophy syndrome']
        # 'Coffin-Siris syndrome 3 ',
        # 'Rhabdoid tumor predisposition syndrome-1',
        # 'severe intellectual disability and choroid plexus hyperplasia with resultant hydrocephalus',
        # 'MANDIBULOACRAL DYSPLASIA WITH TYPE A LIPODYSTROPHY; MADA',
        # 'HUTCHINSON-GILFORD PROGERIA SYNDROME; HGPS',
        # 'EMERY-DREIFUSS MUSCULAR DYSTROPHY 3, AUTOSOMAL RECESSIVE; EDMD3',
        # 'CARDIOMYOPATHY, DILATED, 1A; CMD1A', 'LIPODYSTROPHY, FAMILIAL PARTIAL, TYPE 2; FPLD2', 'Developmental and epileptic encephalopathy 28',
        # 'Spinocerebellar ataxia, autosomal recessive 12', 'Joubert syndrome 10', 'Simpson-Golabi-Behmel syndrome, type 2', 'Orofaciodigital syndrome I',
        # 'Houge-Janssen syndrome 2', 'Greig cephalopolysyndactyly syndrome', 'Polydactyly, postaxial, types A1 and B', 'Pallister-Hall syndrome',
        # 'Luscan-Lumish syndrome', 'Rabin-Pappas syndrome', 'Intellectual developmental disorder, autosomal dominant 70', 'Cryohydrocytosis',
        # 'Renal tubular acidosis, distal, with hemolytic anemia', 'Renal tubular acidosis, distal, autosomal dominant', 'Spherocytosis, type 4',
        # 'Acromelic frontonasal dysostosis', 'Neurodevelopmental disorder with movement abnormalities, abnormal gait, and autistic features',
        # 'Craniometaphyseal dysplasia', 'Chondrocalcinosis 2', 'Coffin-Siris syndrome 8', 'Marfan syndrome', 'Acromicric dysplasia', 'Marfan lipodystrophy syndrome',
        # 'Ectopia lentis, familial', 'Stiff skin syndrome', 'Wolfram syndrome 1', 'Deafness, autosomal dominant 6', 'Albinism, oculocutaneous, type IV',
        # 'ERI1-related disease', 'ZTTK SYNDROME', 'EHH1-related neurodevelopmental disorder', 'epilepsy', 'Atypical SCN2A-related disease',
        # 'developmental and epileptic encephalopathy 11', 'seizures, benign familial infantile, 3', 'autism spectrum disorder', 'EHLERS-DANLOS SYNDROME, VASCULAR TYPE',
        # 'Polymicrogyria with or without vascular-type EDS', 'NDD', 'West Syndrome', 'EOEE', 'Ohtahara Syndrome', 'Other DEE', 'Atypical Rett Syndrome']
    phenopackets_store_gh_url = "https://github.com/monarch-initiative/phenopacket-store.git"
    num_kfold_splits = 5
    include_self_in_closure = True
    remove_pt_terms_not_in_spo = False
    debug = False

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

    # make "spo" (subject predicate object closures) for semsimian and also nx graph
    # assign spo to first element of tuple, graph to second
    spo, hpo_graph = make_hpo_closures_and_graph(
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

    # run genetic algorithm on each kfold split
    # for i in tqdm(range(num_kfold_splits), desc="kfold splits"):
    i = 0
    run_genetic_algorithm(
        semsimian=s,
        disease=disease,
        pt_train_df=pt_test_train_df[i]['train'],
        pt_test_df=pt_test_train_df[i]['test'],
        hpo_graph=hpo_graph,
        node_labels=node_labels,
        debug = debug
    )



