import random
import tarfile
import warnings
import numpy as np
import networkx as nx
from typing import List, Tuple, Optional

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from collections import defaultdict
import os
import json
import tempfile
import wget
from tqdm import tqdm
import itertools


def make_hpo_labels_df(
        url = 'https://kg-hub.berkeleybop.io/kg-obo/hp/2023-04-05/hp_kgx_tsv.tar.gz',
        hp_prefix= ['HP:'],
        cols_to_keep=['id', 'name', 'description'],
        rename_id_col='hpo_term_id',
    ) -> List[Tuple]:
    # get tmp file name
    tmpdir = tempfile.TemporaryDirectory()
    tmpfile = tempfile.NamedTemporaryFile().file.name
    wget.download(url, tmpfile)

    this_tar = tarfile.open(tmpfile, 'r:gz')
    this_tar.extractall(path=tmpdir.name)

    # show files in tmpdir
    node_files = [f for f in os.listdir(tmpdir.name) if 'nodes' in f]
    if len(node_files) != 1:
        raise RuntimeError("Didn't find exactly one edge file in {}".format(tmpdir.name))
    node_file = node_files[0]

    nodes_df = pd.read_csv(os.path.join(tmpdir.name, node_file), sep='\t')
    # select only HP terms
    nodes_df = nodes_df[nodes_df['id'].str.startswith(tuple(hp_prefix))]
    # select only cols we want
    nodes_df = nodes_df[cols_to_keep]
    # rename id col
    nodes_df.rename(columns={'id': rename_id_col}, inplace=True)
    return nodes_df


def make_hpo_closures_and_graph(
        url = 'https://kg-hub.berkeleybop.io/kg-obo/hp/2023-04-05/hp_kgx_tsv.tar.gz',
        pred_col='predicate',
        subject_prefixes= ['HP:'],
        object_prefixes= ['HP:'],
        predicates = ['biolink:subclass_of'],
        root_node_to_use ='HP:0000118',
        include_self_in_closure=False,
    ) -> (List[Tuple], nx.DiGraph):
    # get tmp file name
    tmpdir = tempfile.TemporaryDirectory()
    tmpfile = tempfile.NamedTemporaryFile().file.name
    wget.download(url, tmpfile)

    this_tar = tarfile.open(tmpfile, 'r:gz')
    this_tar.extractall(path=tmpdir.name)

    # show files in tmpdir
    edge_files = [f for f in os.listdir(tmpdir.name) if 'edges' in f]
    if len(edge_files) != 1:
        raise RuntimeError("Didn't find exactly one edge file in {}".format(tmpdir.name))
    edge_file = edge_files[0]

    edges_df = pd.read_csv(os.path.join(tmpdir.name, edge_file), sep='\t')
    if pred_col not in edges_df.columns:
        raise RuntimeError("Didn't find predicate column {} in {} cols: {}".format(pred_col, edge_file, "\n".join(edges_df.columns)))

    # get edges of interest
    edges_df = edges_df[edges_df[pred_col].isin(predicates)]
    # get edges involving nodes of interest
    edges_df = edges_df[edges_df['subject'].str.startswith(tuple(subject_prefixes))]
    edges_df = edges_df[edges_df['object'].str.startswith(tuple(object_prefixes))]

    # make into list of tuples
    # note that we are swapping order of edges (object -> subject) so that descendants are leaf terms
    # and ancestors are root nodes (assuming edges are subclass_of edges)
    edges = list(edges_df[['object', 'subject']].itertuples(index=False, name=None))

    # Create a directed graph using NetworkX
    graph = nx.DiGraph(edges)

    # Create a subgraph from the descendants of phenotypic_abnormality
    descendants = nx.descendants(graph, root_node_to_use)
    pa_subgraph = graph.subgraph(descendants)

    def compute_closure(node):
        return set(nx.ancestors(graph, node))

    # Compute closures for each node
    closures = []
    # set message for tqdm

    for node in tqdm(pa_subgraph.nodes(), desc="Computing closures"):
        if include_self_in_closure:
            closures.append((node, 'dummy_predicate', node))
        for anc in compute_closure(node):
            closures.append((node, 'dummy_predicate', anc))

    return closures, graph


def make_test_train_splits(pt_df,
                           pt_id_col='person_id',
                           pt_label_col='patient_label',
                           num_splits=5,
                           shuffle=True,
                           seed=42) -> list[dict]:
    # given a pandas df with these columns
    # 'person_id', 'hpo_term_id', 'hpo_term_label', 'excluded', 'patient_label'

    # return a list of num_splits dicts with keys 'train' and 'test' containing
    # pandas dfs with the train/test split for each fold

    # make test/train split
    skf = StratifiedKFold(n_splits=num_splits, shuffle=shuffle, random_state=seed)

    pt_id_df = pt_df[[pt_id_col, pt_label_col]].drop_duplicates()

    train_test_kfolds = []
    for i, (train_index, test_index) in enumerate(skf.split(pt_id_df, pt_id_df[pt_label_col])):
        # make a pandas df with train, another with test
        train_test_kfolds.append({
            'train': pt_df.iloc[train_index],
            'test': pt_df.iloc[test_index]
        })
    return train_test_kfolds


def make_cohort(data, disease, negatives,
                drop_duplicates_phenotypes=True,
                expected_vals_for_excluded_col=['observed', 'excluded'],
                ) -> pd.DataFrame:
    # make a pandas dataframe with all disease cases as positive and negative cases as
    # negative
    these_columns=['person_id', 'hpo_term_id', 'hpo_term_label', 'excluded', 'patient_label']

    # data looks like
    # data['Marfan syndrome']['patient_id'] = [list of phenotypes]

    # phenotypes look like:
    # ('HP:0011968', 'Feeding difficulties', 'observed')

    # columns:
    # person_id, hpo_term_id, hpo_term_label, excluded, patient_label

    # make positive cases
    pos_cases = []
    for pt in data[disease]:
        for phenotype in data[disease][pt]:
            pos_cases.append([pt, phenotype[0], phenotype[1], phenotype[2], 1])
    pos_df = pd.DataFrame(pos_cases, columns=these_columns)

    # make negative cases
    neg_cases = []
    for disease in negatives:
        for pt in data[disease]:
            for phenotype in data[disease][pt]:
                neg_cases.append([pt, phenotype[0], phenotype[1], phenotype[2], 0])
    neg_df = pd.DataFrame(neg_cases, columns=these_columns)

    pt_df = pd.concat([pos_df, neg_df], ignore_index=True)

    if 'excluded' not in pt_df.columns or set(list(pt_df['excluded'].unique())) != set(expected_vals_for_excluded_col):
        raise RuntimeError("Didn't get the expected values for the 'excluded' column: {}".format(str(list(pt_df['excluded'].unique()))))

    # convert observed/excluded to boolean
    pt_df['excluded'] = pt_df['excluded'].apply(lambda x: True if x == 'excluded' else False)

    if drop_duplicates_phenotypes:
        pt_df = pt_df.drop_duplicates(subset=['person_id', 'hpo_term_id', 'excluded', 'patient_label'])

    return pt_df


def parse_phenopackets(directory_path) -> dict:
    json_files = []
    if not os.path.exists(directory_path):
        raise RuntimeError("Didn't find directory {}".format(directory_path))
    for foldername, subfolders, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.json'):
                json_files.append(os.path.join(foldername, filename))
    parsed_full_phenopacket_data = defaultdict(list)

    for json_file in json_files:
        # Open and parse the JSON file
        with (open(json_file, 'r') as file):
            try:
                data = json.load(file)

                # get diseases for this phenopacket
                diseases = get_diseases(data)
                for d in diseases:
                    parsed_full_phenopacket_data[d].append(data)

            except Exception as e:
                print(f"Error parsing {json_file}: {e}")

    # go through parsed_full_phenopacket_data and extract phenotypic features for each patient
    parsed_phenotypes = extract_phenotypes(parsed_full_phenopacket_data)

    return {
        'all_data': parsed_full_phenopacket_data,
        'phenotype_data': parsed_phenotypes
    }


def extract_phenotypes(full_data) -> dict:
    extracted_phenotypes = defaultdict(dict)
    for disease, data in full_data.items():
        # this_pt_phenotypes = []
        for pt in data:
            if 'phenotypicFeatures' in pt:
                for p in pt['phenotypicFeatures']:
                    # add this phenotype to the set of phenotypes
                    this_p = (p['type']['id'],
                              p['type']['label'],
                              'excluded' if 'excluded' in p else 'observed')
                    if pt['id'] not in extracted_phenotypes[disease]:
                        extracted_phenotypes[disease][pt['id']] = []
                    extracted_phenotypes[disease][pt['id']].append(this_p)
    return extracted_phenotypes


def get_diseases(data) -> list:
    diseases = set()
    if 'diseases' in data:
        for d in data['diseases']:
            if d['excluded'] is not True:
                # add this disease to the set of diseases
                diseases.add(d['term']['label'])

    if 'interpretations' in data:
        for i in data['interpretations']:
            diseases.add(i['diagnosis']['disease']['label'])

    return list(diseases)


def initialize_profiles(all_hpo_terms: list,
                        n_profiles: int,
                        hpo_terms_per_profile: int,
                        fraction_negated_terms: float = 0.1,
                        ) -> pd.DataFrame:

    # make empty df
    new_profiles = []
    for i in list(range(n_profiles)):
        for j in list(range(hpo_terms_per_profile)):
            hpo_term = random.choice(all_hpo_terms)
            negated = True if random.random() < fraction_negated_terms else False
            new_profiles.append([i, hpo_term, round(random.random(), 3), negated])
    return pd.DataFrame(new_profiles, columns=['profile_id', 'hpo_term_id', 'weight', 'negated'])


def compare_profiles_to_patients(
        semsimian,
        pt_train_df: pd.DataFrame,
        profiles_pd: pd.DataFrame,
        debug=False):
    # Assuming pt_train_df and profiles_pd are your pandas DataFrames

    # Get unique person_ids and profile_ids
    person_ids = pt_train_df['person_id'].unique()
    profile_ids = profiles_pd['profile_id'].unique()

    def compare_person_and_profile(person_id, profile_id):
        this_pt = pt_train_df[pt_train_df['person_id'] == person_id]
        this_pt = this_pt[['hpo_term_id', 'weight', 'negated']]
        this_pt_tuples = list(this_pt.itertuples(index=False, name=None))

        this_profile = profiles_pd[profiles_pd['profile_id'] == profile_id]
        this_profile = this_profile[['hpo_term_id', 'weight', 'negated']]
        this_profile_tuples = list(this_profile.itertuples(index=False, name=None))

        # from embiggen.similarities import DAGResnik
        # from ensmallen.datasets.kgobo import HP
        # hp = HP(directed=True)\
        #     .filter_from_names(
        #         edge_type_names_to_keep=["biolink:subclass_of"],
        #         node_prefixes_to_keep=["HP:"]
        # )\
        #     .to_transposed()\
        #     .remove_disconnected_nodes()
        # hp.enable()
        # hp_model = DAGResnik()
        # hp_model.fit(hp, node_counts={node_name: 1
        #                               for node_name in hp.get_node_names()})

        return semsimian.termset_pairwise_similarity_weighted_negated(
            subject_dat=this_pt_tuples,
            object_dat=this_profile_tuples
        )

    # Generate all combinations of person_ids and profile_ids
    combinations = list(itertools.product(person_ids, profile_ids))

    # Apply the function to all combinations using pd.DataFrame.apply()
    result = pd.DataFrame(combinations, columns=['person_id', 'profile_id'])
    result['similarity'] = result.apply(lambda row: compare_person_and_profile(row['person_id'], row['profile_id']), axis=1)
    return result


def make_ancestors_list(spo):
    # Create a dictionary to store ancestors for each node
    ancestors_dict = {}
    for node, _, ancestor in spo:
        if node not in ancestors_dict:
            ancestors_dict[node] = []
        ancestors_dict[node].append(ancestor)

    # Convert the dictionary to a list of tuples for DataFrame
    result_data = [(node, ancestors_dict[node]) for node in ancestors_dict]

    # Create a Pandas DataFrame
    df = pd.DataFrame(result_data, columns=['hpo_term_id', 'ancestors'])
    return df


def move_terms_on_hierarchy(profiles, move_term_p, hpo_graph, include_list=None, debug=False):
    """Randomly move a term to parent or child on the hierarchy with a frequency of
    move_term_p
    """
    def move_term_on_hierarchy(term,
                               include_list=include_list,
                               hpo_graph=hpo_graph,
                               debug=debug):
        if random.random() < move_term_p:
            if random.random() < 0.5:  # move up
                candidate_terms = list(hpo_graph.predecessors(term))  # get parent(s)
            else:  # move down
                candidate_terms = list(hpo_graph.successors(term))  # get children
            # remove candidates that are not in the include_list
            if include_list is not None:
                candidate_terms = [a for a in candidate_terms if a in include_list]
            if candidate_terms:
                return random.choice(candidate_terms)
            else:
                if debug:
                    warnings.warn(f"Didn't find any candidate parents or children for {term}")
                return term
        else:
            return term

    profiles.reset_index(drop=True, inplace=False)
    profiles['hpo_term_id'] = profiles['hpo_term_id'].apply(move_term_on_hierarchy)
    return profiles


def run_genetic_algorithm(
        semsimian,
        disease: str,
        pt_train_df: pd.DataFrame,
        pt_test_df: pd.DataFrame,
        hpo_graph: nx.DiGraph,
        node_labels: Optional[pd.DataFrame] = None,
        hyper_n_iterations=100,
        hyper_pt_dropout_fraction=0.2,
        hyper_n_profile_pop_size=100,
        hyper_n_initial_hpo_terms_per_profile=5,
        hyper_n_fraction_negated_terms=0.1,
        hyper_n_best_profiles=20,
        hyper_fitness_auc='auprc',
        hyper_add_term_p=0.1,
        hyper_initialize_and_add_terms_only_from_observed_terms=False,
        hyper_remove_term_p=0.3,
        hyper_change_weight_p=0.5,
        hyper_move_term_on_hierarchy_p=0.75,
        debug=False,
    ) -> dict[str, pd.DataFrame]:

    # overall strategy:
    # 1. initialize profiles
    # 2. for each iteration:
    #    a. run termset similarity for each profile vs train split
    #    b. calculate AUC for each profile
    #    c. select top N profiles
    #    d. recombine profiles
    #    e. add/remove terms from profiles
    #    f. change weights for profiles
    #    g. move terms on hierarchy to parent or child
    # 3. run termset similarity for each profile vs test split

    if hyper_initialize_and_add_terms_only_from_observed_terms:
        all_hpo_terms = list(pt_train_df['hpo_term_id'].unique())
    else:
        all_hpo_terms = list(set([e[0] for e in semsimian.get_spo()]))

    # make ancestor pd df
    ancestors_pd = make_ancestors_list(semsimian.get_spo())

    profiles_pd = initialize_profiles(all_hpo_terms=all_hpo_terms,
                                      n_profiles=hyper_n_profile_pop_size,
                                      fraction_negated_terms=hyper_n_fraction_negated_terms,
                                      hpo_terms_per_profile=hyper_n_initial_hpo_terms_per_profile)

    fitness_by_iteration = []

    progress_bar = tqdm(total=hyper_n_iterations, desc="Running genetic algorithm")

    for i in list(range(hyper_n_iterations)):

        # run termset similarity for each profile vs each patient in train split
        sim_results = compare_profiles_to_patients(semsimian=semsimian,
                                                   # apply dropout to pt_train_df:
                                                   pt_train_df=pt_train_df[pt_train_df['person_id'].isin(pt_train_df['person_id'].sample(frac=hyper_pt_dropout_fraction))],
                                                   profiles_pd=profiles_pd,
                                                   debug=debug)
        train_auc_results = make_auc_df(
            sim_results=sim_results,
            patient_labels=pt_train_df[['person_id', 'patient_label']].drop_duplicates()
        )

        # add results to fitness_by_iteration
        new_row = train_auc_results.copy()
        new_row['iteration'] = i
        fitness_by_iteration += [new_row]

        top_n_profiles = train_auc_results.sort_values(by=hyper_fitness_auc,
                                                       ascending=False).head(hyper_n_best_profiles)['profile_id']

        profiles_pd = profiles_pd[profiles_pd['profile_id'].isin(top_n_profiles)]

        # output info for top profile
        pd.set_option('display.max_columns', None)  # pandas smdh
        best = profiles_pd[profiles_pd['profile_id'].isin(top_n_profiles[:1])]
        # sort by weight
        best = best.sort_values(by='weight', ascending=False, inplace=False)
        if node_labels is not None:
            best = best.merge(node_labels, on="hpo_term_id")
            print("\n" + str(best[['hpo_term_id', 'weight', 'negated', 'name']]))
        else:
            print("\n" + str(best[['hpo_term_id', 'weight', 'negated']]))

        profiles_pd = recombine_profiles_pd(profiles=profiles_pd,
                                            ancestors_df=ancestors_pd,
                                            num_profiles=hyper_n_profile_pop_size)

        progress_bar.set_description(
            "Running genetic algorithm - {} for iteration {}: {}".format(
                hyper_fitness_auc, i+1, round(train_auc_results[hyper_fitness_auc].mean(), 2)))

        profiles_pd = add_terms_to_profiles_pd(profiles=profiles_pd,
                                               all_hpo_terms=all_hpo_terms,
                                               ancestor_list=ancestors_pd,
                                               add_term_p=hyper_add_term_p,
                                               fraction_negated_terms=hyper_n_fraction_negated_terms)

        profiles_pd = remove_terms_from_profiles_pd(profiles=profiles_pd,
                                                    remove_term_p=hyper_remove_term_p)

        profiles_pd = change_weights_for_profiles(profiles=profiles_pd,
                                                  change_weight_p=hyper_change_weight_p)

        profiles_pd = move_terms_on_hierarchy(profiles=profiles_pd,
                                              move_term_p=hyper_move_term_on_hierarchy_p,
                                              hpo_graph=hpo_graph,
                                              # to make sure we don't move terms to
                                              # terms that aren't an s in the spo
                                              # to keep semsimian happy
                                              include_list=[t[0] for t in
                                                            semsimian.get_spo()],
                                              debug=debug)

        # increment progress bar
        progress_bar.update(1)

    progress_bar.close()

    test_results = compare_profiles_to_patients(semsimian=semsimian,
                                                pt_train_df=pt_test_df,
                                                profiles_pd=profiles_pd,
                                                debug=debug)
    test_auc_results = make_auc_df(
        sim_results=test_results,
        patient_labels=pt_test_df[['person_id', 'patient_label']].drop_duplicates()
    )

    # add results to fitness_by_iteration
    new_row = test_auc_results.copy()
    new_row['iteration'] = -999
    fitness_by_iteration += [new_row]

    # output average AUC in fitness_by_iteration by iteration
    average_auc_by_iteration = pd.DataFrame(columns=['iteration', 'mean_auroc', 'mean_auprc'])
    for i in range(len(fitness_by_iteration)):
        average_auc_by_iteration.loc[i] = [i, fitness_by_iteration[i]['auroc'].mean(), fitness_by_iteration[i]['auprc'].mean()]

    # output profiles_pd to a file
    if node_labels is not None:
        profiles_pd = profiles_pd.merge(node_labels, on="hpo_term_id", how='left')

    # join in AUC results for each profile
    profiles_pd.reset_index(drop=True, inplace=True)
    test_auc_results.reset_index(drop=True, inplace=True)
    profiles_pd = profiles_pd.merge(test_auc_results, on="profile_id", how='left')
    profiles_pd.drop(['description'], axis=1, inplace=True)

    # sort profiles by AUC
    profiles_pd.sort_values(by=[hyper_fitness_auc, 'profile_id', 'weight'], ascending=False, inplace=True)

    # make outfile with all hyperparameters in its name
    outfile = "ga_results_{}_{}_iterations".format(disease, hyper_n_iterations)
    profiles_pd.to_csv(outfile + ".tsv", index=False, sep="\t")
    return {'test_auc': test_auc_results, 'train_auc': train_auc_results}


def make_auc_df(
        sim_results: pd.DataFrame,
        patient_labels: pd.DataFrame
) -> pd.DataFrame:

    sim_results_with_labels = sim_results.merge(patient_labels, on='person_id', how='inner')

    def calculate_auc_metrics(df):
        profile_id = df['profile_id'].iloc[0]
        y_true = df['patient_label'].values
        y_score = df['similarity'].values
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        return pd.DataFrame([[profile_id, auroc, auprc]],
                            columns=['profile_id', 'auroc', 'auprc'])

    return sim_results_with_labels.groupby('profile_id').apply(calculate_auc_metrics)


def remove_terms_from_profiles_pd(profiles: pd.DataFrame,
                                  remove_term_p: float = 0.1) -> pd.DataFrame:
    """Randomly remove one term from a profile with a frequency of remove_term_p, IFF there
    is more than 1 HPO term for the profile
    profiles: A pandas dataframe describing profiles to evolve to predict an outcome
    remove_term_p: probability that a term will be removed
    """

    def drop_hpo_term(df):
        # Drop one HPO term with a probability of 0.1
        if len(df) > 1 and random.uniform(0, 1) < remove_term_p:
            df = df.drop(np.random.choice(list(df.index)))
        return df
    # df.index.name = None
    return profiles.reset_index(drop=True, inplace=False).groupby('profile_id').apply(drop_hpo_term)


def add_terms_to_profiles_pd(profiles: pd.DataFrame,
                             all_hpo_terms: list,
                             ancestor_list: pd.DataFrame,
                             fraction_negated_terms,
                             add_term_p: float = 0.1,
                             ) -> pd.DataFrame:

    def add_random_hpo_terms(df):
        # Remove ancestors of existing HPO terms from all_hpo_terms
        # here - this will help prevent incoherence in parent/children HPO terms

        # get ancestors of all terms in this profile
        # combine and explode ancestors
        # make list of candidate HPO terms to add:
        #    by subtracting out ancestors from all_hpo_terms
        if 'ancestors' in df.columns:
            terms_to_eliminate = set(list(df.explode('ancestors')['ancestors'].unique()))
        else:
            warnings.warn(f"Didn't find 'ancestors' column in df {' '.join(df.columns)} while adding random HPO term")
            return df
        possible_terms = list(set(all_hpo_terms) - set(terms_to_eliminate))

        profile_id = df['profile_id'].iloc[0]
        # Check if we should add a new row
        if random.random() < add_term_p:
            # Select a random HPO term and weight
            hpo_term = random.choice(possible_terms)

            if hpo_term not in df['hpo_term_id'].unique():
                new_row = {
                    'profile_id': profile_id,
                    'hpo_term_id': hpo_term,
                    'weight': round(random.random(), 3),
                    'negated': True if random.random() < fraction_negated_terms else False,
                    'ancestors': []  # None for ancestors column
                }

                # Append the new row to the DataFrame
                df.loc[max(list(df.index)) + 1] = new_row
                pass

        df = df.drop('ancestors', axis=1)
        return df

    # add ancestor information
    profiles = profiles.merge(ancestor_list, on='hpo_term_id', how='left')

    # Apply the add_random_hpo_terms function to each group
    profiles = profiles.reset_index(drop=True, inplace=False).groupby('profile_id').apply(add_random_hpo_terms)

    return profiles


def change_weights_for_profiles(profiles: pd.DataFrame,
                                change_weight_p: float = 0.1) -> pd.DataFrame:
    """Randomly change the weight of each term with a frequency of change_weight_p, by
    an amount equal to change_weight_fraction
    """
    def change_weight(weight):
        if random.random() < change_weight_p:
            if random.random() < 0.5:
                return min(1.0, round(random.random(), 3))
            else:
                return max(0.0, round(random.random(), 3))
        else:
            return weight

    profiles = profiles.reset_index(drop=True, inplace=False)
    profiles['weight'] = (profiles['weight'].apply(change_weight))
    return profiles


def recombine_profiles_pd(profiles: pd.DataFrame, ancestors_df: pd.DataFrame, num_profiles: int) -> pd.DataFrame:
    """
    Recombines profiles by randomly choosing pairs of profiles and applying the following logic:
    * choose a random HPO term
    * swap all descendants of the the term from profile1 -> profile2 and vice versa
    * return a new dataframe with num_profiles new profiles comprised of recombined profiles from parent dataframe

    Args:
    profiles_df: a pandas DataFrame with columns "profile_id", "HPO_term", and "weight".
    ancestors_df: a pandas DataFrame with columns "hpo_term_id" and "ancestors".
    num_profiles: how many profiles to return
    (NB: this must be an even number, since each swap produces two new profiles. Also occasionally this method
    will return fewer than num_profiles profiles if all HPO terms in profile1 are moved to profile2 or vice versa)

    Returns:
    A pandas DataFrame with columns "profile_id", "HPO_term", and "weight".
    """

    # Join profiles with ancestors to get a DataFrame with profile_id, HPO_term, weight, and ancestors
    joined_profiles_df = pd.merge(profiles, ancestors_df, on='hpo_term_id', how='left')

    all_pids = list(set(profiles['profile_id'].to_list()))

    # make random pairs of profile IDs (i.e. the parents)
    parent_profiles = pd.DataFrame(list(itertools.product(all_pids, all_pids)), columns=['p1', 'p2'])
    parent_profiles = parent_profiles.loc[parent_profiles['p1'] < parent_profiles['p2']].sample(n=num_profiles//2, replace=True)

    # Loop through pairs of profiles, and apply the recombination logic to each pair
    new_profiles_to_add = []
    for i, (profile1_id, profile2_id) in enumerate(parent_profiles[['p1', 'p2']].values):

        profile1 = joined_profiles_df[joined_profiles_df.profile_id == profile1_id].reset_index(drop=True)
        profile2 = joined_profiles_df[joined_profiles_df.profile_id == profile2_id].reset_index(drop=True)

        # assign new profile IDs
        new_profile1_id = i * 2 + 1
        new_profile2_id = i * 2 + 2
        profile1.loc[:, 'profile_id'] = new_profile1_id
        profile2.loc[:, 'profile_id'] = new_profile2_id

        # Apply the recombination logic to the current pair of profiles

        # pick random HPO term from first column of ancestor_list
        random_hpo_term = ancestors_df['hpo_term_id'].sample(n=1, random_state=1).iloc[0]

        tmp_rows = []
        for _, row in profile1.iterrows():
            if len(pd.isnull(row['ancestors'])) > 0 and not pd.isnull(row['hpo_term_id']) and random_hpo_term in row['ancestors']:
                # This row['HPO_term'] is a descendant of random_hpo_term, do the swap

                # Create new row for profile2 with swapped HPO term
                new_row = profile1.loc[profile1['hpo_term_id'] == row['hpo_term_id']].drop('profile_id', axis=1)
                new_row['profile_id'] = new_profile2_id
                tmp_rows.append(new_row)

                # Filter out and keep 'good' HPO terms/weights from profile1
                profile1 = profile1.loc[profile1['hpo_term_id'] != row['hpo_term_id']]

        rows_to_add_to_profile1 = []
        for _, row in profile2.iterrows():
            if len(pd.isnull(row['ancestors'])) > 0 and not pd.isnull(row['hpo_term_id']) and random_hpo_term in row['ancestors']:
                # This row['HPO_term'] is a descendant of random_hpo_term, do the swap

                # Filter out and keep 'good' HPO terms/weights from profile2
                profile2 = profile2.loc[profile2['hpo_term_id'] != row['hpo_term_id']]

                # Add row to list of rows to add to profile1
                try:
                    rows_to_add_to_profile1.append(pd.DataFrame([[new_profile1_id, row['hpo_term_id'], row['weight']]], columns=profiles.columns))
                # catch ValueError
                except ValueError as ve:
                    print("ValueError: {} while adding row['hpo_term_id'] {} and row['weight'] {} to profile1".format(ve, row['hpo_term_id'], row['weight']))

        # Add rows to profile1
        if rows_to_add_to_profile1:
            rows_to_add_to_profile1 = pd.concat(rows_to_add_to_profile1, ignore_index=True)
            profile1 = pd.concat(rows_to_add_to_profile1)
        else:
            profile1 = profile1[['profile_id', 'hpo_term_id', 'weight', 'negated']]

        # now complete swap by putting all tmp rows in profile2
        if tmp_rows:
            profile2 = pd.concat(tmp_rows)
        else:
            profile2 = profile2[['profile_id', 'hpo_term_id', 'weight', 'negated']]

        new_profiles_to_add.append(profile1)
        new_profiles_to_add.append(profile2)

    # Return the recombined profiles
    new_profiles = pd.concat(new_profiles_to_add, ignore_index=True)
    try:
        new_profiles = new_profiles.drop_duplicates()
    except TypeError as te:
        pass

    return new_profiles
