import random
import tarfile
import warnings
import networkx as nx
from typing import List, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from collections import defaultdict
import os
import json
import tempfile
import wget
from tqdm import tqdm


def make_hpo_closures(
        url = 'https://kg-hub.berkeleybop.io/kg-obo/hp/2023-04-05/hp_kgx_tsv.tar.gz',
        pred_col='predicate',
        subject_prefixes= ['HP:'],
        object_prefixes= ['HP:'],
        predicates = ['biolink:subclass_of'],
        root_node_to_use ='HP:0000118'
    ) -> List[Tuple]:
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
        for anc in compute_closure(node):
            closures.append((node, 'dummy_predicate', anc))

    return closures


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
            new_profiles.append([i, hpo_term, random.random(), negated])
    return pd.DataFrame(new_profiles, columns=['profile_id', 'hpo_term_id', 'weight', 'negated'])


def run_genetic_algorithm(
        semsimian,
        spo: list[tuple],
        pt_train_df: pd.DataFrame,
        pt_test_df: pd.DataFrame,

        hyper_n_iterations=60,
        hyper_n_profile_pop_size=100,
        hyper_n_initial_hpo_terms_per_profile=5,
        hyper_n_fraction_negated_terms=0.1,
        hyper_n_best_profiles=20,
        hyper_fitness_auc='auprc',
        hyper_add_term_p=0.3,
        hyper_remove_term_p=0.3,
        hyper_change_weight_p=0.3,
        hyper_change_weight_fraction=0.2,
    ):

    # overall strategy:
    # 1. initialize profiles
    # 2. for each iteration:
    #    a. run termset similarity for each profile vs train split
    #    b. calculate AUC for each profile
    #    c. select top N profiles
    #    d. recombine profiles
    #    e. add/remove terms from profiles
    #    f. change weights for profiles
    # 3. run termset similarity for each profile vs test split

    # test some random similarities
    for term1 in random.sample(list(pt_train_df['hpo_term_id'].unique()), 10):
        for term2 in random.sample(list(pt_train_df['hpo_term_id'].unique()), 10):
            sim = semsimian.termset_pairwise_similarity_weighted_negated(
                subject_dat=[(term1, 2, False)], object_dat=[(term2, 2, False)])
            print("sim between {} and {}: {}".format(term1, term2, sim))

    # pandas df with profile data

    all_hpo_terms = list(set([e[0] for e in spo]))

    profiles_pd = initialize_profiles(all_hpo_terms=all_hpo_terms,
                                      n_profiles=hyper_n_profile_pop_size,
                                      fraction_negated_terms=hyper_n_fraction_negated_terms,
                                      hpo_terms_per_profile=hyper_n_initial_hpo_terms_per_profile)

    array_of_fitness_results = []

    for i in tqdm(list(range(hyper_n_iterations)), desc="Running genetic algorithm"):

        # make tuples for one example patient
        this_pt = pt_train_df[pt_train_df['person_id'] == 'PMID_12203992_B3']
        this_pt['weight'] = 1  # all patient phenotypes are weighted equally
        this_pt = this_pt[['hpo_term_id', 'weight', 'negated']]
        this_pt_tuples = list(this_pt.itertuples(index=False, name=None))

        # make tuples for one profile
        this_profile = profiles_pd[profiles_pd['profile_id'] == 0]
        this_profile = this_profile[['hpo_term_id', 'weight', 'negated']]
        this_profile_tuples = list(this_profile.itertuples(index=False, name=None))

        semsimian.termset_pairwise_similarity_weighted_negated(
            subject_dat=this_pt_tuples,
            object_dat=this_profile_tuples
        )

        phenomizer_results = p.compare_two_dfs_similarity_long_spark_df(df1=train_split,
                                                                        df1_id_col='person_id',
                                                                        df1_hpo_term_col='hpo_term_id',
                                                                        mica_df=mica_df,
                                                                        df2=spark.createDataFrame(
                                                                            profiles_pd,
                                                                            schema=profile_schema),
                                                                        df2_id_col='profile_id',
                                                                        df2_hpo_term_col='HPO_term')

        phenomizer_results_pd = phenomizer_results.toPandas()
        phenomizer_results_pd = phenomizer_results_pd.merge(patient_labels_train,
                                                            left_on='id1',
                                                            right_on='person_id',
                                                            how='inner')

        def calculate_auc_metrics(df):
            profile_id = df['id2'].iloc[0]
            y_true = df['label'].values
            y_score = df['similarity'].values
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
            return pd.DataFrame([[profile_id, auroc, auprc]],
                                columns=['profile_id', 'auroc', 'auprc'])


        phenomizer_metrics_pd = phenomizer_results_pd.groupby('id2').apply(
            calculate_auc_metrics)

        # add results to array_of_fitness_results
        new_row = phenomizer_metrics_pd
        new_row['iteration'] = i
        array_of_fitness_results += [new_row]

        top_n_profiles = \
        phenomizer_metrics_pd.sort_values(by=hyper_fitness_auc, ascending=False).head(
            hyper_n_best_profiles)['profile_id'].to_list()
        profiles_pd = profiles_pd[profiles_pd['profile_id'].isin(top_n_profiles)]

        profiles_pd = recombine_profiles_pd(profiles=profiles_pd,
                                            ancestors_df=ancestor_list.toPandas(),
                                            num_profiles=hyper_n_profile_pop_size)
        profiles_pd = add_terms_to_profiles_pd(profiles=profiles_pd,
                                               all_hpo_terms=all_hpo_terms,
                                               ancestor_list=ancestor_list.toPandas(),
                                               add_term_p=hyper_add_term_p)
        profiles_pd = remove_terms_from_profiles_pd(profiles=profiles_pd,
                                                    remove_term_p=hyper_remove_term_p)
        # profiles = change_weights_for_profiles(profiles=profiles,
        #                                        change_weight_p=hyper_change_weight_p,
        #                                        change_weight_fraction=hyper_change_weight_fraction)

    # Compute performance of last iteration of profiles on test data
    phenomizer_results = p.compare_two_dfs_similarity_long_spark_df(df1=test_split,
                                                                    df1_id_col='person_id',
                                                                    df1_hpo_term_col='hpo_term_id',
                                                                    mica_df=mica_df,
                                                                    df2=spark.createDataFrame(profiles_pd, schema=profile_schema),
                                                                    df2_id_col='profile_id',
                                                                    df2_hpo_term_col='HPO_term')
    phenomizer_results_pd = phenomizer_results.toPandas()
    phenomizer_results_pd = phenomizer_results_pd.merge(patient_labels_test, left_on='id1', right_on='person_id', how='inner')

    phenomizer_metrics_pd = phenomizer_results_pd.groupby('id2').apply(calculate_auc_metrics)

    # add results to array_of_fitness_results
    new_row = phenomizer_metrics_pd
    new_row['iteration'] = -999  # hack to mean performance on test data
    array_of_fitness_results += [new_row]

    # fitness_by_iteration = fitness_by_iteration.union(union_many(*array_of_fitness_results))
    fitness_by_iteration_pd = pd.concat(array_of_fitness_results)
    fitness_by_iteration = spark.createDataFrame(data=fitness_by_iteration_pd)

    output_profiles.write_dataframe(spark.createDataFrame(profiles_pd, schema=profile_schema))
    output_metrics.write_dataframe(fitness_by_iteration)

def remove_terms_from_profiles(profiles: pd.DataFrame,
                               remove_term_p: float = 0.1) -> pd.DataFrame:
    """Randomly remove one term from a profile with a frequency of remove_term_p, IFF there
    is more than 1 HPO term for the profile
    profiles: A spark dataframe describing profiles to evolve to predict an outcome
    remove_term_p: probability that a term will be removed
    """
    # choose random number to decide remove existing term or not (10%?)
    # if yes, remove random term
    check_schema(profiles=profiles)

    @pandas_udf(profile_schema, PandasUDFType.GROUPED_MAP)
    def drop_hpo_term(df):
        # Drop one HPO term with a probability of 0.1
        if len(df) > 1 and random.uniform(0, 1) < remove_term_p:
            index_to_drop = np.random.randint(0, len(df))
            df = df.drop(index_to_drop)
        return df

    return profiles.reset_index(drop = True, inplace = False).apply(drop_hpo_term)

def remove_terms_from_profiles_pd(profiles: pd.DataFrame,
                                  remove_term_p: float = 0.1) -> pd.DataFrame:
    """Randomly remove one term from a profile with a frequency of remove_term_p, IFF there
    is more than 1 HPO term for the profile
    profiles: A pandas dataframe describing profiles to evolve to predict an outcome
    remove_term_p: probability that a term will be removed
    """
    check_schema(profiles=profiles)

    def drop_hpo_term(df):
        # Drop one HPO term with a probability of 0.1
        if len(df) > 1 and random.uniform(0, 1) < remove_term_p:
            df = df.drop(np.random.choice(list(df.index)))
        return df
    # df.index.name = None
    return profiles.reset_index(drop=True, inplace=False).groupby('profile_id').apply(drop_hpo_term)


def add_terms_to_profiles(profiles: pd.DataFrame,
                          all_hpo_terms: list,
                          ancestor_list: pd.DataFrame,
                          add_term_p: float = 0.1,
                          ) -> pd.DataFrame:
    check_schema(profiles=profiles)

    # Define the Pandas UDF function
    @pandas_udf(profile_schema, PandasUDFType.GROUPED_MAP)
    def add_random_hpo_terms(df):
        # Remove ancestors of existing HPO terms from all_hpo_terms
        # here - this will help prevent incoherence in parent/children HPO terms

        # get ancestors of all terms in this profile
        # combine and explode ancestors
        # make list of candidate HPO terms to add:
        #    by subtracting out ancestors from all_hpo_terms
        terms_to_eliminate = set(list(df.explode('ancestors')['ancestors'].unique()))
        possible_terms = list(set(all_hpo_terms) - set(terms_to_eliminate))

        profile_id = df['profile_id'].iloc[0]
        # Check if we should add a new row
        if random.random() < add_term_p:
            # Select a random HPO term and weight
            hpo_term = random.choice(possible_terms)

            if hpo_term not in df['HPO_term'].unique():
                weight = random.random()
                # Create a new row with the random values
                new_row = pd.DataFrame({"profile_id": [profile_id],
                                        "HPO_term": [hpo_term],
                                        "weight": [weight],
                                        })
                # Append the new row to the DataFrame
                df = df.append(new_row, ignore_index=True)

        df = df.drop('ancestors', axis=1)
        return df

    # add ancestor information
    profiles = profiles.join(ancestor_list, (profiles.HPO_term == ancestor_list.hpo_term_id), how='left')
    profiles = profiles.drop('hpo_term_id')

    # for testing
    # profiles.toPandas().groupby("profile_id").apply(add_random_hpo_terms)

    # Apply the Pandas UDF function to the Spark DataFrame
    profiles = profiles.groupby("profile_id").apply(add_random_hpo_terms)

    return profiles


def add_terms_to_profiles_pd(profiles: pd.DataFrame,
                             all_hpo_terms: list,
                             ancestor_list: pd.DataFrame,
                             add_term_p: float = 0.1,
                             ) -> pd.DataFrame:

    def add_random_hpo_terms(df):
        # Remove ancestors of existing HPO terms from all_hpo_terms
        # here - this will help prevent incoherence in parent/children HPO terms

        # get ancestors of all terms in this profile
        # combine and explode ancestors
        # make list of candidate HPO terms to add:
        #    by subtracting out ancestors from all_hpo_terms
        terms_to_eliminate = set(list(df.explode('ancestors')['ancestors'].unique()))
        possible_terms = list(set(all_hpo_terms) - set(terms_to_eliminate))

        profile_id = df['profile_id'].iloc[0]
        # Check if we should add a new row
        if random.random() < add_term_p:
            # Select a random HPO term and weight
            hpo_term = random.choice(possible_terms)

            if hpo_term not in df['HPO_term'].unique():
                weight = random.random()
                # Create a new row with the random values
                new_row = pd.DataFrame({"profile_id": [profile_id],
                                        "HPO_term": [hpo_term],
                                        "weight": [weight],
                                        })
                # Append the new row to the DataFrame
                df = df.append(new_row, ignore_index=True)

        df = df.drop('ancestors', axis=1)
        return df

    # add ancestor information
    profiles = profiles.merge(ancestor_list, left_on='HPO_term', right_on='hpo_term_id', how='left')
    profiles = profiles.drop('hpo_term_id', axis=1)

    # Apply the add_random_hpo_terms function to each group
    profiles = profiles.groupby('profile_id').apply(add_random_hpo_terms)

    return profiles


def change_weights_for_profiles(profiles: pd.DataFrame,
                                change_weight_p: float = 0.1,
                                change_weight_fraction: float = 0.2
                                ) -> pd.DataFrame:
    check_schema(profiles=profiles)

    # choose random number to decide remove existing term or not (10%?)
    # if yes, change weight
    def change_weight(weight):
        if random.random() < change_weight_p:
            #  TODO: make sure weight is never < 0 or > 1
            if random.random() < 0.5:
                return min(1.0, weight * (1 + change_weight_fraction))
            else:
                return max(0.0, weight * (1 - change_weight_fraction))
        else:
            return weight

    change_weight_udf = udf(change_weight, FloatType())

    return profiles.withColumn("weight", change_weight_udf(F.col("weight")))


def run_phenomizer(profiles: pd.DataFrame,
                   patient_data: pd.DataFrame,
                   patient_labels: pd.DataFrame,
                   mica_df: pd.DataFrame) -> pd.DataFrame:
    """Given a set of profiles and patient data, run phenomizer on profiles vs patient data and return metrics on profiles (AUROC, AUPRC)
    based on the performance of each profile in predicting label
    profiles: a dataframe containing HPO terms for each profile
    patient_data: a dataframe containing HPO terms for each patient
    patient_labels: a dataframe indicating whether pt has the outcome of interest
    mica_df: mica DF for the phenomizer comparison (i.e. output of make_mica_df() in semanticsimilarity package)
    """

    # TODO: I think we may want to resolve child/ancestors by only considering leaves in profiles

    p = Phenomizer({})

    phenomizer_results = p.compare_two_dfs_similarity_long_spark_df(df1=patient_data,
                                                                    df1_id_col='person_id',
                                                                    df1_hpo_term_col='hpo_id',
                                                                    mica_df=mica_df,
                                                                    df2=profiles,
                                                                    df2_id_col='profile_id',
                                                                    df2_hpo_term_col='HPO_term')

    # #  +---+---+----------+
    #  |id1|id2|similarity|
    #  +---+---+----------+
    #  |  1|  1|       0.0|
    #  |  1|  2|0.88399124|
    #  |  1|  3|       0.0|
    #  |  1|  4|       0.0|
    #  |  1|  5|       0.0|
    #  |  1|  6|       0.0|

    # join in labels: NB THIS WILL IGNORE PATIENTS WITHOUT LABELS BC OF INNER JOIN

    phenomizer_results = \
        phenomizer_results.join(patient_labels, (phenomizer_results.id1 == patient_labels.person_id), how='inner')

    phenomizer_results = phenomizer_results.withColumn('similarity', F.col('similarity').cast('double'))

    #
    # calculate AUC
    #

    # make empty df
    phenomizer_metrics_schema = StructType([
        StructField("profile_id", IntegerType()),
        StructField("auroc", DoubleType()),
        StructField("auprc", DoubleType())
    ])

    @pandas_udf(phenomizer_metrics_schema, PandasUDFType.GROUPED_MAP)
    def calculate_auc_metrics(df):
        profile_id = df['id2'].iloc[0]
        y_true = df['label'].values
        y_score = df['similarity'].values
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        return pd.DataFrame([[profile_id, auroc, auprc]], columns=['profile_id', 'auroc', 'auprc'])

    phenomizer_metrics_df = phenomizer_results.groupBy('id2').apply(calculate_auc_metrics)

    return phenomizer_metrics_df.toPandas()


def check_schema(profiles, expected_cols=["profile_id", "HPO_term", "weight"]):
    if set(profiles.columns) != set(expected_cols):
        raise RuntimeError("Didn't get the expected columns {}", str(" ".join(expected_cols)))


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
    joined_profiles_df = pd.merge(profiles, ancestors_df, left_on='HPO_term', right_on='hpo_term_id', how='left').drop('hpo_term_id', axis=1)

    # p1 = profiles[['profile_id']].drop_duplicates()
    # p1 = p1.rename(columns={"profile_id": "p1"})
    # p2 = profiles[['profile_id']].drop_duplicates()
    # p2 = p2.rename(columns={"profile_id": "p2"})

    from itertools import product
    all_pids = list(set(profiles['profile_id'].to_list()))

    # make random pairs of profile IDs (i.e. the parents)
    parent_profiles = pd.DataFrame(list(product(all_pids, all_pids)), columns=['p1', 'p2'])
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
            if not row['ancestors'] and not pd.isnull(row['HPO_term']) and random_hpo_term in row['ancestors']:
                # This row['HPO_term'] is a descendant of random_hpo_term, do the swap

                # Create new row for profile2 with swapped HPO term
                new_row = profile1.loc[profile1['HPO_term'] == row['HPO_term']].drop('profile_id', axis=1)
                new_row['profile_id'] = new_profile2_id
                tmp_rows.append(new_row)

                # Filter out and keep 'good' HPO terms/weights from profile1
                profile1 = profile1.loc[profile1['HPO_term'] != row['HPO_term']]

        rows_to_add_to_profile1 = []
        for _, row in profile2.iterrows():
            if not row['ancestors'] and not pd.isnull(row['HPO_term']) and random_hpo_term in row['ancestors']:
                # This row['HPO_term'] is a descendant of random_hpo_term, do the swap

                # Filter out and keep 'good' HPO terms/weights from profile2
                profile2 = profile2.loc[profile2['HPO_term'] != row['HPO_term']]

                # Add row to list of rows to add to profile1
                rows_to_add_to_profile1.append(pd.DataFrame([[new_profile1_id, row['HPO_term'], row['weight']]], columns=profiles.columns))

        # Add rows to profile1
        if rows_to_add_to_profile1:
            rows_to_add_to_profile1 = pd.concat(rows_to_add_to_profile1, ignore_index=True)
            profile1 = pd.concat(rows_to_add_to_profile1)
        else:
            profile1 = profile1[['profile_id', 'HPO_term', 'weight']]

        # now complete swap by putting all tmp rows in profile2
        if tmp_rows:
            profile2 = pd.concat(tmp_rows)
        else:
            profile2 = profile2[['profile_id', 'HPO_term', 'weight']]

        new_profiles_to_add.append(profile1)
        new_profiles_to_add.append(profile2)

    # Return the recombined profiles
    new_profiles = pd.concat(new_profiles_to_add, ignore_index=True)
    new_profiles = new_profiles.drop_duplicates()

    return new_profiles


def extract_holdout(df_pts=None, holdouts=None, no_holdout=1, training=1, debug=False):
    """
        this function takes a VERTICAL patient dataframe (with 3 columns: person_id, HPO, label) and extracts the
        train/test split for the no_holdout stratified holdout (10:90 test:train ratio)
        Args:
        df_pts: the VERTICAL patient dataframe
        holdouts: the indicator dataframe where each column represents one holdout (1 = train/ 0 = test). This dataframe has columns:
        - person_id: patient identifier
        - label: the patient label
        - holdout_x (x = 1:50): each entry in this column equals 1 (0) if the pt with the corresponding person_id is in
        the training (test) set
        NOTE: these holdouts are also used by RF here:https://unite.nih.gov/workspace/vector/view/ri.vector.main.workbook.ec226190-9c8a-4264-bb18-14459a0f6fef?branch=master
        no_holdout: the holdout to extract (default = 1)
        training: extract training (training = 1) or test (training = 0) split (default = training = 1)
     example: if I want to extract the train-split from the third holdout then I can call
     train_split = extract_holdout(complete_df, holdout_splts, no_holdout = 3, 1)
    """
    name_holdout_col = 'holdout_' + str(no_holdout)
    holdout_splits = holdouts.select(*['person_id', name_holdout_col])

    joined_df = df_pts.join(holdout_splits, on=["person_id"])
    train_split = joined_df.filter(F.col(name_holdout_col) == training)
    train_split = train_split.drop(name_holdout_col)
    return train_split
