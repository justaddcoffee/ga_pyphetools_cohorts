import random
import warnings
import pandas as pd
import numpy as np
import itertools

from sklearn.metrics import roc_auc_score, average_precision_score


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
        terms_to_eliminate = set(list(df.explode('ancestors')['ancestors'].unique()))
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
