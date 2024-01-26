from typing import Optional

import networkx as nx
import pandas as pd
from tqdm import tqdm

from ga.utils.ga import (add_terms_to_profiles_pd, change_weights_for_profiles,
                         compare_profiles_to_patients, initialize_profiles,
                         make_ancestors_list, make_auc_df,
                         move_terms_on_hierarchy, recombine_profiles_pd,
                         remove_terms_from_profiles_pd)


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
    hyper_fitness_auc="auprc",
    hyper_add_term_p=0.1,
    hyper_initialize_and_add_terms_only_from_observed_terms=False,
    hyper_remove_term_p=0.2,
    hyper_change_weight_p=0.1,
    hyper_move_term_on_hierarchy_p=0.2,
    debug=False,
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
    #    g. move terms on hierarchy to parent or child
    # 3. run termset similarity for each profile vs test split

    if hyper_initialize_and_add_terms_only_from_observed_terms:
        all_hpo_terms = list(pt_train_df["hpo_term_id"].unique())
    else:
        all_hpo_terms = list(set([e[0] for e in semsimian.get_spo()]))

    # make ancestor pd df
    ancestors_pd = make_ancestors_list(semsimian.get_spo())

    profiles_pd = initialize_profiles(
        all_hpo_terms=all_hpo_terms,
        n_profiles=hyper_n_profile_pop_size,
        fraction_negated_terms=hyper_n_fraction_negated_terms,
        hpo_terms_per_profile=hyper_n_initial_hpo_terms_per_profile,
    )

    fitness_by_iteration = []

    progress_bar = tqdm(total=hyper_n_iterations, desc="Running genetic algorithm")

    for i in list(range(hyper_n_iterations)):
        # run termset similarity for each profile vs each patient in train split

        # Apply dropout to pt_train_df
        # Separate the dataframe into two based on `patient_labels`
        pt_train_df_0 = pt_train_df[pt_train_df['patient_label'] == 0]
        pt_train_df_1 = pt_train_df[pt_train_df['patient_label'] == 1]
        # Perform stratified sampling and concatenate the sampled dataframes
        pt_train_df_sampled = pd.concat(
            [pt_train_df_0.sample(frac=1 - hyper_pt_dropout_fraction),
             pt_train_df_1.sample(frac=1 - hyper_pt_dropout_fraction)])

        sim_results = compare_profiles_to_patients(
            semsimian=semsimian,
            pt_train_df=pt_train_df_sampled,
            profiles_pd=profiles_pd,
            debug=debug,
        )
        train_auc_results = make_auc_df(
            sim_results=sim_results,
            patient_labels=pt_train_df[
                ["person_id", "patient_label"]
            ].drop_duplicates(),
        )

        # add results to fitness_by_iteration
        new_row = train_auc_results.copy()
        new_row["iteration"] = i
        fitness_by_iteration += [new_row]

        top_n_profiles = train_auc_results.sort_values(
            by=hyper_fitness_auc, ascending=False
        ).head(hyper_n_best_profiles)["profile_id"]

        profiles_pd = profiles_pd[profiles_pd["profile_id"].isin(top_n_profiles)]

        # output info for top profile
        pd.set_option("display.max_columns", None)  # pandas smdh
        best = profiles_pd[profiles_pd["profile_id"].isin(top_n_profiles[:1])]
        # sort by weight
        best = best.sort_values(by="weight", ascending=False, inplace=False)
        if node_labels is not None:
            best = best.merge(node_labels, on="hpo_term_id")
            print(best[["hpo_term_id", "weight", "negated", "name"]])
        else:
            print(best[["hpo_term_id", "weight", "negated"]])

        profiles_pd = recombine_profiles_pd(
            profiles=profiles_pd,
            ancestors_df=ancestors_pd,
            num_profiles=hyper_n_profile_pop_size,
        )

        progress_bar.set_description(
            "Running genetic algorithm - {} for iteration {}: {}".format(
                hyper_fitness_auc,
                i + 1,
                round(train_auc_results[hyper_fitness_auc].mean(), 2),
            )
        )

        profiles_pd = add_terms_to_profiles_pd(
            profiles=profiles_pd,
            all_hpo_terms=all_hpo_terms,
            ancestor_list=ancestors_pd,
            add_term_p=hyper_add_term_p,
            fraction_negated_terms=hyper_n_fraction_negated_terms,
        )

        profiles_pd = remove_terms_from_profiles_pd(
            profiles=profiles_pd, remove_term_p=hyper_remove_term_p
        )

        profiles_pd = change_weights_for_profiles(
            profiles=profiles_pd, change_weight_p=hyper_change_weight_p
        )

        profiles_pd = move_terms_on_hierarchy(
            profiles=profiles_pd,
            move_term_p=hyper_move_term_on_hierarchy_p,
            hpo_graph=hpo_graph,
            # to make sure we don't move terms to
            # terms that aren't an s in the spo
            # to keep semsimian happy
            include_list=[t[0] for t in semsimian.get_spo()],
            debug=debug,
        )

        # increment progress bar
        progress_bar.update(1)

    progress_bar.close()

    test_results = compare_profiles_to_patients(
        semsimian=semsimian,
        pt_train_df=pt_test_df,
        profiles_pd=profiles_pd,
        debug=debug,
    )
    test_auc_results = make_auc_df(
        sim_results=test_results,
        patient_labels=pt_test_df[["person_id", "patient_label"]].drop_duplicates(),
    )

    # add results to fitness_by_iteration
    new_row = test_auc_results.copy()
    new_row["iteration"] = -999
    fitness_by_iteration += [new_row]

    # output average AUC in fitness_by_iteration by iteration
    average_auc_by_iteration = pd.DataFrame(
        columns=["iteration", "mean_auroc", "mean_auprc"]
    )
    for i in range(len(fitness_by_iteration)):
        average_auc_by_iteration.loc[i] = [
            i,
            fitness_by_iteration[i]["auroc"].mean(),
            fitness_by_iteration[i]["auprc"].mean(),
        ]

    # output profiles_pd to a file
    if node_labels is not None:
        profiles_pd = profiles_pd.merge(node_labels, on="hpo_term_id", how="left")

    # join in AUC results for each profile
    profiles_pd.reset_index(drop=True, inplace=True)
    test_auc_results.reset_index(drop=True, inplace=True)
    profiles_pd = profiles_pd.merge(test_auc_results, on="profile_id", how="left")
    profiles_pd.drop(["description"], axis=1, inplace=True)

    # sort profiles by AUC
    profiles_pd.sort_values(
        by=[hyper_fitness_auc, "profile_id", "weight"], ascending=False, inplace=True
    )

    # make outfile with all hyperparameters in its name
    outfile = "ga_results_{}_{}_iterations".format(disease, hyper_n_iterations)
    profiles_pd.to_csv(outfile + ".tsv", index=False, sep="\t")
