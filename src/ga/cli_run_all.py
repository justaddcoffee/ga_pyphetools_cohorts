
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import click

from semsimian import Semsimian
from ga.utils.cohort import make_cohort, make_kfold_stratified_test_train_splits
from ga.utils.hpo import make_hpo_closures_and_graph, make_hpo_labels_df
from ga.utils.phenopacket import parse_phenopackets
from ga.utils.vertical_to_horizontal import vertical_to_horizontal
from ga.utils.utils_dt import run_dt_algorithm
from ga.utils.utils_lr import run_lr_algorithm
from ga.utils.utils_rf import run_rf_algorithm
from ga.utils.utils import run_genetic_algorithm


@click.command("run_all")
@click.option(
    "--phenopacket-dir",
    "-p",
    required=True,
    help="Path to directory containing phenopackets",
    type=Path,
)
# @click.option(
#    "--hpo-url",
#    "-h",
#    required=False,
#    help="URL to HPO ontology in KGX TSV format",
#    type=str,
#    default="https://kg-hub.berkeleybop.io/kg-obo/hp/2023-04-05/hp_kgx_tsv.tar.gz",
#    show_default=True,
# )
@click.option("--disease", "-d", required=True, help="Disease to analyze", type=str)
@click.option(
    "--diseases-to-remove-from-negatives",
    "-r",
    required=False,
    help="Diseases to remove from negative phenotypes",
    type=str,
    multiple=True,
    default=[],
    show_default=True,
)
@click.option(
    "--hpo-root-node-to-use",
    "-r",
    required=False,
    help="HPO root node to use",
    type=str,
    default="HP:0000001",
    show_default=True,
)
@click.option(
    "--remove-pt-terms-not-in-spo",
    "-r",
    required=False,
    help="Remove patient HPO terms that are not in the closures we're using",
    type=bool,
    default=True,
    show_default=True,
)
@click.option(
    "--debug",
    "-b",
    required=False,
    help="Debug mode",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--hpo-url",
    "-h",
    required=False,
    help="URL to HPO ontology in KGX TSV format",
    type=str,
    default="https://kg-hub.berkeleybop.io/kg-obo/hp/2023-04-05/hp_kgx_tsv.tar.gz",
    show_default=True,
)

def run_all_command(
        phenopacket_dir: Path,
        hpo_url: str,
        disease: str,
        diseases_to_remove_from_negatives: list[str],
        hpo_root_node_to_use: str = "HP:0000001",
        remove_pt_terms_not_in_spo: bool = True,
        pt_id_col='person_id',
        pt_label_col='patient_label',
        min_num_pts=3,
        max_depth=None,
        n_estimators=100,
        num_splits=2,
        num_rep=1,
        ndigits=2,
        debug=False
):
    data = parse_phenopackets(phenopacket_dir)

    # make a cohort to analyze

    # make cohort
    negatives = list(data["phenotype_data"].keys())
    if disease in negatives:
        negatives.remove(disease)
    for r in diseases_to_remove_from_negatives:
        if r in negatives:
            negatives.remove(r)
        else:
            warnings.warn(
                f"{r} is not in the negatives list, so I can't remove it from the negatives list"
            )
    negatives.sort()
    pt_df = make_cohort(data["phenotype_data"], disease, negatives)
    pt_df.rename(columns={"excluded": "negated"}, inplace=True)
    # all pt phenotypes are weighted equally
    pt_df["weight"] = 1.0

    # make "spo" (subject predicate object closures) for semsimian and also nx graph
    # assign spo to first element of tuple, graph to second
    spo, hpo_graph = make_hpo_closures_and_graph(
        url=hpo_url,
        root_node_to_use=hpo_root_node_to_use,
        include_self_in_closure=True,
    )

    node_labels = make_hpo_labels_df(url=hpo_url)

    # check how many pt HPO terms we have that aren't in the spo
    all_pt_hpo_terms = set(pt_df["hpo_term_id"].unique())
    all_spo_hpo_terms = set([s[0] for s in spo] + [s[2] for s in spo])

    # find all_pt_hpo_terms that aren't in all_spo_hpo_terms
    pt_hpo_terms_not_in_spo = all_pt_hpo_terms.difference(all_spo_hpo_terms)
    warnings.warn(
        f"There are {str(len(pt_hpo_terms_not_in_spo))} "
        f"({str(round(100 * len(pt_hpo_terms_not_in_spo) / len(all_pt_hpo_terms), 2))}%) patient HPO terms are not in the closures we're using using: "
        f"{' '.join(pt_hpo_terms_not_in_spo)}\n"
        f"These are possibly obsolete terms, or terms that are not in the induced subgraph of the `root_node_to_use` arg passed to make_hpo_closures(). "
        f"These terms will have 0 semantic similarity to other terms, and may cause a semsimian panic"
    )
    # get rid of these terms
    if remove_pt_terms_not_in_spo:
        warnings.warn(
            f"Removing {str(len(pt_hpo_terms_not_in_spo))} patient HPO terms that are not in the closures I'm using"
        )
        pt_df = pt_df[~pt_df["hpo_term_id"].isin(pt_hpo_terms_not_in_spo)]

    check_df = pt_df[[pt_id_col,pt_label_col]].drop_duplicates()
    if sum(check_df[pt_label_col]) == 0:
        warnings.warn(
            f"I have no positive patients for {disease}: num neg = {str(sum(check_df[pt_label_col]==0))}, num pos = {str(sum(check_df[pt_label_col]))} - interrupting execution"
        )
        return

    if sum(check_df[pt_label_col] == 0 ) == 0:
        warnings.warn(
            f"I have no negative patients for {disease}: num neg = {str(sum(check_df[pt_label_col]==0))}, num pos = {str(sum(check_df[pt_label_col]))} - interrupting execution"
        )
        return

    all_rep_perfs_RF = []
    all_rep_perfs_DT = []
    all_rep_perfs_LR = []
    for nn in range(num_rep):

        # test/train split
        pt_test_train_dict = make_kfold_stratified_test_train_splits(
            pt_df=pt_df, num_splits=num_splits, seed=42, pt_id_col=pt_id_col, pt_label_col=pt_label_col
        )

        # print(sum(df_train['patient_label']), sum(y_test), sum(y_train == 0), sum(y_test == 0))

        # s = Semsimian(spo=spo)
        # run_genetic_algorithm(
        #    semsimian=s,
        #   disease=disease,
        #    pt_train_df=pt_test_train_dict[0]["train"],
        #    pt_test_df=pt_test_train_dict[0]["test"],
        #    hpo_graph=hpo_graph,
        #    node_labels=node_labels,
        #    hyper_initialize_and_add_terms_only_from_observed_terms=True,
        #    debug=debug,
        # )
        all_split_res_LR = []
        all_split_res_DT = []
        all_split_res_RF = []
        for nsplit in range(num_splits):

            # check that there are enough train/test pos/neg. Enough means that all the train/test neg/pos sets must contain at least min_num_pts
            train_labels = pt_test_train_dict[nsplit]["train"][[pt_id_col, pt_label_col]].drop_duplicates()
            test_labels = pt_test_train_dict[nsplit]["test"][[pt_id_col, pt_label_col]].drop_duplicates()


            if (sum(train_labels[pt_label_col]) < min_num_pts):
                print("too few positive training cases (", sum(train_labels[pt_label_col]), "<", min_num_pts, ')')
                return

            if (sum(train_labels[pt_label_col] == 0) < min_num_pts):
                print("too few negative training cases (", sum(train_labels[pt_label_col] == 0), "<", min_num_pts,
                      ')')
                return

            if (sum(test_labels[pt_label_col]) < min_num_pts):
                print("too few positive test cases (", sum(test_labels[pt_label_col]), "<", min_num_pts, ')')
                return

            if (sum(test_labels[pt_label_col] == 0) < min_num_pts):
                print("too few negative test cases (", sum(test_labels[pt_label_col] == 0), "<", min_num_pts, ')')
                return

            pt_train_df = pt_test_train_dict[nsplit]["train"]
            # convert training and test sets to horizontal dfs but minding patient ids
            #pt_train_df["new_col"] = "train"
            #new = pt_train_df["new_col"].copy()
            #pt_train_df[id_col] = pt_train_df[id_col].str.cat(new, sep="_")
            #pt_train_df = pt_train_df.drop(columns=['new_col'])

            pt_test_df = pt_test_train_dict[nsplit]["test"]
            #pt_test_df["new_col"] = "test"
            #new = pt_test_df["new_col"].copy()
            #pt_test_df[id_col] = pt_test_df[id_col].str.cat(new, sep="_")
            #pt_test_df = pt_test_df.drop(columns=['new_col'])

            # just check how many train/test positives/negatives you have befor vertical to horizontal conversion
            ids_train = pt_train_df.drop(
                columns=['hpo_term_label', 'negated', 'weight', 'hpo_term_id']).drop_duplicates()
            ids_test = pt_test_df.drop(columns=['hpo_term_label', 'negated', 'weight', 'hpo_term_id']).drop_duplicates()

            print(sum(ids_train[pt_label_col]), sum(ids_test[pt_label_col]), sum(ids_train[pt_label_col] == 0),
                  sum(ids_test[pt_label_col] == 0))

            pt_df_h = vertical_to_horizontal(pd.concat([pt_train_df, pt_test_df]))
            df_train = pt_df_h.merge(ids_train, on=[pt_id_col, pt_label_col], how='inner')
            df_test = pt_df_h.merge(ids_test, on=[pt_id_col, pt_label_col], how='inner')

            print(sum(df_train[pt_label_col]), sum(df_test[pt_label_col]), sum(df_train[pt_label_col] == 0),
                  sum(df_test[pt_label_col] == 0))

            res_scores_RF = run_rf_algorithm(
                disease=disease,
                df_train=df_train,
                df_test=df_test,
                pt_label_col=pt_label_col,
                pt_id_col=pt_id_col,
                max_depth=max_depth,
                n_estimators=n_estimators
            )
            # append to list of split results
            all_split_res_RF.append(res_scores_RF)

            res_scores_LR = run_lr_algorithm(
                df_train=df_train,
                df_test=df_test,
                pt_label_col=pt_label_col,
                pt_id_col=pt_id_col,
                disease=disease
            )
            all_split_res_LR.append(res_scores_LR)

            res_scores_DT = run_dt_algorithm(
                df_train=df_train,
                df_test=df_test,
                pt_label_col=pt_label_col,
                pt_id_col=pt_id_col,
                disease=disease,
                max_depth=max_depth
            )
            all_split_res_DT.append(res_scores_DT)

        all_rep_perfs_RF.append([np.mean(tup) for tup in zip(*all_split_res_RF)])
        all_rep_perfs_DT.append([np.mean(tup) for tup in zip(*all_split_res_DT)])
        all_rep_perfs_LR.append([np.mean(tup) for tup in zip(*all_split_res_LR)])


    all_final_res = [tup for tup in zip(*all_rep_perfs_LR)]
    final_res_LR = [round(np.mean(tup), ndigits=ndigits) for tup in all_final_res]
    str_res = "\t".join([str(i) for i in final_res_LR])
    print(
        f'DISOKALL:\tLR\t{disease}\t{str_res}')


    all_final_res = [tup for tup in zip(*all_rep_perfs_DT)]
    final_res_DT = [round(np.mean(tup), ndigits=ndigits) for tup in all_final_res]
    str_res = "\t".join([str(i) for i in final_res_DT])
    print(
        f'DISOKALL:\tDT\t{disease}\t{str_res}')


    all_final_res = [tup for tup in zip(*all_rep_perfs_RF)]
    final_res_RF = [round(np.mean(tup), ndigits=ndigits) for tup in all_final_res]
    str_res = "\t".join([str(i) for i in final_res_RF])
    print(
        f'DISOKALL:\tRF\t{disease}\t{str_res}')


# do not need the following
def run_smoke_test(s):
    # this termset sim seems to be 0.0, which seems fishy
    pt_test_tuples = [
        ("HP:0002650", 1.0, False),
        ("HP:0000098", 1.0, False),
        ("HP:0001166", 1.0, False),
        ("HP:0001083", 1.0, False),
        ("HP:0000545", 1.0, False),
        ("HP:0002616", 1.0, False),
    ]
    profile_test_tuples = [
        ("HP:0033127", 0.7594267694796112, True),
        ("HP:0033677", 0.2590903171508303, False),
        ("HP:0010730", 0.7373312314046617, False),
        ("HP:0005206", 0.16651076083997507, False),
        ("HP:0033729", 0.30911732402073555, False),
    ]

    # version of these variables with only the term
    pt_test_terms = [
        "HP:0002650",
        "HP:0000098",
        "HP:0001166",
        "HP:0001083",
        "HP:0000545",
        "HP:0002616",
    ]
    profile_test_terms = [
        "HP:0033127",
        "HP:0033677",
        "HP:0010730",
        "HP:0005206",
        "HP:0033729",
    ]

    test_sim = s.termset_pairwise_similarity_weighted_negated(
        subject_dat=pt_test_tuples, object_dat=profile_test_tuples
    )
    return test_sim