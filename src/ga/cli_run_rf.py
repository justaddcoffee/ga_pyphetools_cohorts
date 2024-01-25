import warnings
from pathlib import Path

import click
from semsimian import Semsimian

from ga.utils.cohort import make_cohort, make_kfold_stratified_test_train_splits, \
    make_test_train_split
from ga.utils.hpo import make_hpo_closures_and_graph, make_hpo_labels_df
from ga.utils.phenopacket import parse_phenopackets
from ga.utils.rf import run_rf_algorithm
from ga.utils.utils import run_genetic_algorithm
# from ga.utils.utils_rf import run_rf_algorithm


@click.command("run_rf")
@click.option(
    "--phenopacket-dir",
    "-p",
    required=True,
    help="Path to directory containing phenopackets",
    type=Path,
)
#@click.option(
#    "--hpo-url",
#    "-h",
#    required=False,
#    help="URL to HPO ontology in KGX TSV format",
#    type=str,
#    default="https://kg-hub.berkeleybop.io/kg-obo/hp/2023-04-05/hp_kgx_tsv.tar.gz",
#    show_default=True,
#)
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
def run_rf_command(
    phenopacket_dir: Path,
    #hpo_url: str,
    disease: str,
    diseases_to_remove_from_negatives: list[str],
    #hpo_root_node_to_use: str = "HP:0000001",
    #remove_pt_terms_not_in_spo: bool = True,
    debug: bool = False,
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

    # test/train split
    pt_test_train_dict = make_kfold_stratified_test_train_splits(
        pt_df=pt_df, num_splits=2, seed=42
    )
    s = Semsimian(spo=spo)

    #run_genetic_algorithm(
    #    semsimian=s,
    #   disease=disease,
    #    pt_train_df=pt_test_train_dict[0]["train"],
    #    pt_test_df=pt_test_train_dict[0]["test"],
    #    hpo_graph=hpo_graph,
    #    node_labels=node_labels,
    #    hyper_initialize_and_add_terms_only_from_observed_terms=True,
    #    debug=debug,
    #)

    run_rf_algorithm(
        #semsimian=s,
        disease=disease,
        pt_train_df=pt_test_train_dict[0]["train"],
        pt_test_df=pt_test_train_dict[0]["test"],
        #hpo_graph=hpo_graph,
        #node_labels=node_labels,
        #hyper_initialize_and_add_terms_only_from_observed_terms=True,
        #debug=debug,
    )

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
