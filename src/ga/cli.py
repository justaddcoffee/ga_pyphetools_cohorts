import click

from ga.cli_opt_hyper import optimize_hyperparameters_command
from ga.cli_run import run_ga_command


@click.group()
def main():
    pass


main.add_command(run_ga_command)
main.add_command(optimize_hyperparameters_command)
if __name__ == "__main__":
    main()


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
