import click

from ga.cli_opt_hyper import optimize_hyperparameters_command
from ga.cli_make_synthetic_data import make_synthetic_data_command
from ga.cli_make_phenopacket_store_data import make_phenopacket_store_data_command
from ga.cli_run import run_ga_command
from ga.cli_run_all import run_all_command


@click.group()
def main():
    pass


main.add_command(run_ga_command)
main.add_command(run_all_command)
#main.add_command(run_rf_command)
#main.add_command(run_lr_command)
#main.add_command(run_dt_command)
main.add_command(optimize_hyperparameters_command)
main.add_command(make_synthetic_data_command)
main.add_command(make_phenopacket_store_data_command)
if __name__ == "__main__":
    main()
