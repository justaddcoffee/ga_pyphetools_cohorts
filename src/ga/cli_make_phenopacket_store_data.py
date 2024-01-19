import click
from pathlib import Path
import subprocess


@click.option(
    "--phenopacket-store-url",
    "-p",
    required=False,
    default='https://github.com/monarch-initiative/phenopacket-store.git',
    help="URL for phenopacket store github repo. Defaults to https://github.com/monarch-initiative/phenopacket-store.git",
    type=str
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    help="Path to output directory.",
    type=Path,
    default="data/phenopacket_store",
    show_default=True,
)
@click.command("make-phenopacket-store-data")
def make_phenopacket_store_data_command(
        output_dir: Path,
        phenopacket_store_url: str,
):
    """Clone phenopacket store, which contains lots of phenopackets
    """
    # make output dir if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    # Clone the repository
    subprocess.run(["git", "clone", "--depth", "1", phenopacket_store_url, output_dir])



