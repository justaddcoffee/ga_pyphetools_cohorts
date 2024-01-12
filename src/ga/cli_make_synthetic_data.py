import shutil
import urllib
from pathlib import Path

import click
from phenotype2phenopacket.create.create import create_synthetic_patient_phenopacket
from phenotype2phenopacket.utils.utils import (
    filter_diseases,
    load_ontology,
    load_ontology_factory,
    return_phenotype_annotation_data,
)
from pheval.prepare.custom_exceptions import MutuallyExclusiveOptionError


@click.option(
    "--phenotype-annotation",
    "-p",
    required=False,
    help="Path for phenotype.hpoa. We will download from https://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa if this argument is not provided.",
    type=Path
)
@click.option(
    "--num-disease",
    "-n",
    required=False,
    help="Number of diseases to create synthetic patient phenopackets for.",
    type=int,
    default=0,
    cls=MutuallyExclusiveOptionError,
    mutually_exclusive=["omim_id_list"],
)
@click.option(
    "--omim-id-list",
    "-l",
    required=False,
    help="Path to .txt file containing OMIM IDs to create synthetic patient phenopackets,"
    "with each OMIM ID separated by a new line.",
    type=Path,
    default=None,
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    help="Path to output directory.",
    type=Path,
    default="phenopackets",
    show_default=True,
)
@click.command("make-synthetic-phenopackets")
def make_synthetic_data_command(
        num_disease: int,
        omim_id_list: Path,
        phenotype_annotation: Path,
        output_dir: Path = Path("data"),
        phenotype_annotation_url: str =
        "https://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa",
):
    """Create a set of synthetic patient phenopackets from a phenotype
    annotation file."""

    if phenotype_annotation is None:
        phenotype_annotation = get_phenotype_annotation_file(phenotype_annotation_url)

    output_dir.mkdir(exist_ok=True)

    phenotype_annotation_data = return_phenotype_annotation_data(phenotype_annotation)
    human_phenotype_ontology = load_ontology()
    ontology_factory = load_ontology_factory()

    omim_id = None  # not supporting this option for now
    grouped_omim_diseases = filter_diseases(
        num_disease, omim_id, omim_id_list, phenotype_annotation_data
    )
    for omim_disease in grouped_omim_diseases:
        create_synthetic_patient_phenopacket(
            human_phenotype_ontology,
            omim_disease,
            ontology_factory,
            output_dir,
            phenotype_annotation_data.version,
        )


def get_phenotype_annotation_file(phenotype_annotation_url: str,
                                  data_dir=Path("data"),
                                  filename="phenotype.hpoa"
                                  ) -> Path:
    data_dir.mkdir(exist_ok=True)
    phenotype_annotation = data_dir.joinpath(filename)
    if phenotype_annotation.exists():
        print(f"Using phenotype.hpoa from {phenotype_annotation}")
        return phenotype_annotation
    else:
        print(f"Downloading {phenotype_annotation_url} to {phenotype_annotation}")
        with urllib.request.urlopen(phenotype_annotation_url) as response, open(phenotype_annotation, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        return phenotype_annotation
