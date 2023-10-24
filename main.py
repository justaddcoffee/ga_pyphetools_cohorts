import os
import json
from collections import defaultdict
from ga import run_genetic_algorithm

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get data
    if not os.path.exists('phenopacket-store'):
        os.system("git clone https://github.com/monarch-initiative/phenopacket-store.git")

    phenopackets_path = 'phenopacket-store/phenopackets/'
    data = parse_phenopackets(phenopackets_path)
    parse_phenopackets(phenopackets_path)


