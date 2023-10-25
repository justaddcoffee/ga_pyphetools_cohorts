import os
from ga.ga import parse_phenopackets


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get data (can't add as dependency)
    if not os.path.exists('phenopacket-store'):
        os.system("git clone https://github.com/monarch-initiative/phenopacket-store.git")

    phenopackets_path = 'phenopacket-store/phenopackets/'
    data = parse_phenopackets(phenopackets_path)
    pass


