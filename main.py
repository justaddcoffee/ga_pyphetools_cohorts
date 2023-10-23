import os
import json
from collections import defaultdict


def parse_phenopackets(directory_path):
    json_files = []
    for foldername, subfolders, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.json'):
                json_files.append(os.path.join(foldername, filename))
    parsed_data = defaultdict(list)

    for json_file in json_files:
        # Open and parse the JSON file
        with (open(json_file, 'r') as file):
            try:
                # Parse JSON data into a dictionary
                data = json.load(file)

                # # # print out disease
                # if not 'interpretations' in data or len(data['interpretations']) == 0:
                #     raise Exception(f"Error parsing {json_file}: no disease found")

                if 'diseases' in data:
                    for d in data['diseases']:
                        if d['excluded'] != False:
                            parsed_data[i['diagnosis']['disease']['id']].append(data)

                if 'interpretations' in data:
                    for i in data['interpretations']:
                      parsed_data[i['diagnosis']['disease']['id']].append(data)

            except Exception as e:
                print(f"Error parsing {json_file}: {e}")
    return parsed_data


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get data
    if not os.path.exists('phenopacket-store'):
        os.system("git clone https://github.com/monarch-initiative/phenopacket-store.git")

    phenopackets_path = 'phenopacket-store/phenopackets/'
    data = parse_phenopackets(phenopackets_path)
    parse_phenopackets(phenopackets_path)


