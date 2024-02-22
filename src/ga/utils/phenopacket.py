import json
import os
from collections import defaultdict


def parse_phenopackets(directory_path) -> dict:
    json_files = []
    if not os.path.exists(directory_path):
        raise RuntimeError("Didn't find directory {}".format(directory_path))
    for foldername, subfolders, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith(".json"):
                json_files.append(os.path.join(foldername, filename))
    parsed_full_phenopacket_data = defaultdict(list)

    for json_file in json_files:
        # Open and parse the JSON file
        with open(json_file, "r") as file:
            try:
                data = json.load(file)

                data['parsedPhenotypicFeatures'] = extract_phenotypes(data)
                # get diseases for this phenopacket
                diseases = get_diseases(data)
                for d in diseases:
                    parsed_full_phenopacket_data[d].append(data)

            except Exception as e:
                print(f"Error parsing {json_file}: {e}")

    return {"by_disease": parsed_full_phenopacket_data}


def extract_phenotypes(data) -> list:
    extracted_phenotypes = []

    if "phenotypicFeatures" in data:
        for p in data["phenotypicFeatures"]:
            # add this phenotype to the set of phenotypes
            this_p = (
                p["type"]["id"],
                p["type"]["label"],
                "excluded" if "excluded" in p['type'] else "observed",
            )
            extracted_phenotypes.append(this_p)
    return extracted_phenotypes


def get_diseases(data) -> list:
    diseases = set()
    if "diseases" in data:
        for d in data["diseases"]:
            if not ("excluded" in d and d["excluded"] is True):
                # add this disease to the set of diseases
                diseases.add(d["term"]["label"])

    if "interpretations" in data:
        for i in data["interpretations"]:
            diseases.add(i["diagnosis"]["disease"]["label"])

    return list(diseases)
