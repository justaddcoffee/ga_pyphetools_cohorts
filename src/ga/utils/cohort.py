import warnings

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def make_kfold_stratified_test_train_splits(
    pt_df,
    pt_id_col="person_id",
    pt_label_col="patient_label",
    num_splits=5,
    shuffle=True,
    seed=42,
) -> list[dict]:
    # given a pandas df with these columns
    # 'person_id', 'hpo_term_id', 'hpo_term_label', 'excluded', 'patient_label'

    # return a list of num_splits dicts with keys 'train' and 'test' containing
    # pandas dfs with the train/test split for each fold

    # make test/train split
    skf = StratifiedKFold(n_splits=num_splits, shuffle=shuffle, random_state=seed)

    pt_id_df = pt_df[[pt_id_col, pt_label_col]].drop_duplicates()

    train_test_kfolds = []
    for i, (train_index, test_index) in enumerate(
        skf.split(pt_id_df, pt_id_df[pt_label_col])
    ):
        # make a pandas df with train, another with test
        train = (
            pt_id_df.drop('patient_label', axis=1).iloc[train_index].merge(pt_df,
                                                                           on=pt_id_col,
                                                                           how="left"))
        test = (
            pt_id_df.drop('patient_label', axis=1).iloc[test_index].merge(pt_df,
                                                                          on=pt_id_col,
                                                                          how="left"))
        train_test_kfolds.append({'train': train, 'test': test})

    return train_test_kfolds


def make_test_train_split(
    pt_df,
    pt_id_col="person_id",
    pt_label_col="patient_label",
    shuffle=True,
    seed=42,
) -> dict:
    # make a single test/train split
    pt_id_df = pt_df[[pt_id_col, pt_label_col]].drop_duplicates()

    train_pts, test_pts = train_test_split(
        pt_id_df, stratify=pt_id_df[pt_label_col], shuffle=shuffle, random_state=seed
    )

    # merge back in the phenotypes
    train = train_pts.drop(pt_label_col, axis=1).merge(pt_df, on=pt_id_col, how="left")
    test = test_pts.drop(pt_label_col, axis=1).merge(pt_df, on=pt_id_col, how="left")

    return {"train": train, "test": test}


def make_cohort_df(
    data,
    disease,
    diseases_to_remove_from_negatives=[],
    drop_duplicates_phenotypes=True,
    expected_vals_for_excluded_col=None,
) -> pd.DataFrame:
    """Make a pandas dataframe with positive and negative cases for a disease
    """

    if disease not in data["by_disease"]:
        raise RuntimeError(f"{disease} is not in the data")

    if expected_vals_for_excluded_col is None:
        expected_vals_for_excluded_col = ["observed", "excluded"]
    negatives = list(data["by_disease"].keys())
    if disease in negatives:
        negatives.remove(disease)
    for r in diseases_to_remove_from_negatives:
        if r in negatives:
            negatives.remove(r)
        else:
            warnings.warn(
                f"{r} is not in the negatives list, so I can't remove it from the "
                f"negatives list"
            )

    negatives.sort()
    # make a pandas dataframe with all disease cases as positive and negative cases as
    # negative
    these_columns = [
        "person_id",
        "hpo_term_id",
        "hpo_term_label",
        "excluded",
        "patient_label",
    ]

    # data looks like
    # data['by_diseaese']['Marfan syndrome']['patient_id'] = [list of phenotypes]

    # phenotypes look like:
    # ('HP:0011968', 'Feeding difficulties', 'observed')

    # columns:
    # person_id, hpo_term_id, hpo_term_label, excluded, patient_label

    # make positive cases
    pos_cases = []
    seen_pt_ids = set()
    for pt in data['by_disease'][disease]:
        this_pt_id = pt['subject']['id']
        # check for duplicate patient ids
        if this_pt_id in seen_pt_ids:
            warnings.warn(f"Duplicate patient id: {this_pt_id}")
        seen_pt_ids.add(this_pt_id)

        for phenotype in pt['parsedPhenotypicFeatures']:
            pos_cases.append([this_pt_id, phenotype[0], phenotype[1], phenotype[2], 1])
    pos_df = pd.DataFrame(pos_cases, columns=these_columns)

    # make negative cases
    neg_cases = []
    for disease in negatives:
        for pt in data['by_disease'][disease]:
            this_pt_id = pt['subject']['id']

            # check for duplicate patient ids
            if this_pt_id in seen_pt_ids:
                warnings.warn(f"Duplicate patient id: {this_pt_id}")
            seen_pt_ids.add(this_pt_id)

            for phenotype in pt['parsedPhenotypicFeatures']:
                neg_cases.append([this_pt_id, phenotype[0], phenotype[1], phenotype[2], 0])
    neg_df = pd.DataFrame(neg_cases, columns=these_columns)

    pt_df = pd.concat([pos_df, neg_df], ignore_index=True)

    if ("excluded" not in pt_df.columns or
            not set(list(pt_df["excluded"].unique()))
                    .issubset(set(expected_vals_for_excluded_col))):
        raise RuntimeError(
            "Didn't get the expected values for the 'excluded' column: {}".format(
                str(list(pt_df["excluded"].unique()))
            )
        )

    # convert observed/excluded to boolean
    pt_df["excluded"] = pt_df["excluded"].apply(
        lambda x: True if x == "excluded" else False
    )

    if drop_duplicates_phenotypes:
        pt_df = pt_df.drop_duplicates(
            subset=["person_id", "hpo_term_id", "excluded", "patient_label"]
        )

    pt_df.rename(columns={"excluded": "negated"}, inplace=True)
    # all pt phenotypes are weighted equally
    pt_df["weight"] = 1.0

    return pt_df
