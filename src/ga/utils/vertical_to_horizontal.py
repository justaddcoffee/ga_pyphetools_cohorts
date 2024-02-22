def vertical_to_horizontal(df):
    print(df.columns)
    df = df.drop_duplicates()
    # Group by "person_id" and "patient_label" and pivot on "hpo_term_id"
    df_pivoted = df.drop(columns=['hpo_term_label', 'negated', 'weight']).groupby(
        ["person_id", "patient_label", "hpo_term_id"]).size().unstack(fill_value=0).reset_index()

    print(df_pivoted.columns)

    return df_pivoted
