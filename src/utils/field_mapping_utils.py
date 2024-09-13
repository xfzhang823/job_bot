# utils/field_mapping_utils.py

import pandas as pd

# Define a dictionary for mapping file column names to expected code names
COLUMN_NAMES_TO_VARS_MAPPING = {
    "BERTScore Precision": "bert_score_precision",
    "Soft Similarity": "soft_similarity",
    "Word Mover's Distance": "word_movers_distance",
    "Deberta Entailment Scoree": "deberta_entailment_score",
    # Add any other mappings as needed
}


def translate_column_names(df, column_mapping=COLUMN_NAMES_TO_VARS_MAPPING):
    """
    Translate DataFrame column names using the provided mapping.

    Args:
        df (pd.DataFrame): The DataFrame with original column names.
        column_mapping (dict): A dictionary mapping original column names to new column names.

    Returns:
        pd.DataFrame: A DataFrame with translated column names.
    """
    # Use the pandas rename method to rename columns
    df = df.rename(columns=column_mapping)
    return df
