# utils/field_mapping_utils.py

import pandas as pd
import re

# Define a dictionary for mapping file column names to expected code names
COLUMN_NAMES_TO_VARS_MAPPING = {
    "BERTScore Precision": "bert_score_precision",
    "Soft Similarity": "soft_similarity",
    "Word Mover's Distance": "word_movers_distance",
    "Deberta Entailment Scoree": "deberta_entailment_score",
    # Add any other mappings as needed
}


def rename_df_columns(df, column_mapping=None):
    """
    Rename DataFrame column names:
    - If a column_mapping is provided, it renames the columns according to the mapping.
    - If not, it just standardizes the column names by replacing spaces with underscores,
    removing apostrophes, and converting to lowercase.

    Args:
        df (pd.DataFrame): The DataFrame with original column names.
        column_mapping (dict, optional): A dictionary mapping original column names to new column names.

    Returns:
        pd.DataFrame: A DataFrame with renamed columns.
    """
    # If a column_mapping is provided, apply it without standardization
    if column_mapping is not None:
        df = df.rename(columns=column_mapping)
    else:
        # Define a regular expression pattern to remove unwanted characters
        pattern = re.compile(r"[^a-zA-Z0-9_]+")
        # Standardize column names in the DataFrame
        df.columns = df.columns.str.replace(" ", "_").str.replace("'", "").str.lower()
        df.columns = [pattern.sub("", col) for col in df.columns]
    return df
