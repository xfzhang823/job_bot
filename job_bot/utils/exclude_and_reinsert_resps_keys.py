"""
File: exclude_and_reinsert_resp_keys.py

Functions to:
* exclude certain keys (like "promoted to ... in ....) from optimization
* re-insert them afterwards to be included in the final resume 
"""


def exclude_keys(json_data: dict) -> dict:
    """
    Creates a new dictionary excluding specified keys from the original.

    Args:
        json_data (dict): The input dictionary.

    Returns:
        dict: A new dictionary with the specified keys excluded.

    Notes:
        This function does not modify the original dictionary.
    """
    keys_to_exclude = ["3.responsibilities.5"]

    if not isinstance(json_data, dict):
        raise ValueError("Input must be a dictionary")

    return {k: v for k, v in json_data.items() if k not in keys_to_exclude}


def reinsert_keys(json_data: dict) -> dict:
    """
    Creates a new dictionary by adding back specified keys to JSON data.

    Args:
        json_data (dict): The input dictionary.

    Returns:
        dict: A new dictionary with the specified keys added.

    Notes:
        This function does not modify the original dictionary.
    """
    if not isinstance(json_data, dict):
        raise ValueError("Input must be a dictionary")

    keys_to_add = ["3.responsibilities.5"]
    values_to_add = [
        "Promoted from Senior Research Analyst to Research Manager in February 2007."
    ]

    # Create a copy to avoid modifying the original dictionary
    new_json_data = json_data.copy()

    # Use zip to pari keys and values
    for key, value in zip(keys_to_add, values_to_add):
        new_json_data[key] = value

    return new_json_data
