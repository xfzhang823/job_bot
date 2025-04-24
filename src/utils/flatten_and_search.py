"""TBA"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

# root_dir = Path(__file__).resolve().parent.parent / "src"
# sys.path.append(str(root_dir))

logger = logging.getLogger(__name__)


# Function to flatten a dict
def flatten_dict(d, parent_key="", sep="."):
    """
    Flattens a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The prefix for keys in the flattened dictionary.
        sep (str): Separator used between keys.

    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.extend(flatten_list(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Funct to flatten a list
def flatten_list(lst, parent_key: Optional[str] = "", sep: str = "."):
    """
    Flattens a nested list.

    Args:
        lst (list): The list to flatten.
        parent_key (str): The prefix for keys in the flattened dictionary.
        sep (str): Separator used between keys.

    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for i, item in enumerate(lst):
        new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
        if isinstance(item, dict):
            items.extend(flatten_dict(item, new_key, sep=sep).items())
        elif isinstance(item, list):
            items.extend(flatten_list(item, new_key, sep=sep).items())
        else:
            items.append((new_key, item))
    return dict(items)


def flatten_dict_and_list(
    obj: Union[Dict, List, Any], parent_key: str = "", sep: str = "."
):
    """
    Flattens a nested structure that could be a dictionary or a list.

    Args:
        - obj (dict | list | any): The object to flatten. Can be a dictionary,
        list, or single value.
        - parent_key (str): The prefix for keys in the flattened dictionary
        (used for recursion).
        - sep (str): Separator used between nested keys (default is ".").

    Returns:
        dict: A flattened dictionary where:
            - Nested dictionary keys are concatenated with `sep`.
            - List elements are indexed numerically.

    Note:
        - For dictionaries, keys are concatenated with a separator (e.g., `.`).
        - For lists, indices are used as keys.
        - If `obj` is a primitive value, it will be wrapped in a dictionary.

    >>> Example:
        Given a nested structure like:
        ```json
        {
            "pie_in_the_sky": [
                "Masters’ degree or equivalent in business",
                "PhD in Computer Science"
            ],
            "down_to_earth": [
                "5+ years of experience in software development",
                "Strong knowledge of Python"
            ]
        }
        ```
        Calling `flatten_dict_and_list(obj)` will return:
        ```json
        {
            "0.pie_in_the_sky.0": "Masters’ degree or equivalent in business",
            "0.pie_in_the_sky.1": "PhD in Computer Science",
            "1.down_to_earth.0": "5+ years of experience in software development",
            "1.down_to_earth.1": "Strong knowledge of Python"
        }

    How Key Generation Works:
    - Dictionary keys are preserved but concatenated with `sep` if nested.
    - List elements are indexed numerically, ensuring uniqueness.
    - "0.pie_in_the_sky.0" represents:
        - "pie_in_the_sky" as the original key.
        - The first (`0`) element in its list.
    - "1.down_to_earth.0" represents:
        - `"down_to_earth" as the original key.
        - The first (`0`) element in its list.

    Notes:
        - Ensures every value is uniquely identified.
        - Helps structure complex, nested data for easy access and modification.
    """
    if isinstance(obj, dict):
        return flatten_dict(obj, parent_key, sep)
    elif isinstance(obj, list):
        return flatten_list(obj, parent_key, sep)
    else:
        # Ensure a dictionary return type
        return {
            parent_key if parent_key else "root": obj
        }  # "root" key ensures dict output


def recursive_search(d, search_key=None, search_value=None, results=None):
    """
    Recursively searches for a specific key or value in a nested dictionary.

    Args:
        d (dict): The nested dictionary to search.
        search_key (str, optional): The key to search for.
        search_value (Any, optional): The value to search for.
        branches (list, optional): The list to store found branches.

    Returns:
        None: Modifies the branches list in place. We use "implicit return"
        to be more memory-efficient
    """
    if results is None:
        results = []

    if isinstance(d, dict):
        # If searching by key, append only the key-value pair found
        if search_key in d:
            results.append({search_key: d[search_key]})

        # If searching by value, append only the key-value pair with the matched value
        elif search_value in d.values():
            for k, v in d.items():
                if v == search_value:
                    results.append({k: v})

        # Recursively search in nested dictionaries
        for k, v in d.items():
            if isinstance(v, dict):
                recursive_search(v, search_key, search_value, results)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        recursive_search(item, search_key, search_value, results)


def fetch_subtrees(d: Dict, search_key: str = "", search_value: str = ""):
    """
    Fetches only the immediate items containing a specified key or value in a nested
    dictionary.

    Args:
        d (dict): The nested dictionary to search.
        search_key (str, optional): The key to search for.
        search_value (Any, optional): The value to search for.

    Returns:
        list: A list of immediate items (key-value pairs) containing the search key
        or value.

        Example:
        Given the nested dictionary:
        {
            "A": {
                "B": {
                    "D": {},
                    "E": {}
                },
                "C": {
                    "F": {}
                }
            }
        }

        Searching for key "B" will return:
        [
            {
                "B": {
                    "D": {},
                    "E": {}
                }
            }
        ]

        This returns the entire subtree starting at "B" with all its descendants.
    """
    results = []
    recursive_search(d, search_key, search_value, results)
    return results


def fetch_subtrees_under_subtrees(d, parent_key, search_key):
    """
    Fetches subtrees under a specific parent subtree containing a specified key.

    Args:
        d (dict): The nested dictionary to search.
        parent_key (str): The key of the parent subtree to search under.
        search_key (str): The key to search for within the parent subtree.

    Returns:
        list: A list of subtrees under the specified parent containing the search key.
    """
    # Step 1: Fetch the subtree under the parent key
    parent_subtrees = fetch_subtrees(d, search_key=parent_key)

    # If no parent subtree is found, return an empty list
    if not parent_subtrees:
        return []

    # Step 2: Search for the key under each found parent subtree
    subtrees_under_parent = []
    for parent_subtree in parent_subtrees:
        # Search for the key under the parent subtree
        subtrees_under_parent.extend(
            fetch_subtrees(parent_subtree, search_key=search_key)
        )

    return subtrees_under_parent


def fetch_branches(d, search_key=None, search_value=None):
    """
    Fetches entire branches (sub-dictionaries) containing a specified key or value.

    Args:
        d (dict): The nested dictionary to search.
        search_key (str, optional): The key to search for.
        search_value (Any, optional): The value to search for.

    Returns:
        list: A list of branches (sub-dictionaries) containing the search key or value.

     Example:
        Given the nested dictionary:
        {
            "A": {
                "B": {
                    "D": {},
                    "E": {}
                },
                "C": {
                    "F": {}
                }
            }
        }

        Searching for key "E" will return:
        [
            ["A", "B", "E"]
        ]

        This represents the branch or path from the root "A" down to the node "E".
    """
    # Initialize results list to store matched branches
    results = []

    # Use recursive_search to fetch only immediate matching items
    recursive_search(d, search_key, search_value, results)

    # Initialize branches list to store full branches containing the matched items
    branches = []

    # Iterate through the results to find and add entire branches
    for result in results:
        # Search the parent dictionary to find the full branch containing the result
        for k, v in d.items():
            if isinstance(v, dict) and (result.keys() & v.keys()):
                branches.append(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and (result.keys() & item.keys()):
                        branches.append(item)

    return branches


# testing
def main():
    # Example Usage
    print()


if __name__ == "__main__":
    main()
