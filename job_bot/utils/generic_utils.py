"""Tool functions"""

import re
import os
import json
import shutil
import logging
from pathlib import Path
from typing import Union, Dict, List, Any, overload, Optional, Tuple
from pydantic import BaseModel
import pandas as pd
import demjson3

# From project modules
from job_bot.utils.file_name_utils import get_file_names

# Setup logger
logger = logging.getLogger(__name__)


def add_to_json_file(
    new_data: Union[list, dict], file_path: Union[Path, str], key: str = ""
):
    """
    Adds or updates data in a master JSON file.

    Args:
        new_data (dict or list): The data to be added or updated in the JSON file.
        file_path (str): The path to the master JSON file.
        key (str, optional):
            - If provided, adds or updates the data under this specific key.
            - If not provided, the new data is merged directly with the existing JSON data.

    Returns:
        None
    """
    try:
        # Load existing data from the file
        file_path = Path(file_path)  # Ensure file_path ends up as a Path obj.
        with open(file_path, "r", encoding="utf-8") as f:
            master_data = json.load(f)

        # Log the types of the existing and new data for better debugging
        logger.debug(
            f"Loaded data type: {type(master_data).__name__}, New data type: {type(new_data).__name__}"
        )

        # If a key is provided, ensure master_data is a dict and handle the update under the key
        if key:
            if not isinstance(master_data, dict):
                raise ValueError(
                    "Master JSON data must be a dictionary when using a key."
                )

            # Ensure the key exists and update accordingly
            if key in master_data:
                # If both are lists, extend the list
                if isinstance(master_data[key], list) and isinstance(new_data, list):
                    master_data[key].extend(new_data)
                # If both are dicts, update the dictionary
                elif isinstance(master_data[key], dict) and isinstance(new_data, dict):
                    master_data[key].update(new_data)
                else:
                    # If the data types for the existing key and new data don't match, raise an error
                    raise ValueError(
                        f"Incompatible data types for merging under key '{key}': "
                        f"{type(master_data[key]).__name__} vs {type(new_data).__name__}"
                    )
            else:
                # If the key does not exist, add it with the new data
                master_data[key] = new_data
        else:
            # If no key is provided, attempt to merge the data directly
            if isinstance(master_data, list) and isinstance(new_data, list):
                master_data.extend(new_data)
            elif isinstance(master_data, dict) and isinstance(new_data, dict):
                master_data.update(new_data)
            else:
                raise ValueError(
                    f"Mismatched data types: cannot merge {type(master_data).__name__} "
                    f"with {type(new_data).__name__}."
                )

        # Save the updated master data back to the file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(master_data, f, indent=4, ensure_ascii=False)

        logger.info(f"Data successfully added to {file_path}.")

    except FileNotFoundError:
        # If the file doesn't exist, create a new one with the new data
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)
        logger.info(f"File {file_path} created and data added.")

    except (KeyError, ValueError) as e:
        logger.error(f"Error adding data to {file_path}: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error adding data to {file_path}: {e}")
        raise


def clean_and_fix_json(json_file: Path):
    """
    Detects and fixes incorrectly nested JSON structures, formatting issues,
    and ensures the JSON is properly structured using demjson3 for auto-repair.

    Args:
        json_file (Path): Path to the JSON file to be fixed.

    Returns:
        None: Fixes the JSON in place and overwrites the file.

    Example Usage:
        clean_and_fix_json(Path("your file path here..."))
    """
    try:
        # Load the JSON file content as a string
        with open(json_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 🔹 Step 1: Use demjson3 to repair the JSON
        repaired_data = demjson3.decode(content)  # Auto-repairs broken JSON

        # Error handling
        if not isinstance(repaired_data, dict):
            raise ValueError(
                f"JSON repair failed! Expected dict but got {type(repaired_data)}"
            )
        # 🔹 Step 2: Detect and fix nested duplicate keys
        fixed_entries = 0
        for key, value in repaired_data.items():
            if isinstance(value, dict) and key in value:
                print(f"Incorrect nesting detected for key: {key}")
                repaired_data[key] = value[key]  # Flatten the structure
                fixed_entries += 1

        # 🔹 Step 3: Save the fully cleaned JSON back to the file
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(repaired_data, f, indent=4)  # Ensures proper formatting

        print(
            f"Fixed {fixed_entries} nested entries and cleaned JSON formatting in {json_file.name}!"
        )

    except Exception as e:
        print(f"Error fixing JSON in {json_file.name}: {e}")


# Example Usage:
# clean_and_fix_json(Path("C:/github/job_bot/input_output/preprocessing/jobpostings.json"))


def copy_files_btw_directories(src_folder: str, dest_folder: str, src_file: str = ""):
    """
    Copies a single file or all files from the source folder to the destination folder.

    Args:
        * src_folder (str): The path to the source folder where files are located.
        * dest_folder (str): The path to the destination folder where files will be copied.
        * src_file (str, optional):
            - The path to a specific file (can be a full path or just the file name).
            - If None, all files in the source folder will be copied.

    Raises:
        FileNotFoundError: If the source file or folder does not exist.
        Exception: For any general errors during file copying.

    Returns:
        None

    Example:
        # To copy a specific file (just the file name):
        copy_files('/path/to/source/folder', '/path/to/destination/folder', \
            'example.txt')

        # To copy a specific file (using the full path):
        copy_files('/path/to/source/folder', '/path/to/destination/folder', \
            '/path/to/source/folder/example.txt')

        # To copy all files from the source folder:
        copy_files('/path/to/source/folder', '/path/to/destination/folder')
    """
    # Ensure destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Copy a single file if src_file is provided
    if src_file:
        # Check if src_file is an absolute path, if not, combine it with src_folder
        if not os.path.isabs(src_file):
            full_src_file_path = os.path.join(src_folder, src_file)
        else:
            full_src_file_path = src_file

        # Ensure the file is actually present
        if os.path.isfile(full_src_file_path):
            dest_file_path = os.path.join(dest_folder, os.path.basename(src_file))
            try:
                shutil.copy2(full_src_file_path, dest_file_path)
                print(f"Copied: {full_src_file_path} to {dest_file_path}")
            except Exception as e:
                print(f"Error occurred while copying file {full_src_file_path}: {e}")
        else:
            print(f"File '{src_file}' not found at location: {full_src_file_path}")

    # Copy all files if src_file is None
    else:
        for file_name in os.listdir(src_folder):
            full_src_file_path = os.path.join(src_folder, file_name)
            dest_file_path = os.path.join(dest_folder, file_name)

            # Only copy if it's a file (not a subdirectory)
            if os.path.isfile(full_src_file_path):
                try:
                    shutil.copy2(full_src_file_path, dest_file_path)
                    print(f"Copied: {file_name} to {dest_folder}")
                except Exception as e:
                    print(
                        f"Error occurred while copying file {full_src_file_path}: {e}"
                    )


def compare_keys_in_json_files(
    file1: Union[Path, str], file2: Union[Path, str]
) -> Tuple[List[str], List[str]]:
    """
    Compare the keys of two JSON files and return missing keys in both directions.

    Args:
        file1 (Union[Path, str]): Path to the first JSON file.
        file2 (Union[Path, str]): Path to the second JSON file.

    Returns:
        Tuple[List[str], List[str]]:
        - A tuple where the first list contains keys missing in file2 but present in file1,
        - and the second list contains keys missing in file1 but present in file2.
    """
    # Convert str to Path if necessary
    file1 = Path(file1) if isinstance(file1, str) else file1
    file2 = Path(file2) if isinstance(file2, str) else file2

    # Load JSON data from both files
    with file1.open("r") as f1, file2.open("r") as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Get the keys from both files
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())

    # Find missing keys in both directions
    missing_in_file2 = keys1 - keys2  # Keys present in file1 but missing in file2
    missing_in_file1 = keys2 - keys1  # Keys present in file2 but missing in file1

    return list(missing_in_file2), list(missing_in_file1)


def check_if_json(file_path: str):
    """
    Checks if the file path exists and its extension is .json.

    Args:
        filepath (str): The file path to check.

    Returns:
        bool: True if the file exists and has a .json extension, False otherwise.
    """
    # Check if the file exists and has a .json extension
    file_path = str(file_path)
    if os.path.exists(file_path) and file_path.lower().endswith(".json"):
        logger.info(f"The file '{file_path}' already exists.")
        return True
    else:
        logger.info(
            f"The file '{file_path}' does not exist or does not have a .json extension."
        )
        return False


# Function to support save_to_json_file function
def convert_keys_to_str(obj: Any) -> Any:
    """
    Recursively convert all dictionary keys in the given object to strings.

    Args:
        obj (Any): The object to process.

    Returns:
        Any: The processed object with string keys.
    """
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            new_key = str(key)
            new_value = convert_keys_to_str(value)
            new_dict[new_key] = new_value
        return new_dict
    elif isinstance(obj, list):
        return [convert_keys_to_str(item) for item in obj]
    elif isinstance(obj, BaseModel):
        return convert_keys_to_str(obj.model_dump())
    else:
        return obj


# Function to support save to json file function
def convert_keys_and_paths_to_str(obj: Any) -> Any:
    """
    Recursively convert all dictionary keys to strings and Path objects to strings.

    Args:
        obj (Any): The object to process.

    Returns:
        Any: The processed object with string keys and string values where necessary.
    """
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            # Convert key to string if it's not already
            new_key = str(key) if not isinstance(key, str) else key
            # Recursively process the value
            new_value = convert_keys_and_paths_to_str(value)
            new_dict[new_key] = new_value
        return new_dict
    elif isinstance(obj, list):
        return [convert_keys_and_paths_to_str(item) for item in obj]
    elif isinstance(obj, Path):
        # Convert Path object to string
        return str(obj)
    elif isinstance(obj, BaseModel):
        # Convert Pydantic model to dict and process recursively
        return convert_keys_and_paths_to_str(obj.model_dump())
    else:
        return obj


# Utils tool to list to dict format
def ensure_dict_format(data, prefix="item"):
    """
    Ensure that the input data is in dictionary format.
    If it is a list, convert it to a dictionary with enumerated keys.

    Args:
        data (dict or list): Input data to ensure it is a dictionary.
        prefix (str): Prefix for keys if the data is a list.

    Returns:
        dict: Data converted to a dictionary format.
    """
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        # Convert list to dictionary with keys like 'item_0', 'item_1', etc.
        return {f"{prefix}_{i}": item for i, item in enumerate(data)}
    else:
        raise TypeError("Input data must be a list or dictionary.")


def fetch_new_urls(existing_url_list_file, url_list_file):
    """
    Fetch job posting URLs from the new jobs file, compare them with the existing
    jobs file, and return only the new URLs that have not been previously processed.

    Args:
        job_descriptions_json_file (str): File path of the JSON file containing
            already processed job postings. Typically from 'job_postings.json'.
        list_of_urls_file (str): File path of the JSON file containing the list
            of new job posting URLs.

    Returns:
        list: A list of URLs that are new and have not been processed yet.

    Notes:
        - Both files must be in valid JSON format.
        - The existing job descriptions file should use URLs as keys.
        - The new jobs file should have a structure where job postings are
        listed under "jobs" and each job has a "url" field.
    """

    # Load new job posting URLs from the file
    with open(url_list_file, "r") as f:
        new_job_data = json.load(f)

    # Extract URLs from the new job postings
    urls_to_check = [job["url"] for job in new_job_data["jobs"]]

    # Load the existing job descriptions from the existing JSON file
    with open(existing_url_list_file, "r") as f:
        existing_job_data = json.load(f)

    # Extract existing URLs (assuming URLs are the keys)
    existing_urls = list(existing_job_data.keys())

    # Filter out URLs that already exist
    new_urls = [url for url in urls_to_check if url not in existing_urls]

    if not new_urls:
        logger.info("No new URLs found.")  # Optional logging or handling

    return new_urls


def find_project_root(starting_path=None, marker=".git") -> Path | None:
    """
    Recursively find the root directory of the project by looking for a specific marker.

    Args:
        - starting_path (str or Path): The starting path to begin the search.
        Defaults to the current script's directory.
        - marker (str): The marker to look for (e.g., '.git', 'setup.py', 'README.md').

    Returns:
        Path: The Path object pointing to the root directory of the project,
        or None if not found.
    """
    # Start from the directory of the current file if not specified
    if starting_path is None:
        starting_path = Path(__file__).resolve().parent

    # Convert starting_path to a Path object if it's not already
    starting_path = Path(starting_path)

    # Traverse up the directory tree
    for parent in starting_path.parents:
        # Check if the marker exists in the current directory
        if (parent / marker).exists():
            return parent

    return None  # Return None if the marker is not found


def find_value_recursive(d: Dict[str, Any], key: str) -> Optional[Any]:
    """
    Recursively searches for a key in a nested dictionary, case-insensitive.

    Args:
        d (Dict[str, Any]): Dictionary to search.
        key (str): The key to find (case-insensitive).

    Returns:
        Optional[Any]: The value if found, else None.
    """
    if isinstance(d, dict):
        for k, v in d.items():
            if k.lower() == key.lower():  # Case-insensitive match
                return v
            if isinstance(v, dict):  # Recursively search in nested dictionaries
                result = find_value_recursive(v, key)
                if result is not None:
                    return result
    return None


def get_company_and_job_title(
    job_posting_url: str, json_data: Dict[str, Any]
) -> Dict[str, Optional[str]]:
    """
    Looks up the company and job title for a given job posting URL from the JSON data.
    This function searches recursively through all nested levels (case-insensitive).

    Args:
        job_posting_url (str): The URL of the job posting to look up.
        json_data (Dict[str, Any]): The JSON-like dictionary containing the job postings data.

    Returns:
        Dict[str, Optional[str]]: A dictionary containing 'company' and 'job_title',
        with values set to None if not found.
    """
    job_data: Optional[Dict[str, Any]] = json_data.get(job_posting_url)

    if job_data:
        company: Optional[str] = find_value_recursive(job_data, "company")
        job_title: Optional[str] = find_value_recursive(job_data, "job_title")

        return {"company": company, "job_title": job_title}

    print(f"Job posting not found for URL: {job_posting_url}")
    return {"company": None, "job_title": None}


def inspect_json_structure(obj):
    "Helper function to inspecct nested JSON obj structure."
    if isinstance(obj, dict):
        return {
            str(type(k).__name__): inspect_json_structure(v)
            for k, v in list(obj.items())[:1]
        }
    elif isinstance(obj, list):
        return [inspect_json_structure(obj[0])] if obj else []
    else:
        return type(obj).__name__


def is_existing(direcotry_path: Path, file_name: Path):
    """
    Checks if a file exists in the specified directory.

    Args:
        dir (str): Directory path
        file_name (str): File name

    Returns:
        bool: True if file exists, False otherwise
    """
    existing_files = get_file_names(
        directory_path=direcotry_path, full_path=False, file_type_inclusive=True
    )
    if file_name in existing_files:
        logger.info(f"File ({file_name}) already exists. Skipping this step.")
        return True
    return False


def load_or_create_json(filepath, key=None):
    """
    Loads a JSON file if it exists and checks if a specific key (e.g., URL) exists in it.
    If the file does not exist, it initializes an empty dictionary for creating a new entry.

    Args:
        filepath (str): The file path to the JSON file.
        key (str, optional): The specific key to check within the JSON data (e.g., a URL).

    Returns:
        tuple: (data, is_existing)
            - data (dict): The loaded or newly created JSON data.
            - is_existing (bool): True if the key exists in the JSON data, False otherwise.
    """
    # Check if the JSON file exists
    if check_if_json(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Check if the specific key exists in the JSON data
                if key and key in data:
                    logger.info(f"Data for key '{key}' already exists in '{filepath}'.")
                    return data, True  # Key exists, return data and True
                else:
                    return data, False  # Key does not exist, return data and False
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error reading {filepath}: {e}")
            return {}, False  # Return empty dict if there is an error
    else:
        # Initialize an empty dictionary if the file does not exist
        return {}, False  # Return empty dict and False since file does not exist


def normalize_dataframe_column_names(df):
    """
    This function converts the column names of a DataFrame to lowercase
    and replaces spaces with underscores.

    Args:
    - df (pd.DataFrame): Input DataFrame with columns to be normalized.

    Returns:
    - df (pd.DataFrame): DataFrame with normalized column names.
    """
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def pretty_print_json(data):
    """
    Pretty prints a Python dictionary or list in JSON format.

    Args:
        data (dict or list): The JSON data (Python dictionary or list) to be pretty printed.

    Returns:
        None
    """
    # Check if the data is a dictionary or list to format it as JSON
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=4))
    else:
        print("The provided data is not a valid JSON object (dict or list).")


@overload
def read_from_json_file(
    json_file: Union[str, Path], key: None = ...
) -> Dict[str, Any]: ...


@overload
def read_from_json_file(json_file: Union[str, Path], key: str) -> Any: ...


def read_from_json_file(
    json_file: Union[str, Path], key: Optional[str] = None
) -> Union[Dict[str, Any], Any]:
    """
    Loads data from a master JSON file and extracts specific sections or values.

    Args:
        json_file (str or Path): The path to the master JSON file.
        key (str, optional): If provided, extracts the data under this specific key.

    Returns:
        dict or Any: The extracted data from the JSON file.

    Raises:
        KeyError: If the specified key is not found in the JSON data.
        FileNotFoundError: If the JSON file is not found.
        JSONDecodeError: If there is an error decoding the JSON data.
    """
    json_file = Path(json_file)

    # Check if directory exists
    if not json_file.parent.exists():
        raise FileNotFoundError(
            f"Directory '{json_file.parent}' does not exist. Please create the directory first."
        )

    # Check if file exists; if not, return an empty dictionary
    if not json_file.exists():
        logger.info(f"File {json_file} not found. It will be created.")
        return {}

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            master_data = json.load(f)

        logger.info(f"Loaded data from {json_file}")

        # If a key is provided, extract data under that key
        if key:
            if key in master_data:
                logger.info(f"Data for key '{key}' found in {json_file}.")
                return master_data[key]
            else:
                raise KeyError(f"Key '{key}' not found in the JSON data.")
        else:
            return master_data  # Return the entire JSON data if no key is provided

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {json_file}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {json_file}: {e}")
        raise


def replace_spaces_in_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace all spaces with underscores in the column names of the DataFrame.

    Parameters:
    - df: The input DataFrame.

    Returns:
    - DataFrame with updated column names.
    """
    df.columns = df.columns.str.replace(" ", "_")
    return df


def save_to_json_file(data: Any, file_path: Union[str, Path]) -> None:
    """
    Saves a Python dictionary, list, or Pydantic model to a JSON file.

    Args:
        data (Any): The data to be saved in JSON format. Can be a dict, list, or Pydantic model.
        file_path (str): The full path to the file where the data will be saved.

    Raises:
        ValueError: If the data is not a dictionary, list, or Pydantic model.
        FileNotFoundError: If the provided file path's directory does not exist.
        Exception: For any general errors during file saving.

    Returns:
        None
    """
    # Change file_path to Path if it's str.
    file_path = Path(file_path)

    try:
        # Validate file path
        directory = Path(file_path).parent
        if not directory.exists():
            raise FileNotFoundError(
                f"Directory does not exist for the file path: {file_path}"
            )

        # Convert Pydantic models to dict
        if isinstance(data, BaseModel):
            data = data.model_dump()
            logger.debug("Converted Pydantic model to dictionary.")

        # Validate data type
        if not isinstance(data, (dict, list)):
            raise ValueError(
                f"Invalid data type. Expected dict, list, or Pydantic model, got {type(data).__name__}"
            )

        # Convert all dict keys and Path objects to strings
        serializable_data = convert_keys_and_paths_to_str(data)
        logger.debug("Converted all dictionary keys and Path objects to strings.")

        # Attempt to write data to the JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Data successfully saved to {file_path}.")

    except FileNotFoundError as e:
        logger.error(f"File path not found: {file_path} - Error: {e}")
        raise
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise
    except TypeError as e:
        logger.error(f"TypeError: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        raise


def validate_json_file(file: Path):
    if not file.exists():
        logger.error(f"File not found: {file}")
        return False
    if not file.suffix == ".json":
        logger.error(f"Invalid file format (expected .json): {file}")
        return False
    return True


def verify_file(file_path: Union[str, Path]) -> bool:
    """
    Verify if the file exists and is accessible.

    Args:
        file_path (Path): The path to the file to be checked.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    file_path = Path(file_path)
    if file_path.exists() and file_path.is_file():
        return True
    logger.error(f"File does not exist or is not a valid file: {file_path}")
    return False


def verify_dir(dir_path: Union[str, Path], create_dir: bool = False) -> bool:
    """
    Verify if the directory exists and is accessible. Create the directory
    if it doesn't exist.

    Args:
        dir_path (Path): The path to the directory to be checked.
        create_dir (bool): Create the directory if it doesn't exist.
        Defaults to False.
        logger (Logger): Logger instance. Defaults to the root logger.

    Returns:
        bool: True if the directory exists (or was created), False otherwise.
    """
    dir_path = Path(dir_path)

    if dir_path.exists() and dir_path.is_dir():
        return True

    try:
        if create_dir:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created: {dir_path}")
            return True
        else:
            logger.error(f"Directory: {dir_path} does not exist.")
            return False
    except OSError as e:
        logger.error(f"Failed to create directory: {dir_path}. Error: {e}")
        return False
