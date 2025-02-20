"""Async tool functions"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import io
import pandas as pd
import asyncio
import aiofiles
from pydantic import BaseModel

# User defined
from utils.get_file_names import get_file_names
from utils.generic_utils import convert_keys_and_paths_to_str
import logging_config

# Setup logger
logger = logging.getLogger(__name__)


async def add_to_and_update_json_file_async(
    new_data: Union[dict, list], json_file: Union[Path, str], key: Optional[str] = ""
) -> None:
    """
    Asynchronously adds or updates data in a master JSON file under a specific key.
    If there is only one key, it adds or updates that key. If there are multiple keys,
    it updates the existing ones and adds the new keys.

    Args:
        - new_data (dict or list): The data to be added or updated in the JSON file.
        - json_file (str): The path to the master JSON file.
        - key (str, optional):
            If provided, adds or updates the data under this specific key.
            If not provided, the new data is merged directly with the existing JSON data.

    Returns:
        None
    """
    try:
        # Open the file asynchronously
        async with aiofiles.open(json_file, "r", encoding="utf-8") as f:
            # Read the file content
            file_content = await f.read()
            # Parse the JSON data
            master_data = json.loads(file_content)

        logging.debug(
            f"Loaded data type: {type(master_data).__name__}, New data type: {type(new_data).__name__}"
        )

        # If key is provided, only add or update that specific key
        if key:
            if not isinstance(master_data, dict):
                raise ValueError(
                    "Master JSON data must be a dictionary when using a key."
                )

            # Update or add new data under the provided key
            if key in master_data:
                if isinstance(master_data[key], list) and isinstance(new_data, list):
                    master_data[key].extend(new_data)  # Extend the existing list
                elif isinstance(master_data[key], dict) and isinstance(new_data, dict):
                    master_data[key].update(new_data)  # Update the existing dictionary
                else:
                    raise ValueError(
                        f"Incompatible data types for merging under key '{key}': "
                        f"{type(master_data[key]).__name__} vs {type(new_data).__name__}."
                    )
            else:
                # If the key doesn't exist, add the new data
                master_data[key] = new_data

        else:
            # If key is not provided, handle the case where there are multiple keys
            if isinstance(new_data, dict):
                if not isinstance(master_data, dict):
                    raise ValueError(
                        "Cannot merge a dictionary with non-dictionary master data."
                    )

                # Iterate over each item in new_data
                for key, value in new_data.items():
                    # If the key exists, update it
                    if key in master_data:
                        if isinstance(master_data[key], list) and isinstance(
                            value, list
                        ):
                            master_data[key].extend(value)  # Extend the list
                        elif isinstance(master_data[key], dict) and isinstance(
                            value, dict
                        ):
                            master_data[key].update(value)  # Update the dictionary
                        else:
                            raise ValueError(
                                f"Incompatible data types for merging under key '{key}': "
                                f"{type(master_data[key]).__name__} vs {type(value).__name__}."
                            )
                    else:
                        # If the key doesn't exist, add it
                        master_data[key] = value

            elif isinstance(new_data, list):
                if isinstance(master_data, list):
                    master_data.extend(new_data)  # Extend the existing list
                else:
                    raise ValueError("Cannot add a list to non-list master data.")

            else:
                raise ValueError(
                    f"Cannot merge {type(master_data).__name__} with {type(new_data).__name__}."
                )

        # Save the updated master data back to the file asynchronously
        async with aiofiles.open(json_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(master_data, indent=4, ensure_ascii=False))

        logging.info(f"Data successfully added or updated in {json_file}.")

    except FileNotFoundError:
        # If the file doesn't exist, create a new one with the new data
        logging.warning(
            f"File {json_file} not found. Creating new file and adding data."
        )
        async with aiofiles.open(json_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(new_data, indent=4, ensure_ascii=False))
        logging.info(f"File {json_file} created and data added.")

    except (KeyError, ValueError) as e:
        logging.error(f"Error adding or updating data in {json_file}: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error adding or updating data in {json_file}: {e}")
        raise


async def add_new_data_to_json_file_async(
    new_data: Union[dict, list], filename: Union[Path, str], key: Optional[str] = ""
) -> None:
    """
    Asynchronously adds new data to a master JSON file, ensuring no duplicate keys are added.
    Handles both dictionaries and lists for new_data.

    Args:
        - new_data (dict or list): The data to be added to the JSON file.
        - filename (str): The path to the master JSON file.
        - key (str): The specific key under which the new data will be added (optional).

    Returns:
        None
    """
    if not isinstance(new_data, (dict, list)):
        raise ValueError("new_data must be a dictionary or list.")

    try:
        # Load existing data from the file
        async with aiofiles.open(filename, "r", encoding="utf-8") as f:
            master_data = json.loads(
                await f.read()
            )  # json.loads works b/c running async (load is only for sync)

        logging.debug(
            f"Loaded data type: {type(master_data).__name__}, New data type: {type(new_data).__name__}"
        )

        if key:
            if not isinstance(master_data, dict):
                raise ValueError(
                    "Master JSON data must be a dictionary when using a key. "
                    f"Found {type(master_data).__name__} instead."
                )
            if key in master_data:
                logging.warning(
                    f"Key '{key}' already exists in the file. Skipping update."
                )
            else:
                master_data[key] = new_data
        else:
            if isinstance(new_data, dict):
                if not isinstance(master_data, dict):
                    raise ValueError(
                        "Cannot merge a dictionary with non-dictionary master data."
                    )
                for k, v in new_data.items():
                    if k not in master_data:
                        master_data[k] = v
                    else:
                        logging.warning(
                            f"Key '{k}' already exists in the file. Skipping update."
                        )
            elif isinstance(new_data, list):
                if isinstance(master_data, list):
                    master_data.extend(new_data)
                else:
                    raise ValueError(
                        "Cannot add a list to non-list master data without a key."
                    )
            else:
                raise ValueError(
                    f"Cannot merge {type(master_data).__name__} with {type(new_data).__name__}."
                )

        # Save the updated master data back to the file
        async with aiofiles.open(filename, "w", encoding="utf-8") as f:
            await f.write(
                json.dumps(master_data, indent=4, ensure_ascii=False)
            )  # use dumps instead dump b/c running async

        logging.info(f"New data successfully added to {filename}.")

    except FileNotFoundError:
        # Initialize the file based on the type of new_data
        if key:
            master_data = {key: new_data}
        else:
            master_data = new_data if isinstance(new_data, (dict, list)) else [new_data]
        async with aiofiles.open(filename, "w", encoding="utf-8") as f:
            await f.write(
                json.dumps(master_data, indent=4, ensure_ascii=False)
            )  # use dumps
        logging.info(f"File {filename} created and new data added.")

    except (KeyError, ValueError) as e:
        logging.error(f"Error adding new keys to {filename}: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error adding new keys to {filename}: {e}")
        raise


def check_json_file(file_path):
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
        logging.info(f"The file '{file_path}' already exists.")
        return True
    else:
        logging.info(
            f"The file '{file_path}' does not exist or does not have a .json extension."
        )
        return False


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
    Fetch job posting URLs from the new jobs file, compare them with the existing jobs file,
    and return only the new URLs that have not been previously processed.

    Args:
        job_descriptions_json_file (str): File path of the JSON file containing already processed job postings.
                                          Typically from 'job_postings.json'.
        list_of_urls_file (str): File path of the JSON file containing the list of new job posting URLs.

    Returns:
        list: A list of URLs that are new and have not been processed yet.

    Notes:
        - Both files must be in valid JSON format.
        - The existing job descriptions file should use URLs as keys.
        - The new jobs file should have a structure where job postings are listed under "jobs" and each job has a "url" field.
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


def find_project_root(starting_path=None, marker=".git"):
    """
    Recursively find the root directory of the project by looking for a specific marker.

    Args:
        starting_path (str or Path): The starting path to begin the search. Defaults to the current script's directory.
        marker (str): The marker to look for (e.g., '.git', 'setup.py', 'README.md').

    Returns:
        Path: The Path object pointing to the root directory of the project, or None if not found.
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


def get_company_and_job_title(job_posting_url, json_data):
    """
    Looks up the company and job title for a given job posting URL from the JSON data.

    Args:
        job_posting_url (str): The URL of the job posting to look up.
        json_data (dict): The JSON-like dictionary containing the job postings data.

    Returns:
        dict: A dictionary containing the 'company' and 'job_title' if found, otherwise None for both.

    Example:
        json_data = json.load(open('job_postings.json'))
        result = get_company_and_job_title('url...', json_data)
        print(result)
        # Output: {'company': 'Google', 'job_title': 'AI Market Intelligence Principal'}
    """
    # Get the job posting data by URL
    job_data = json_data.get(job_posting_url)

    if job_data:
        company = job_data.get("company", None)
        job_title = job_data.get("job_title", None)
        return {"company": company, "job_title": job_title}
    else:
        print(f"Job posting not found for URL: {job_posting_url}")
        return {"company": None, "job_title": None}


def is_existing(dir, file_name):
    """
    Checks if a file exists in the specified directory.

    Args:
        dir (str): Directory path
        file_name (str): File name

    Returns:
        bool: True if file exists, False otherwise
    """
    existing_files = get_file_names(
        dir_path=dir, full_path=False, file_type_inclusive=True
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
    if check_json_file(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Check if the specific key exists in the JSON data
                if key and key in data:
                    logging.info(
                        f"Data for key '{key}' already exists in '{filepath}'."
                    )
                    return data, True  # Key exists, return data and True
                else:
                    return data, False  # Key does not exist, return data and False
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.error(f"Error reading {filepath}: {e}")
            return {}, False  # Return empty dict if there is an error
    else:
        # Initialize an empty dictionary if the file does not exist
        return {}, False  # Return empty dict and False since file does not exist


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


async def read_from_json_async(filename, key=None):
    """
    Asynchronous version of reading a JSON file:
    Loads data from a master JSON file and extracts specific sections or values.

    Args:
        filename (str): The path to the master JSON file.
        key (str, optional): If provided, extracts the data under this specific key.

    Returns:
        dict or list: The extracted data from the JSON file.
        None: If the file does not exist or is empty.

    Raises:
        KeyError: If the specified key is not found in the JSON data.
        FileNotFoundError: If the JSON file is not found.
        JSONDecodeError: If there is an error decoding the JSON data.
    """
    # Check directory exist or not
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        raise FileNotFoundError(
            f"Directory '{directory}' does not exist. Please create the directory first."
        )

    # Check if the file exists; if not, return an empty dictionary to handle the creation
    if not os.path.exists(filename):
        logging.info(f"File {filename} not found. It will be created.")
        return {}

    try:
        async with aiofiles.open(filename, "r", encoding="utf-8") as f:
            file_content = await f.read()

        # Parse the JSON content
        master_data = json.loads(file_content)

        # Debugging: Log the loaded data structure
        logger.info(f"Loaded data from {filename}")

        # If a key is provided, extract data under that key
        if key:
            if key in master_data:
                logging.info(f"Data for key '{key}' found in {filename}.")
                return master_data[key]
            else:
                raise KeyError(f"Key '{key}' not found in the JSON data.")
        else:
            return master_data  # Return the entire JSON data if no key is provided

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filename}: {e}")
        logger.error(f"Error decoding JSON from {filename}: {e}")
        raise  # Re-raise the exception to handle it outside the function
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        logger.error(f"Error loading data from {filename}: {e}")
        raise  # Re-raise any other exceptions to handle them outside the function


async def read_from_csv_async(filepath):
    """Asynchronously read CSV file using aiofiles and io.StringIO."""
    async with aiofiles.open(filepath, mode="r") as f:
        content = await f.read()
    # Use Python's built-in io.StringIO to read CSV from the string content
    return pd.read_csv(io.StringIO(content))


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


async def save_df_to_csv_file_async(
    df: pd.DataFrame, filepath: Union[Path, str]
) -> None:
    """
    Asynchronously save a DataFrame as a CSV file.

    Args:
        - df (pd.DataFrame): The DataFrame to be saved.
        - filepath (Path or str): The full path where the CSV file will be stored.

    * Note on Async CSV Writing:
    `aiofiles.open()` **does not support** `pandas.DataFrame.to_csv()` directly because
    `to_csv()` writes to file handles synchronously, while `aiofiles` requires asynchronous
    string-based writing.

    Two Solutions:
    1. **For full async writing** (small/medium files, cloud storage):
    - Convert the DataFrame to a string using `StringIO`, then write asynchronously with
    `aiofiles.open()`.
    - This avoids blocking but may not be ideal for large files.

    2. **For high-performance writing** (large files):
    - Use `asyncio.to_thread(df.to_csv, str(filepath))` to offload the CSV writing to
    a separate thread.
    - This leverages multithreading for better efficiency without blocking async execution.
    """

    filepath = Path(filepath)  # Convert to Path obj if it's str

    # Ensure parent directory exists
    if not filepath.parent.exists():
        logger.error(f"Directory '{filepath.parent}' does not exist.")
        raise FileNotFoundError(f"Directory '{filepath.parent}' does not exist.")

    df.reset_index(drop=True, inplace=True)  # Reset and drop original index

    try:
        # Convert to string first and then save
        csv_data = df.to_csv(index=False, encoding="utf-8")

        # Write CSV asynchronously
        async with aiofiles.open(filepath, mode="w", encoding="utf-8") as f:
            await f.write(csv_data)

        logger.info(f"Successfully saved DataFrame to {filepath}")

    except OSError as e:
        logger.error(f"OS error writing to file {filepath}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving DataFrame to {filepath}: {e}")
        raise


async def save_data_to_json_file_async(
    data: Union[Dict, List, BaseModel, pd.DataFrame], file_path: Union[str, Path]
) -> None:
    """
    Asynchronously saves a Python dictionary, list, Pydantic model, or DataFrame
    to a JSON file.

    Args:
        - data (Union[Dict, List, BaseModel, pd.DataFrame]): The data to be saved in
        JSON format. Can be a dict, list, Pydantic model, or DataFrame.
        - file_path (str or Path): The full path to the file where the data will be saved.

    Raises:
        - ValueError: If the data is not serializable.
        - FileNotFoundError: If the provided file path's directory does not exist.
        - IOError: For any I/O-related errors during file saving.

    Returns:
        None
    """
    # Convert file_path to Path if it's a string
    file_path = Path(file_path)

    try:
        # Validate file path
        directory = file_path.parent
        if not directory.exists():
            raise FileNotFoundError(
                f"Directory does not exist for the file path: {file_path}"
            )

        # Convert Pydantic models to dictionary
        if isinstance(data, BaseModel):
            data = data.model_dump()
            logger.debug("Converted Pydantic model to dictionary.")

        # Convert DataFrame to list of dictionaries
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")
            logger.debug("Converted DataFrame to list of dictionaries.")

        # Validate data type
        if not isinstance(data, (dict, list)):
            raise ValueError(
                f"Invalid data type. Expected dict, list, Pydantic model, or DataFrame, got {type(data).__name__}"
            )

        # Asynchronously write data to the JSON file
        async with aiofiles.open(file_path, mode="w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=4, ensure_ascii=False))

        logger.info(f"✅ Data successfully saved to {file_path}.")

    except FileNotFoundError as e:
        logger.error(f"❌ File path not found: {file_path} - Error: {e}")
        raise
    except ValueError as e:
        logger.error(f"❌ ValueError: {e}")
        raise
    except TypeError as e:
        logger.error(f"❌ TypeError: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to save JSON to {file_path}: {e}")
        raise
