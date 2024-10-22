"""
evaluation_optimization_utils.py
Utility functions to support evaluation and optimization pipelines.
"""

from pathlib import Path
import os
import re
import logging
import logging_config
from typing import List, Union
import pandas as pd
import json
from utils.generic_utils import read_from_json_file
from utils.get_file_names import get_file_names
from evaluation_optimization.multivariate_indexer import MultivariateIndexer
import os
import logging
from utils.generic_utils import save_to_json_file
from preprocessing.resume_preprocessor import ResumeParser
from preprocessing.requirements_preprocessor import JobRequirementsParser

# Set up logging
logger = logging.getLogger(__name__)


# Helper function to create standardized file names
def create_file_name(
    company: str, job_title: str, suffix: str = "", ext: str = ".json"
) -> str | None:
    """
    Create a standardized file name from company and job title.

    Args:
        company (str): The company name.
        job_title (str): The job title.
        suffix (str): suffix to add at the end of the file name.
        ext (str): file extension (default to ".csv")

    Returns:
        str or None: A standardized file name, or None if both company and job title are missing.
    """
    # Return None if both company and job title are empty
    if not company.strip() and not job_title.strip():
        return None

    # Check if suffix exists; underscore prefix to suffix if it doesn't already start with one
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    # If suffix starts with "_", leave it as is
    else:
        suffix = suffix or ""

    # Clean up company and job title names (remove/replace invalid characters)
    illegal_chars = r"[^\w\s-]"  # Remove everything except alphanumeric
    # characters, spaces, dashes, and underscores
    company = re.sub(illegal_chars, "_", company or "Unknown_Company").replace(" ", "_")
    job_title = re.sub(illegal_chars, "_", job_title or "Unknown_Title").replace(
        " ", "_"
    )

    # Handle empty strings
    suffix = suffix or ""
    ext = ext or ".json"

    # Ensure there are no double periods or trailing periods
    clean_file_name = f"{company}_{job_title}{suffix}.{ext}".replace("..", ".")

    return clean_file_name


def check_file_existence(file_path):
    """
    Check if a file exists at the given path.

    Args:
        file_path (str or Path): The path to the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return Path(file_path).is_file()


def get_files_wo_multivariate_indices(
    data_directory: Union[str, Path], indices: list = None
) -> list:
    """
    Find .csv files in the specified directory that contain the required metrics
    but do not contain the specified multivariate indices.

    Parameters:
    - data_directory: The path to the directory containing .csv files.
    - indices (list): List of index column names to check for.
                      Default is composite score and PCA score.

    Returns:
    - List of full file paths to .csv files without the specified multivariate indices.
    """
    # Change data_directory to Path obj if it's str.
    data_directory = Path(data_directory)

    if indices is None:
        indices = ["composite_score", "pca_score"]
    if not os.path.exists(data_directory):
        raise ValueError(f"The provided directory '{data_directory}' does not exist.")

    # Get the list of .csv files in the directory
    file_list = get_file_names(
        directory_path=data_directory,
        full_path=True,
        recursive=True,
        file_types=[".csv"],
    )

    files_without_indices = []

    # Check each file for the required metrics and the specified indices
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)

            # Log the columns of the current file
            logger.info(f"Processing file: {file_path}, columns: {df.columns}")

            # Initialize the MultivariateIndexer class (internally validates the metrics)
            indexer = MultivariateIndexer(df)

            # Validate if the required metrics are present
            indexer.validate_metrics()

            # Check if any of the specified indices are missing in the columns
            missing_indices = [index for index in indices if index not in df.columns]
            if missing_indices:
                logger.info(f"File {file_path} is missing indices: {missing_indices}")
                files_without_indices.append(file_path)
            else:
                logger.info(f"File {file_path} already has all the required indices.")

        except pd.errors.EmptyDataError:
            logger.warning(f"Warning: {file_path} is empty and was skipped.")
        except ValueError as ve:
            logger.warning(f"Metrics validation failed for {file_path}: {ve}")
            continue
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    return files_without_indices


# Main function to handle flat json and sim metrics file creation
def get_file_paths_and_create_files(
    job_descriptions,
    output_dir,
    suffixes=None,
    create_func=None,
):
    """
    Get URLs that don't have corresponding files in the output directory and create files if necessary.

    Args:
        job_descriptions (dict): A dictionary of job descriptions with URLs as keys.
        output_dir (str): Directory where files are saved.
        suffixes (list): List of suffixes for different file types (without extension).
        create_func (callable, optional): A function to create and save files (e.g., process requirements/responsibilities).

    Returns:
        dict: A dictionary where the keys are new URLs and the values are the created file paths.
    """
    if suffixes is None:
        suffixes = ["reqs_flat", "resps_flat", "sim_metrics"]

    new_urls_and_file_paths = {}

    for url, info in job_descriptions.items():
        company = info.get("company")
        job_title = info.get("job_title")

        for suffix in suffixes:
            # Create file name dynamically for each type (suffix) with proper extension
            file_name = create_file_name(
                company=company,
                job_title=job_title,
                suffix=suffix,
                ext=(
                    ".json" if "flat" in suffix else ".csv"
                ),  # Flat files as .json, metrics as .csv
            )

            if file_name:
                file_path = Path(output_dir) / file_name

                # If the file doesn't exist, add it to the processing list
                if not file_path.exists():
                    new_urls_and_file_paths[url] = file_path

                    # Call the optional file creation function if provided
                    if create_func:
                        create_func(url, file_path)
                else:
                    logger.info(f"File already exists: {file_path}, skipping creation.")
            else:
                logger.warning(
                    f"Skipping URL due to missing company or job title: {url}"
                )

    return new_urls_and_file_paths


def get_new_urls_and_file_names(job_descriptions, output_dir):
    """
    Get URLs that don't have corresponding files in the output directory,
    and return a dictionary with the new URLs as keys and their corresponding file names as values.

    Args:
        job_descriptions (dict): A dictionary of job descriptions, with URLs as keys.
        output_dir (str or Path): The directory to check for existing files.

    Returns:
        dict: A dictionary where the keys are new URLs and the values are the corresponding \
            file names (w/o the full path).
    """
    new_urls_and_file_names = {}
    for url, info in job_descriptions.items():
        company = info.get("company")
        job_title = info.get("job_title")
        file_name = create_file_name(company, job_title)
        if file_name:
            file_path = Path(output_dir) / file_name
            if not check_file_existence(file_path):
                new_urls_and_file_names[url] = file_name
        else:
            logger.warning(f"Skipping URL due to missing company and job title: {url}")

    return new_urls_and_file_names


def get_new_urls_and_metrics_file_paths(job_descriptions, output_dir):
    """
    Get URLs that don't have corresponding files in the output directory,
    and return a dictionary with the new URLs as keys and their corresponding file names as values.

    Args:
        job_descriptions (dict): A dictionary of job descriptions, with URLs as keys.
        output_dir (str or Path): The directory to check for existing files.

    Returns:
        dict: A dictionary where the keys are new URLs and the values are the corresponding \
            file names (w/o the full path).
    """
    new_urls_and_file_paths = {}
    for url, info in job_descriptions.items():
        company = info.get("company")
        job_title = info.get("job_title")
        file_name = create_file_name(company, job_title)
        if file_name:
            file_path = Path(output_dir) / file_name
            if not check_file_existence(file_path):
                new_urls_and_file_paths[url] = file_path
        else:
            logger.warning(f"Skipping URL due to missing company and job title: {url}")
    logger.info("New urls found and metrics file paths created.")
    return new_urls_and_file_paths


def get_new_urls_and_flat_json_file_paths(job_descriptions, output_dir):
    """
    Get URLs that don't have corresponding files in the output directory,
    and return a dictionary with the new URLs as keys and their corresponding file names as values.

    Args:
        job_descriptions (dict): A dictionary of job descriptions, with URLs as keys.
        output_dir (str or Path): The directory to check for existing files.

    Returns:
        new_urls_and dict: A dictionary where the keys are new URLs and the values are the corresponding \
            file names (w/o the full path).
    """
    new_urls_and_file_paths = {}
    for url, info in job_descriptions.items():
        company = info.get("company")
        job_title = info.get("job_title")
        file_name = create_file_name(
            company, job_title, suffix="requirements_flat", ext=".json"
        )
        if file_name:
            file_path = Path(output_dir) / file_name
            if not check_file_existence(file_path):
                new_urls_and_file_paths[url] = file_path
        else:
            logger.warning(f"Skipping URL due to missing company and job title: {url}")
    logger.info("New urls found and flat JSON file paths created.")
    return new_urls_and_file_paths


# def generate_and_save_file_mapping(
#     job_descriptions, output_dir, output_file, suffixes=None, append=False
# ):
#     """
#     Generate a mapping of URLs to file paths for requirements and responsibilities,
#     and save it to a file. If append is True, it will add to the existing mapping file.

#     Args:
#         -job_descriptions (dict): A dictionary containing job descriptions with URLs as keys.
#         -output_dir (Path): Directory where the files are stored.
#         -output_file (Path): Path to the file where the mapping will be saved.
#         -suffixes (list): List of suffixes for file names (e.g., 'reqs_flat', 'resps_flat').
#         -append (bool): Whether to append to the existing file or overwrite it.
#         Defaults to False (overwrite).

#     Returns:
#         None
#     """
#     # Safe default for suffixes
#     if suffixes is None:
#         suffixes = ["reqs_flat", "resps_flat", "sim_metrics"]

#     file_mapping = {}

#     # If appending, load the existing mapping if it exists
#     if append and output_file.exists():
#         with open(output_file, "r") as f:
#             file_mapping = json.load(f)
#         print(f"Loaded existing mapping from {output_file}")

#     # Generate new mappings and add to the existing file_mapping
#     for url, info in job_descriptions.items():
#         company = info.get("company")
#         job_title = info.get("job_title")

#         # Generate file names for requirements, responsibilities, and metrics
#         mapping_entry = {}
#         for suffix in suffixes:
#             ext = ".json" if "flat" in suffix else ".csv"
#             file_name = f"{company}_{job_title}_{suffix}{ext}"
#             file_path = output_dir / file_name
#             mapping_entry[suffix] = str(file_path)  # Store the file path as a string

#         # Add the new mapping to the existing file_mapping
#         file_mapping[url] = mapping_entry

#     # Save the updated mapping back to the file
#     with open(output_file, "w") as f:
#         json.dump(file_mapping, f, indent=4)

#     print(f"Mapping saved to {output_file}")


# Function to create and save job requirements JSON files
def process_and_save_requirements_by_url(
    requirements_json_file: str, url: str, requirements_flat_json_file: str
):
    """
    This function:
    - reads and extract more important job requirements from a single jobposting record,
    within a large JSON file, filtered by filtered by a specific URL (each posting has
    a unique url)
    - flattens the nested structure
    - saves them in another JSON file.

    Args:
        requirements_json_file (str): The file path to the job requirements JSON file.
        url (str): The URL to filter the job requirements.
        requirements_flat_json_file (str): The file path to save the flattened requirements JSON.

    Returns:
        None
    """

    # Check if the flattened JSON file already exists
    if os.path.exists(requirements_flat_json_file):
        logger.info(
            f"Flattened requirements JSON file already exists: {requirements_flat_json_file}"
        )
        return

    # Parse and flatten job requirements based on URL
    job_reqs_parser = JobRequirementsParser(requirements_json_file, url)
    reqs_flat = job_reqs_parser.extract_flatten_reqs()

    # Save the flattened requirements to a JSON file
    save_to_json_file(reqs_flat, requirements_flat_json_file)
    logger.info(f"requirements flattened and saved to {requirements_flat_json_file}")


def process_and_save_responsibilities_from_resume(
    resume_json_file: str, responsibilities_flat_json_file: str
):
    """
    Extract, flatten, and save responsibilities from the resume JSON file.

    This function reads the resume JSON file (containing a single record), flattens
    the nested structure of responsibilities, and saves the flattened responsibilities to a JSON file.

    Args:
        resume_json_file (str): The file path to the resume JSON file.
        responsibilities_flat_json_file (str): The file path to save the flattened responsibilities JSON.

    Returns:
        None
    """

    # Check if the flattened JSON file already exists
    if os.path.exists(responsibilities_flat_json_file):
        logger.info(
            f"Flattened responsibilities JSON file already exists: {responsibilities_flat_json_file}"
        )
        return

    # Parse and flatten responsibilities from the resume
    resume_parser = ResumeParser(resume_json_file)
    resps_flat = resume_parser.extract_and_flatten_responsibilities()

    # Save the flattened responsibilities to a JSON file
    save_to_json_file(resps_flat, responsibilities_flat_json_file)
    logger.info(
        f"Responsibilities flattened and saved to {responsibilities_flat_json_file}"
    )


class DataMerger:
    """
    Class responsible for merging multiple responsibility vs requirement metrics DataFrames on \
        the specified keys.
    """

    def __init__(self, dfs: List[pd.DataFrame], resp_key: str, req_key: str):
        self.dfs = dfs
        self.resp_key = resp_key
        self.req_key = req_key
        self.df_merged = None

    def merge_dataframes(self):
        """
        Merge multiple DataFrames on the responsibility_key and requirement_key.
        """
        df_merged = self.dfs[0]
        for i, df in enumerate(self.dfs[1:], start=1):
            df_merged = pd.merge(
                df_merged,
                df,
                on=[self.resp_key, self.req_key],
                suffixes=(f"_{i-1}", f"_{i}"),
                how="inner",
            )
        self.df_merged = df_merged
        return self.df_merged
