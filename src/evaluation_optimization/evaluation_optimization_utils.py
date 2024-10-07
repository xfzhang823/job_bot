"""
evaluation_optimization_utils.py
Utility functions to support evaluation and optimization pipelines.
"""

from pathlib import Path
import os
import re
import logging
import logging_config
from typing import List
import pandas as pd
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


def create_file_name(
    company: str, job_title: str, suffix: str = None, ext: str = ".csv"
):
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
    if not company and not job_title:
        return None

    if suffix:
        suffix = f"_{suffix}"

    # Clean names for file saving
    illegal_chars = r"[^a-zA-Z0-9._-]"
    company = re.sub(illegal_chars, "_", company or "Unknown_Company")
    job_title = re.sub(illegal_chars, "_", job_title or "Unknown_Title")

    return f"{company}_{job_title}{suffix}.{ext}"


def check_file_existence(file_path):
    """
    Check if a file exists at the given path.

    Args:
        file_path (str or Path): The path to the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return Path(file_path).is_file()


def find_files_wo_multivariate_indices(
    data_directory: str, indices: list = None
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


def process_and_save_requirements_by_url(
    requirements_json_file: str, url: str, requirements_flat_json_file: str
):
    """
    Extract, flatten, and save job requirements from the JSON file, filtered by a specific URL.

    This function reads the job posting JSON file, filters the requirements based on the provided URL,
    flattens the nested structure, and saves the flattened requirements to a JSON file.

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
    logger.info(f"Requirements flattened and saved to {requirements_flat_json_file}")


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
