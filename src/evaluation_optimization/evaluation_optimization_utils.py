"""
evaluation_optimization_utils.py
Utility functions to support evaluation and optimization pipelines.
"""

from pathlib import Path
import re
import logging
import logging_config
from utils.generic_utils import read_from_json_file

# Set up logging
logger = logging.getLogger(__name__)


def create_file_name(company, job_title):
    """
    Create a standardized file name from company and job title.

    Args:
        company (str): The company name.
        job_title (str): The job title.

    Returns:
        str or None: A standardized file name, or None if both company and job title are missing.
    """
    if not company and not job_title:
        return None

    # Clean names for file saving
    illegal_chars = r"[^a-zA-Z0-9._-]"
    company = re.sub(illegal_chars, "_", company or "Unknown_Company")
    job_title = re.sub(illegal_chars, "_", job_title or "Unknown_Title")

    return f"{company}_{job_title}.csv"


def check_file_existence(file_path):
    """
    Check if a file exists at the given path.

    Args:
        file_path (str or Path): The path to the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return Path(file_path).is_file()


def get_new_urls_and_file_names(job_descriptions, output_dir):
    """
    Get URLs that don't have corresponding files in the output directory,
    and return a dictionary with the new URLs as keys and their corresponding file names as values.

    Args:
        job_descriptions (dict): A dictionary of job descriptions, with URLs as keys.
        output_dir (str or Path): The directory to check for existing files.

    Returns:
        dict: A dictionary where the keys are new URLs and the values are the corresponding file names.
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
