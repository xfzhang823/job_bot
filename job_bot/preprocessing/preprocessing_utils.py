"""Helper functions for preprocessing functions"""

import logging
import json
from pathlib import Path
from typing import List, Tuple
from utils.generic_utils import read_from_json_file

logger = logging.getLogger(__name__)


def check_and_read_from_json_file(file_path: Path) -> dict:
    """Reads a JSON file safely, returning an empty dictionary if the file is missing or empty."""
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}. Returning empty dictionary.")
        return {}
    data = read_from_json_file(file_path)
    if not data:
        logger.warning(f"File is empty: {file_path}. Returning empty dictionary.")
        return {}
    return data


def find_new_urls(
    job_posting_urls_file: Path,
    job_descriptions_file: Path,
    job_requirements_file: Path,
) -> Tuple[List[str], List[str]]:
    """
    Reads job posting, description, and requirement files, then calculates new
    and semi-new URLs.

    Args:
        - job_posting_urls_file (Path): Path to the job postings JSON file.
        - job_descriptions_file (Path): Path to the job descriptions JSON file.
        - job_requirements_file (Path): Path to the job requirements JSON file.

    Returns:
        - Tuple[List[str], List[str]]:
            - new_urls (URLs not in job descriptions or requirements).
            - semi_new_urls (URLs in job descriptions but not in requirements).
    """
    logger.info("Checking for new urls and semi_new urls...")
    # Read job postings, job descriptions, and job requirements
    job_posting_data = check_and_read_from_json_file(job_posting_urls_file)
    job_description_data = check_and_read_from_json_file(job_descriptions_file)
    job_requirement_data = check_and_read_from_json_file(job_requirements_file)

    # Extract URLs
    job_posting_urls = set(job_posting_data.keys())
    job_description_urls = set(job_description_data.keys())
    job_requirement_urls = set(job_requirement_data.keys())

    # todo: debugging: delete later
    logger.info(f"No. of job posting urls: {len(job_posting_urls)}")
    logger.info(f"No. of job descriptions urls: {len(job_description_urls)}")
    logger.info(f"No. of job requirements urls: {len(job_requirement_urls)}")

    # Compute new and semi-new URLs

    # New URLs → URLs in postings but missing from descriptions
    new_urls = list(job_posting_urls - job_description_urls)

    # Semi-new URLs → URLs in descriptions but missing from requirements
    missing_requirements_urls = list(job_description_urls - job_requirement_urls)

    logger.info(
        f"Identified {len(new_urls)} new URLs and {len(missing_requirements_urls)} semi-new URLs."
    )
    logger.info("Finished checking for new and semi-new urls.")

    return new_urls, missing_requirements_urls
