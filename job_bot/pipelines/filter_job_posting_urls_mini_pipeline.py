"""
Mini pipeline to mask job URLs to allow running small batches of URLs.
This script loads a main job postings JSON file and a JSON file containing URLs to exclude,
filters out any job postings whose keys (URLs) are in the exclusion list, and saves the filtered
dictionary to a new JSON file.
"""

from pathlib import Path
import logging

from job_bot.utils.generic_utils import read_from_json_file, save_to_json_file
from job_bot.config.project_config import (
    JOB_POSTING_URLS_FILE,
    JOB_POSTING_URLS_TO_EXCLUDE_FILE,
    JOB_POSTING_URLS_FILTERED_FILE,
)

logger = logging.getLogger(__name__)


def run_filtering_job_posting_urls_mini_pipe_line(
    all_urls_file: Path | str = JOB_POSTING_URLS_FILE,
    urls_to_exclude_file: Path | str = JOB_POSTING_URLS_TO_EXCLUDE_FILE,
    filtered_urls_file: Path | str = JOB_POSTING_URLS_FILTERED_FILE,
) -> None:
    """
    Filters job posting URLs by excluding those specified in an exclusion JSON file.

    This function performs the following steps:
    1. Loads the main job postings from a JSON file.
    2. Loads the exclusion URLs from a separate JSON file.
    3. Filters out job postings whose URLs are present in the exclusion list.
    4. Saves the filtered job postings to a new JSON file.

    Args:
        all_urls_file (Path | str): Path to the JSON file containing all job posting URLs.
        urls_to_exclude_file (Path | str): Path to the JSON file containing URLs to exclude.
        filtered_urls_file (Path | str): Path to the output JSON file where the filtered job
        postings will be saved.
    """
    try:
        all_urls_file = Path(all_urls_file)
        urls_to_exclude_file = Path(urls_to_exclude_file)
        filtered_urls_file = Path(filtered_urls_file)

        logger.info(f"Loading main job postings from {all_urls_file}")
        all_jobs = read_from_json_file(all_urls_file)

        logger.info(f"Loading exclusion URLs from {urls_to_exclude_file}")
        exclude_jobs = read_from_json_file(urls_to_exclude_file)
    except Exception as e:
        logger.error(f"Error reading JSON files: {e}")
        return

    # Create a set of URLs to exclude (the keys from the exclusion file)
    exclude_set = set(exclude_jobs.keys())
    logger.info(f"Excluding {len(exclude_set)} URLs from main job postings.")

    # Filter out any job postings whose key (URL) is in the exclude_set
    filtered_jobs = {
        url: details for url, details in all_jobs.items() if url not in exclude_set
    }

    logger.info(
        f"Filtered out {len(all_jobs) - len(filtered_jobs)} job postings; {len(filtered_jobs)} remain."
    )

    try:
        save_to_json_file(data=filtered_jobs, file_path=filtered_urls_file)
        logger.info(f"Filtered job postings saved successfully to {filtered_urls_file}")
    except Exception as e:
        logger.error(f"Error saving filtered job postings to {filtered_urls_file}: {e}")


if __name__ == "__main__":
    run_filtering_job_posting_urls_mini_pipe_line(
        all_urls_file=JOB_POSTING_URLS_FILE,
        urls_to_exclude_file=JOB_POSTING_URLS_TO_EXCLUDE_FILE,
        filtered_urls_file=JOB_POSTING_URLS_FILTERED_FILE,
    )
