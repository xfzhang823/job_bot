"""
Module: exclude_responsibilities_mini_pipeline.py

This module defines a mini pipeline for excluding specific responsibilities from a set of JSON files
based on the provided file mapping. It reads a mapping file, processes each job posting's associated 
responsibility files, and removes specific keys/values from the responsibility data. The pruned data 
is saved back to the output files.

Functions:
    run_pipeline(mapping_file): Main function to run the pipeline. Loads the mapping file, processes 
                                responsibilities, prunes keys, and saves the updated files.
"""

from pathlib import Path
import logging
import logging_config
from utils.generic_utils import read_from_json_file, save_to_json_file
from utils.exclude_and_reinsert_resps_keys import exclude_keys

logger = logging.getLogger(__name__)


def run_pipeline(mapping_file):
    """
    Run the exclusion mini pipeline to remove specific responsibility keys from job posting files.

    Args:
        mapping_file (str): Path to the JSON mapping file that defines the input and output
                            file paths for each job posting.

    Returns:
        None: The function logs errors and info but does not return any values.

    Steps:
        1. Load the mapping file using the read_from_json_file function.
        2. Iterate through each job posting URL and its corresponding file paths.
        3. Read responsibilities from the 'resps_flat' file.
        4. Remove the specified responsibility keys/values using the exclude_keys function.
        5. Save the pruned responsibility data to the 'pruned_resps_flat' file.

    Raises:
        FileNotFoundError: If the mapping file cannot be found.
        Exception: For other errors during file operations.
    """
    # Step 1: Load the mapping file
    try:
        file_mapping = read_from_json_file(mapping_file)
        logger.info(f"Loaded mapping file from {mapping_file}")
    except FileNotFoundError:
        logger.error(f"Mapping file not found: {mapping_file}")
        return
    except Exception as e:
        logger.error(f"Error loading mapping file: {e}")
        return

    # Step 2: Remove responsibilities keys/values from responsibilities files
    for url, paths in file_mapping.items():
        logger.info(f"Processing job posting from {url}")

        # Step 2.1: Extract file paths for responsibilities and requirements
        resps_file = Path(paths["resps"])
        pruned_file = Path(paths["pruned_resps"])

        # Read responsibilities JSON, exclude keys, and save the pruned data
        resps_json = read_from_json_file(resps_file)
        pruned_resps_json = exclude_keys(resps_json)
        save_to_json_file(pruned_resps_json, pruned_file)
