"""
Filename: copying_requirements_mini_pipeline.py

This module sets up the input/output files for job posting requirements from iteration 1.
It includes functions to copy the requirements from the previous iteration to the current iteration
and verify the existence of required input files and output directories. The process involves
validating the file paths and contents, leveraging Pydantic models for requirements validation.
"""

from pathlib import Path
import shutil
import os
import logging
import logging_config
from typing import Union
from evaluation_optimization.create_mapping_file import (
    load_existing_or_create_new_mapping,
)
from models.resume_job_description_io_models import Requirements
from utils.pydantic_model_loaders_from_files import (
    load_job_file_mappings_model,
)
from evaluation_optimization.evaluation_optimization_utils import check_mapping_keys
from utils.generic_utils import (
    read_from_json_file,
    save_to_json_file,
    verify_dir,
    verify_file,
)

logger = logging.getLogger(__name__)


def set_directory_paths(
    mapping_file_prev: Union[str, Path], mapping_file_curr: Union[str, Path]
) -> dict:
    """
    Set the directory paths for copying requirements based on the previous and current mapping files.

    This function loads the previous and current iteration mapping files as Pydantic models,
    validates the mapping keys (URLs), and constructs a dictionary of paths for input/output
    requirements. It checks the validity of the file paths and logs warnings for any URLs
    that are missing in the current mapping file.

    Args:
        mapping_file_prev (Union[str, Path]): Path to the mapping file for the previous iteration.
        mapping_file_curr (Union[str, Path]): Path to the mapping file for the current iteration.

    Returns:
        dict: A dictionary where each key is a URL from the mapping, and the value is another
        dictionary containing:
            - 'requirements_input': Path to the requirements file from the previous iteration.
            - 'requirements_output': Path to the requirements file for the current iteration.

        If any file is missing or an error occurs, the function logs the issue and skips
        processing for that URL.
    """
    # Convert mapping files to Path objects (if they aren't already)
    mapping_file_prev = Path(mapping_file_prev)
    mapping_file_curr = Path(mapping_file_curr)

    # Load the mapping files using the Pydantic model loader
    file_mapping_prev_model = load_job_file_mappings_model(mapping_file_prev)
    file_mapping_curr_model = load_job_file_mappings_model(mapping_file_curr)

    if file_mapping_prev_model is None or file_mapping_curr_model is None:
        logger.error(
            f"Failed to load one or both mapping files: {mapping_file_prev}, {mapping_file_curr}"
        )
        return {}

    # Extract mappings from the Pydantic models' root attribute
    file_mapping_prev = file_mapping_prev_model.root
    file_mapping_curr = file_mapping_curr_model.root

    # Initialize dictionary to hold paths
    paths_dict = {}

    # Iterate through URLs from the previous mapping file
    for url, prev_paths in file_mapping_prev.items():
        if url not in file_mapping_curr:
            logger.warning(
                f"URL {url} not found in the current iteration's mapping file."
            )
            continue

        curr_paths = file_mapping_curr[url]

        # Create the dictionary of paths for the current URL
        paths_dict[url] = {
            "requirements_input": Path(
                prev_paths.reqs
            ),  # Directly access from the Pydantic model
            "requirements_output": Path(
                curr_paths.reqs
            ),  # current iteration requirements
        }

        # Verify the file paths and log errors for missing files
        if not verify_file(paths_dict[url]["requirements_input"]):
            logger.error(f"Missing requirements file for {url}. Skipping URL.")
            continue

        # Verify the output directory exists, but not necessarily the file
        if not paths_dict[url]["requirements_output"].parent.exists():
            logger.error(
                f"Output directory for requirements file does not exist for {url}. Skipping URL."
            )
            continue

    logger.info(f"Directory path dictionary:\n{paths_dict}")
    return paths_dict


def verify_paths(
    mapping_file_prev: Union[Path, str], mapping_file_curr: Union[Path, str]
) -> bool:
    """Function to test input files exist and output directories are valid."""
    mapping_file_prev, mapping_file_curr = Path(mapping_file_prev), Path(
        mapping_file_curr
    )  # Change to Path obj if str

    # Get the directory paths
    paths_dict = set_directory_paths(mapping_file_prev, mapping_file_curr)

    # Flag to track if all verifications pass
    all_valid = True

    # Check each URL entry and verify input/output paths
    for url, paths in paths_dict.items():
        requirements_input = paths["requirements_input"]

        # Verify if input files exist
        if not requirements_input.exists():
            logger.error(
                f"Requirements input file does not exist for URL {url}: {requirements_input}"
            )
            all_valid = False  # Set to False if any file is missing
    return all_valid


def run_copying_reqs_to_next_iter_mini_pipeline(
    mapping_file_prev: Union[str, Path],
    mapping_file_curr: Union[str, Path],
) -> None:
    """
    Copy JSON files from the requirements folder in the previous iteration to the
    requirements folder in the current iteration.

    This function reads the previous iteration's requirements files and copies them
    to the corresponding directory in the current iteration. It validates the paths
    and logs any errors encountered during the process.

    Args:
        - mapping_file_prev (Union[Path, str]): Path to the previous iteration's
        mapping file.
        - mapping_file_curr (Union[Path, str]): Path to the current iteration's
        mapping file.

    Returns:
        None
    """
    # Step 0: Ensure mapping file paths are Path objects
    mapping_file_prev, mapping_file_curr = Path(mapping_file_prev), Path(
        mapping_file_curr
    )

    # Step 1: Verify paths before proceeding
    if not verify_paths(mapping_file_prev, mapping_file_curr):
        logger.error("Path verification failed, stopping execution.")
        return  # early return

    logger.info("I/O dir/file paths are correct. Proceed with processing.")

    # Step 2: Setup directory and file paths
    paths_dict = set_directory_paths(mapping_file_prev, mapping_file_curr)
    if not paths_dict:
        logger.error("Failed to set up directory paths.")
        return

    logger.info(f"paths_dict:\n{paths_dict}")

    # Step 3: Process each URL's requirements
    for url, paths in paths_dict.items():
        logger.info(f"Processing job posting from {url}")

        output_file = paths["requirements_output"]
        if output_file.exists():
            logger.info(f"Output file already exists for {url}, skipping processing.")
            continue

        try:
            # Load the input requirements
            reqs_in_file = paths["requirements_input"]
            if not reqs_in_file.exists():
                raise FileNotFoundError(f"File {reqs_in_file} not found for {url}")

            requirements_in = read_from_json_file(reqs_in_file)
            if not requirements_in:
                raise ValueError(
                    f"Requirements file {reqs_in_file} is empty or invalid for {url}"
                )

            # Use Pydantic for validation and save the validated requirements
            validated_requirements = Requirements(**requirements_in)
            save_to_json_file(validated_requirements.model_dump(), output_file)

            logger.info(f"Requirements file saved: {output_file}")

        except (FileNotFoundError, ValueError, Exception) as error:
            logger.error(f"Error processing {url}: {error}")
            continue

    logger.info("Pipeline execution completed.")
