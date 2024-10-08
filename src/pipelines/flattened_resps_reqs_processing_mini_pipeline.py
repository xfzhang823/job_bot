"""generating_flat_requirements_and_responsibilities_pipeline.py"""

import os
from pathlib import Path
from typing import Union
import json
import logging
import logging_config
from evaluation_optimization.evaluation_optimization_utils import (
    process_and_save_requirements_by_url,
    process_and_save_responsibilities_from_resume,
)
from evaluation_optimization.create_mapping_file import (
    load_existing_or_create_new_mapping,
)
from utils.generic_utils import read_from_json_file

# Setup logger
logger = logging.getLogger(__name__)


def process_files(
    file_paths_dict: dict,
    process_func: callable,
    file_type: str,
    source_json_file: Union[str, Path],
):
    """
    Process and save the flattened job files (requirements or responsibilities).

    Args:
        file_paths_dict (dict): A dictionary of URLs and their corresponding file paths.
        process_func (callable): The function to process the JSON data (either for requirements or responsibilities).
        file_type (str): The type of file being processed ('reqs_flat' or 'resps_flat').
        source_json_file (str or Path): The source JSON file to process (requirements or resume).

    Returns:
        None
    """
    for url, files in file_paths_dict.items():
        try:
            file_path = Path(files[file_type])
            if not file_path.exists():  # Process only if the file does not exist
                process_func(
                    source_json_file=source_json_file,  # Pass the correct source JSON file
                    url=url,
                    output_file=file_path,
                )
                logger.info(f"{file_type.capitalize()} file saved: {file_path}")
            else:
                logger.info(
                    f"{file_type.capitalize()} file already exists, skipping: {file_path}"
                )
        except Exception as e:
            logger.error(f"Error processing {file_type} file {file_path}: {e}")


def run_pipeline(
    mapping_file: Union[str, Path],
    job_requirements_file: Union[str, Path],
    resume_json_file: Union[str, Path],
):
    """
    Processes responsibilities and requirements based on the mapping file.

    Args:
        mapping_file (str or Path): Path to the JSON mapping file.
        job_requirements_file (str or Path): Path to the source job requirements JSON file.
        resume_json_file (str or Path): Path to the resume JSON file.

    Returns:
        None
    """
    # Convert inputs to Path objects if they are not already
    mapping_file = Path(mapping_file)
    job_requirements_file = Path(job_requirements_file)
    resume_json_file = Path(resume_json_file)

    # Debugging: checking input variables
    logger.info(f"Mapping file path: {mapping_file}")
    logger.info(f"Job requirements file path: {job_requirements_file}")
    logger.info(f"Resume JSON file path: {resume_json_file}")

    # Step 1: Read the mapping file
    try:
        file_mapping = read_from_json_file(mapping_file)
        logger.info(f"Loaded mapping file: {mapping_file}")
    except FileNotFoundError as e:
        logger.error(f"Mapping file not found: {mapping_file}")
        raise e

    # Step 2: Process and save flattened requirements
    process_files(
        file_mapping,
        lambda url, output_file: process_and_save_requirements_by_url(
            url=url,
            requirements_json_file=job_requirements_file,  # Source data
            requirements_flat_json_file=output_file,
        ),
        file_type="reqs_flat",
        source_json_file=job_requirements_file,  # Pass the correct source JSON file for requirements
    )

    # Step 3: Process and save flattened responsibilities
    process_files(
        file_mapping,
        lambda url, output_file: process_and_save_responsibilities_from_resume(
            url=url,
            resume_json_file=resume_json_file,  # Source data
            responsibilities_flat_json_file=output_file,
        ),
        file_type="resps_flat",
        source_json_file=resume_json_file,  # Pass the correct source JSON file for responsibilities
    )
