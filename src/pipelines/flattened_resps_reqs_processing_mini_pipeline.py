"""generating_flat_requirements_and_responsibilities_pipeline.py"""

from pathlib import Path
from typing import Union, Callable
import logging
from evaluation_optimization.evaluation_optimization_utils import (
    process_and_save_requirements_by_url,
    process_and_save_responsibilities_from_resume,
)
from utils.generic_utils import read_from_json_file

# Setup logger
logger = logging.getLogger(__name__)


def process_files(
    file_paths_dict: dict,
    process_func: Callable,
    file_type: str,
):
    """
    Process and save the flattened job files (requirements or responsibilities).

    Args:
        - file_paths_dict (dict): A dictionary of URLs and their corresponding file paths.
        - process_func (callable): The function to process the JSON data (either for
        requirements or responsibilities).
        - file_type (str): The type of file being processed ('reqs_flat' or 'resps_flat').

    Returns:
        None
    """
    logger.debug(f"Checking file_paths_dict: {file_paths_dict}")

    for url, files in file_paths_dict.items():
        try:
            # Check if file_type exists
            if file_type not in files:
                logger.error(f"Key '{file_type}' not found in files for URL: {url}")
                logger.debug(f"Available keys: {list(files.keys())}")  # Log actual keys
                continue  # Skip this entry

            file_path = Path(files[file_type])

            if not file_path.exists():  # Process only if the file does not exist
                process_func(
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


def run_flatten_resps_reqs_processing_mini_pipeline(
    mapping_file: Union[str, Path],
    job_requirements_file: Union[str, Path],
    resume_json_file: Union[str, Path],
):
    """
    *Async version
    Processes and saves flattened job requirements and responsibilities based on a mapping file.

    Workflow:
    1. Loads the mapping file to retrieve URLs and their associated file paths.
    2. Processes job requirements:
       - For each URL, checks if a flattened requirements file already exists.
       - If not, processes the job requirements data from `job_requirements_file`
         and saves the flattened output.
    3. Processes job responsibilities:
       - For each URL, checks if a flattened responsibilities file already exists.
       - If not, processes the responsibilities data from `resume_json_file`
         and saves the flattened output.

    Args:
        - mapping_file (str or Path): Path to the JSON mapping file.
        - job_requirements_file (str or Path): Path to the source job requirements JSON file.
        - resume_json_file (str or Path): Path to the resume JSON file
        (source for responsibilities).

    Returns:
        None
    """
    # Convert inputs to Path objects if they are not already
    mapping_file, job_requirements_file, resume_json_file = (
        Path(mapping_file),
        Path(job_requirements_file),
        Path(resume_json_file),
    )

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

    # * Step 2: Process and save flattened requirements
    logger.debug(f"Processing requirements for: {list(file_mapping.keys())}")
    process_files(
        file_paths_dict=file_mapping,
        process_func=lambda url, output_file: process_and_save_requirements_by_url(
            requirements_json_file=job_requirements_file,  # Source data file
            requirements_flat_json_file=output_file,
            url=url,
        ),
        file_type="reqs",
    )

    # * Step 3: Process and save flattened responsibilities
    logger.debug(f"Processing responsibilities for: {list(file_mapping.keys())}")
    process_files(
        file_paths_dict=file_mapping,
        process_func=lambda url, output_file: process_and_save_responsibilities_from_resume(
            resume_json_file=resume_json_file,  # Source data
            responsibilities_flat_json_file=output_file,
            url=url,
        ),
        file_type="resps",
    )
