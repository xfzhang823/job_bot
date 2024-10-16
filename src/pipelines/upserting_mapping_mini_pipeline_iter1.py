"""upserting_mapping_file.py"""

import os
import json
from pathlib import Path
import logging
from typing import Union, Dict
from evaluation_optimization.create_mapping_file import (
    load_existing_or_create_new_mapping,
)
from utils.generic_utils import read_from_json_file, save_to_json_file
from evaluation_optimization.create_mapping_file import MappingConfig


logger = logging.getLogger(__name__)


# Set up config for the mapping file using dataclass MappingConfig
def customize_mapping_config(
    iteration: float, iteration_dir: Union[str, Path], mapping_file_name: str
):
    """
    Customize the mapping configuration for a specific iteration.

    This function generates a `MappingConfig` object tailored for a specific iteration
    of the pipeline by adjusting file names and directory paths accordingly.

    Args:
        iteration (float): The iteration number for the current run (e.g., 0, 1, 2).
        iteration_dir (Union[str, Path]): The directory path where the iteration files
            will be stored.
        mapping_file_name (str): The name of the mapping file for the current iteration.

    Returns:
        MappingConfig: A customized configuration object with paths and suffixes
        specific to the provided iteration.

    Example:
        customize_mapping_config(
            iteration=1,
            iteration_dir="C:/project/iteration_1",
            mapping_file_name="url_to_file_mapping.json"
        )

    Logs:
        Logs the customized configuration for the iteration using the logger.
    """
    # Ensure it's a Path object and resolve for platform correctness
    iteration_dir = Path(iteration_dir).resolve()
    mapping_config = MappingConfig(
        mapping_file=(
            iteration_dir / mapping_file_name
        ).as_posix(),  # Ensure forward slashes
        base_output_dir=iteration_dir.as_posix(),
        reqs_dir_name="requirements",
        resps_dir_name="responsibilities",
        metrics_dir_name="similarity_metrics",
        pruned_resps_dir_name="pruned_responsibilities",
        reqs_suffix=f"_reqs_iter{iteration}",
        resps_suffix=f"_resps_iter{iteration}",  # responsibilities files are no longer flat
        metrics_suffix=f"_sim_metrics_iter{iteration}",
        pruned_resps_suffix=f"_pruned_resps_iter{iteration}",  # responsibilities files are no longer flat
    )

    logger.info(f"Iteration {iteration} configuration:\n{mapping_config}")
    return mapping_config


def run_pipeline(
    job_descriptions_file: Union[str, Path],
    iteration: float,
    iteration_dir: Union[str, Path],
    mapping_file_name: str,
) -> Dict:
    """
    Create or update the mapping file based on job descriptions for a specific iteration.

    This function loads job descriptions from a specified file, generates or updates a
    mapping configuration for the current iteration, and returns the updated mapping
    file content.

    Args:
        job_descriptions_file (Union[str, Path]): Path to the JSON file containing job
            descriptions.
        iteration (float): The iteration number for the current run (e.g., 0, 1, 2).
        iteration_dir (Union[str, Path]): The directory path where the iteration files
            will be stored.
        mapping_file_name (str): The name of the mapping file for the current iteration.

    Returns:
        Dict: The updated or newly created mapping file content.

    Raises:
        ValueError: If the job descriptions file cannot be loaded or is not in valid JSON format.

    Example:
        run_pipeline(
            job_descriptions_file="C:/project/input/job_descriptions.json",
            iteration=1,
            iteration_dir="C:/project/iteration_1",
            mapping_file_name="url_to_file_mapping.json"
        )

    Logs:
        - Logs the file path of the job descriptions file.
        - Logs when the mapping file is successfully created or updated.
    """
    logger.info(f"Job descriptions file path: {job_descriptions_file}")

    # Ensure job description file is Path object
    job_descriptions_file = Path(job_descriptions_file)

    # Load job descriptions -> dict
    job_descriptions = read_from_json_file(job_descriptions_file)
    if job_descriptions is None:
        raise ValueError("Failed to load job descriptions")

    # Create or update mapping file
    mapping_config = customize_mapping_config(
        iteration=iteration,
        iteration_dir=iteration_dir,
        mapping_file_name=mapping_file_name,
    )
    file_mapping = load_existing_or_create_new_mapping(job_descriptions, mapping_config)

    logger.info("Mapping file created or updated successfully.")
    logger.info(file_mapping)
    return file_mapping
