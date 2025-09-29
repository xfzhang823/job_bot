"""
Module: upserting_mapping_file_iter1_mini_pipeline

This module creates or updates the mapping file for iteration 1 of the pipeline.
It processes job descriptions, configures the mapping file structure, and ensures
proper paths and suffixes are applied for the iteration.
"""

from pathlib import Path
import logging
from typing import Union, Dict

# User defined
from job_bot.evaluation_optimization.create_mapping_file import (
    load_existing_or_create_new_mapping,
)
from job_bot.utils.generic_utils import read_from_json_file, save_to_json_file
from job_bot.evaluation_optimization.create_mapping_file import MappingConfig
from job_bot.models.resume_job_description_io_models import JobFileMappings

logger = logging.getLogger(__name__)


def customize_mapping_config_iter1(
    iteration: float, iteration_dir: Union[str, Path], mapping_file_name: str
) -> MappingConfig:
    """
    Customize the mapping configuration for a specific iteration.

    This function generates a 'MappingConfig' object tailored for a specific
    iteration of the pipeline by adjusting file names and directory paths
    accordingly.

    Args:
        - iteration (float): The iteration number for the current run (e.g., 0, 1, 2).
        - iteration_dir (Union[str, Path]): The directory path where the iteration files
            will be stored.
        - mapping_file_name (str): The name of the mapping file for the current iteration.

    Returns:
        MappingConfig: A customized configuration object with paths and suffixes
        specific to the provided iteration.

    Logs:
        Logs the customized configuration for the iteration using the logger.
    """
    iteration_dir = Path(iteration_dir)
    mapping_config = MappingConfig(
        mapping_file=(iteration_dir / mapping_file_name).as_posix(),
        base_output_dir=iteration_dir.as_posix(),
        reqs_dir_name="requirements",
        resps_dir_name="responsibilities",
        metrics_dir_name="similarity_metrics",
        pruned_resps_dir_name="pruned_responsibilities",
        reqs_suffix=f"_reqs_flat_iter{iteration}",
        resps_suffix=f"_resps_nested_iter{iteration}",
        metrics_suffix=f"_sim_metrics_iter{iteration}",
        pruned_resps_suffix=f"_pruned_resps_flat_iter{iteration}",
    )

    logger.info(f"Iteration {iteration} configuration:\n{mapping_config}")
    return mapping_config


def run_upserting_mapping_file_iter1_mini_pipeline(
    job_descriptions_file: Union[str, Path],
    iteration: int,
    iteration_dir: Union[str, Path],
    mapping_file_name: str,
) -> JobFileMappings:
    """
    Create or update the mapping file based on job descriptions for a specific iteration.

    This function loads job descriptions from a specified file, generates or updates a
    mapping configuration for the current iteration, and returns the updated mapping
    file content.

    Args:
        - job_descriptions_file (Union[str, Path]): Path to the JSON file containing job
            descriptions.
        - iteration (float): The iteration number for the current run (e.g., 0, 1, 2).
        - iteration_dir (Union[str, Path]): The directory path where the iteration files
            will be stored.
        - mapping_file_name (str): The name of the mapping file for the current iteration.

    Returns:
        JobFileMappings: The updated or newly created mapping file content pydantic model.

    Raises:
        ValueError: If the job descriptions file cannot be loaded or is not in
        valid JSON format.

    Logs:
        - Logs the file path of the job descriptions file.
        - Logs when the mapping file is successfully created or updated.
    """
    logger.info(f"Job descriptions file path: {job_descriptions_file}")

    job_descriptions_file = Path(job_descriptions_file)
    job_descriptions = read_from_json_file(job_descriptions_file)

    if job_descriptions is None:
        raise ValueError("Failed to load job descriptions")

    mapping_config = customize_mapping_config_iter1(
        iteration=iteration,
        iteration_dir=iteration_dir,
        mapping_file_name=mapping_file_name,
    )
    file_mapping = load_existing_or_create_new_mapping(job_descriptions, mapping_config)

    logger.info("Mapping file created or updated successfully.")
    logger.info(file_mapping)
    return file_mapping
