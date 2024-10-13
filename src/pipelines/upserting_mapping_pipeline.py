"""upserting_mapping_file.py"""

import os
import json
from pathlib import Path
import logging
from typing import Union
from evaluation_optimization.create_mapping_file import (
    load_existing_or_create_new_mapping,
)
from utils.generic_utils import read_from_json_file, save_to_json_file

logger = logging.getLogger(__name__)


def run_pipe_line(
    job_descriptions_file: Union[str, Path],
    flat_reqs_output_files_dir: Union[str, Path],
    flat_resps_output_files_dir: Union[str, Path],
    sim_metrics_output_files_dir: Union[str, Path],
    pruned_resps_output_files_dir: Union[str, Path],
    mapping_file_dir: Union[str, Path],
    mapping_file_name: Union[str, Path],
):
    """
    Create or update the mapping file based on job descriptions.

    Args:
        -job_descriptions_file (str or Path): Path to the JSON file containing job descriptions.
        -flat_reqs_output_files_dir (str or Path): Directory where flattened requirements files
        are stored.
        -flat_resps_output_files_dir (str or Path): Directory where flattened responsibilities files
        are stored.
        -sim_metrics_output_files_dir (str or Path): Directory where similarity metrics files
        are stored.
        -pruned_resps_output_files_dir (str or Path): Direcotry where pruned responsibilities
        files (very low scored responsibilities eliminated) are stored
        -mapping_file_dir (str or Path): Directory where to save the mapping file containing URL
        and different file names for the iteration.
        -mapping_file_name (str or Path): File name of the mapping file containing URL and
        different file names for the iteration.

    Returns:
        dict: Updated or newly created mapping file.
    """
    # Convert inputs to Path objects if they are not already
    job_descriptions_file = Path(job_descriptions_file)
    flat_reqs_output_files_dir = Path(flat_reqs_output_files_dir)
    flat_resps_output_files_dir = Path(flat_resps_output_files_dir)
    sim_metrics_output_files_dir = Path(sim_metrics_output_files_dir)
    pruned_resps_output_files_dir = Path(pruned_resps_output_files_dir)
    mapping_file_dir = Path(mapping_file_dir)

    # Ensure the directories exist
    mapping_file_dir.mkdir(parents=True, exist_ok=True)
    flat_reqs_output_files_dir.mkdir(parents=True, exist_ok=True)
    flat_resps_output_files_dir.mkdir(parents=True, exist_ok=True)
    sim_metrics_output_files_dir.mkdir(parents=True, exist_ok=True)
    pruned_resps_output_files_dir.mkdir(parents=True, exist_ok=True)

    # Create the full mapping file path
    mapping_file = mapping_file_dir / mapping_file_name

    logger.info(f"Job descriptions file path: {job_descriptions_file}")
    logger.info(f"Mapping file path: {mapping_file}")

    # Load job descriptions -> dict
    job_descriptions = read_from_json_file(job_descriptions_file)

    # Create or update mapping file
    file_mapping = load_existing_or_create_new_mapping(
        mapping_file,
        job_descriptions,
        flat_reqs_output_files_dir,
        flat_resps_output_files_dir,
        sim_metrics_output_files_dir,
        pruned_resps_output_files_dir,
    )

    logger.info("Mapping file created or updated successfully.")
    logger.info(file_mapping)
    return file_mapping
