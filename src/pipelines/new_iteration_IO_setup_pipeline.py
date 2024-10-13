"""
Filename: coopy_requirements_mini_pipeline.py

Set up the input/output files for iteration 1
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
from utils.generic_utils import read_from_json_file, save_to_json_file

logger = logging.getLogger(__name__)


def copy_files(src_folder: str, dest_folder: str, src_file: str = None):
    """
    Copies a single file or all files from the source folder to the destination folder.

    Args:
        * src_folder (str): The path to the source folder where files are located.
        * dest_folder (str): The path to the destination folder where files will be copied.
        * src_file (str, optional):
            - The path to a specific file (can be a full path or just the file name).
            - If None, all files in the source folder will be copied.

    Raises:
        FileNotFoundError: If the source file or folder does not exist.
        Exception: For any general errors during file copying.

    Returns:
        None

    Example:
        # To copy a specific file (just the file name):
        copy_files('/path/to/source/folder', '/path/to/destination/folder', \
            'example.txt')

        # To copy a specific file (using the full path):
        copy_files('/path/to/source/folder', '/path/to/destination/folder', \
            '/path/to/source/folder/example.txt')

        # To copy all files from the source folder:
        copy_files('/path/to/source/folder', '/path/to/destination/folder')
    """
    # Ensure destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Copy a single file if src_file is provided
    if src_file:
        # Check if src_file is an absolute path, if not, combine it with src_folder
        if not os.path.isabs(src_file):
            full_src_file_path = os.path.join(src_folder, src_file)
        else:
            full_src_file_path = src_file

        # Ensure the file is actually present
        if os.path.isfile(full_src_file_path):
            dest_file_path = os.path.join(dest_folder, os.path.basename(src_file))
            try:
                shutil.copy2(full_src_file_path, dest_file_path)
                print(f"Copied: {full_src_file_path} to {dest_file_path}")
            except Exception as e:
                print(f"Error occurred while copying file {full_src_file_path}: {e}")
        else:
            print(f"File '{src_file}' not found at location: {full_src_file_path}")

    # Copy all files if src_file is None
    else:
        for file_name in os.listdir(src_folder):
            full_src_file_path = os.path.join(src_folder, file_name)
            dest_file_path = os.path.join(dest_folder, file_name)

            # Only copy if it's a file (not a subdirectory)
            if os.path.isfile(full_src_file_path):
                try:
                    shutil.copy2(full_src_file_path, dest_file_path)
                    print(f"Copied: {file_name} to {dest_folder}")
                except Exception as e:
                    print(
                        f"Error occurred while copying file {full_src_file_path}: {e}"
                    )


def run_pipeline(
    mapping_file: str,
    previous_iter_dir: str,
    current_iter_dir: str,
    job_descriptions_file: str,
    previous_reqs_output_dir: str,
    reqs_output_dir: str,
    dir: str,
    reqs_out_dir: str,
):




def run_pipe_line(
    previous_flat_reqs_output_files_dir: Union[str, Path],
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
    previous_flat_reqs_output_files_dir = Path(previous_flat_reqs_output_files_dir)
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

    # Step 1: Copy requirements flat files (redundant but keep the data i/o structure
    # consistent across all iteration)
    copy_files(src_folder=previous_flat_reqs_output_files_dir, dest_folder=flat_reqs_output_files_dir)
    
    # Step 2: Create mapping file

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
