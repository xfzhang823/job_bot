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


def run_copying_resps_to_pruned_resps_dir_mini_pipeline(mapping_file: Union[Path, str]):
    """
    Copy json files from responsibilities folder to pruned_responsibilities folder.

    Args:
        Mapping file.

    Returns:
        None
    """
    mapping_file = Path(mapping_file)  # Change to Path obj if str.

    mappings = read_from_json_file(
        mapping_file
    )  # No need for error handling b/c the function has built in error handling already
    if not mappings:
        raise ValueError(
            f"The file {mapping_file} is empty or contains invalid content."
        )

    for url, job_data in mappings.items():
        resps_path = job_data.get("resps")
        pruned_resps_path = job_data.get("prune_resps")

        if resps_path and pruned_resps_path:
            try:
                # Ensure the output directory exists for pruned responsibilities
                output_dir = os.path.dirname(pruned_resps_path)
                os.makedirs(output_dir, exist_ok=True)

                # Read from responsibilities path
                resps_content = read_from_json_file(resps_path)
                if not resps_content:
                    raise ValueError(
                        f"The file {resps_path} is empty or contains invalid content."
                    )
                save_to_json_file(resps_content, pruned_resps_path)

                logger.info(
                    f"Successfully copied from {resps_path} to {pruned_resps_path}"
                )

            except FileNotFoundError:
                print(f"File not found: {resps_path}")
            except Exception as e:
                print(f"An error occurred for {url}: {e}")
        else:
            print(f"Missing paths for {url}")
