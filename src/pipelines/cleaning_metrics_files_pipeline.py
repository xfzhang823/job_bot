"""cleaning_similarity_metrics_files_pipeline.py"""

import os
import json
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Union
from utils.pydantic_model_loaders_from_files import (
    load_job_file_mappings_model,
)

# Set up logging
logger = logging.getLogger(__name__)


def run_cleaning_similarity_metrics_files_pipeline(
    mapping_file: Union[str, Path],
) -> None:
    """
    Find similarity metrics CSV files from a mapping file and clean them by removing
    empty rows.

    Args:
        mapping_file (str | Path): Path to the mapping file containing job postings
        and file paths.
    """
    logger.info(f"Start running cleaning similarity metrics files pipeline...")
    mapping_file = Path(mapping_file)

    # Load mappings
    file_mappings = load_job_file_mappings_model(mapping_file)
    if not file_mappings:
        logger.error(f"Failed to load mapping file: {mapping_file}")
        return

    # Extract similarity metrics file paths
    for job_url, file_path in file_mappings.root.items():

        # Prevents potential AttributeError in cases where sim_metrics doesn't exist in some entries.
        if not hasattr(file_path, "sim_metrics"):
            logger.warning(
                f"Skipping job URL {job_url} due to missing 'sim_metrics' attribute."
            )
            continue

        sim_metrics_file = Path(file_path.sim_metrics)

        if not sim_metrics_file.exists():
            logger.warning(f"Skipping missing file: {sim_metrics_file}")
            continue

        try:
            logging.info(
                f"Cleaning metrics file {file_path} for jobposting ur {job_url}"
            )
            # Load CSV
            df = pd.read_csv(sim_metrics_file)

            # Remove empty rows
            df_cleaned = df.dropna(how="all")

            # Save cleaned file
            df_cleaned.to_csv(sim_metrics_file, index=False)
            logger.info(f"Cleaned file saved: {sim_metrics_file}")

        except Exception as e:
            logger.error(f"Error processing {sim_metrics_file}: {e}")

    logger.info(f"Finished running cleaning similarity metrics files pipeline...")
