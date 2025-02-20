import os
import json
import pandas as pd
import logging
from pathlib import Path
from typing import Union
from evaluation_optimization.create_mapping_file import load_mappings_model_from_json
from project_config import ITERATE_0_ANTHROPIC_DIR, mapping_file_name

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_urls_to_similarity_metrics(mapping_file: Union[str, Path]) -> None:
    """
    Adds job posting URLs to similarity metrics CSV files by looking up the mapping file.

    Args:
        mapping_file (str | Path): Path to the mapping file containing job postings and file paths.
    """
    mapping_file = Path(mapping_file)

    # Load mappings
    file_mappings = load_mappings_model_from_json(mapping_file)
    if not file_mappings:
        logger.error(f"Failed to load mapping file: {mapping_file}")
        return

    # Iterate through job URLs and corresponding similarity metrics files
    for job_url, file_paths in file_mappings.root.items():

        # Ensure 'sim_metrics' exists
        if not hasattr(file_paths, "sim_metrics"):
            logger.warning(
                f"Skipping job URL {job_url} due to missing 'sim_metrics' attribute."
            )
            continue

        sim_metrics_file = Path(file_paths.sim_metrics)

        if not sim_metrics_file.exists():
            logger.warning(f"Skipping missing file: {sim_metrics_file}")
            continue

        try:
            # Load CSV
            df = pd.read_csv(sim_metrics_file)

            # Add job URL as the first column if not already present
            if "job_posting_url" not in df.columns:
                df.insert(0, "job_posting_url", str(job_url))
                df.to_csv(sim_metrics_file, index=False)
                logger.info(f"Added URL to {sim_metrics_file}")
            else:
                logger.info(f"URL already exists in {sim_metrics_file}, skipping.")

        except Exception as e:
            logger.error(f"Error processing {sim_metrics_file}: {e}")


# Example usage
if __name__ == "__main__":
    mapping_file_path = ITERATE_0_ANTHROPIC_DIR / mapping_file_name
    add_urls_to_similarity_metrics(mapping_file_path)
