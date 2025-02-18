"""
#Todo: WIP; to be used later
"""

import pandas as pd
import logging
import json
from typing import List
from evaluation_optimization.resume_pruner import ResponsibilitiesPruner
from utils.generic_utils import save_to_json_file, read_from_json_file

logger = logging.getLogger(__name__)


def load_mapping_file(mapping_file: str) -> dict:
    """
    Load the mapping file containing paths for requirements, responsibilities, and metrics.

    The mapping file defines the paths for each job posting's responsibility, requirement, and
    similarity metrics files.

    Args:
        mapping_file (str): Path to the JSON file containing mappings of job posting URLs
                            to relevant file paths.

    Returns:
        dict: A dictionary of mappings loaded from the JSON file.
    """
    mapping = read_from_json_file(mapping_file)
    return mapping


def prune_responsibilities_for_jobposting(
    similarity_metrics_file: str,
    pruning_method: str = "elbow",
    group_by_responsibility: bool = False,
    **method_params: dict,
) -> dict:
    """
    Prune responsibilities for a specific job posting based on similarity metrics.

    Args:
        similarity_metrics_file (str): Path to the CSV file containing similarity metrics.
        pruning_method (str): The pruning method to use ('elbow', 'manual', or 'threshold').
        group_by_responsibility (bool): Whether to group by responsibility_key.
        **method_params: Additional parameters specific to the pruning method.

    Returns:
        dict: A dictionary containing pruned responsibilities in JSON format.
    """
    # Load the DataFrame from the CSV file
    df = pd.read_csv(similarity_metrics_file)

    # Log the original columns for tracking/debugging purposes
    logger.info(f"Original columns: {df.columns}")

    # Initialize the ResponsibilitiesPruner with the loaded DataFrame
    pruner = ResponsibilitiesPruner(df)

    # # Optionally, plot the score distribution
    # pruner.plot_score_distribution(save_path="score_distribution.png")

    # Call run_pruning_process with unpacked method_params
    if method_params is None:
        method_params = {}

    # Run the pruning process using the specified method and parameters
    pruned_responsibilities = pruner.run_pruning_process(
        method=pruning_method,
        group_by_responsibility=group_by_responsibility,
        **method_params,
    )

    # Optionally, log the pruning stats
    pruned_df = pruner.df  # The pruned DataFrame after pruning
    logger.info(f"Pruning stats: {pruner.get_pruning_stats(df, pruned_df)}")

    return pruned_responsibilities


def run_resume_pruning_pipeline(
    mapping_file: str, pruning_method: str, **method_params
) -> None:
    """
    Run the full pruning pipeline for multiple job postings based on the paths defined in
    the mapping file.

    Args:
        -mapping_file (str): Path to the mapping file containing job posting URLs and
        associated file paths.
        -pruning_method (str): The pruning method to use ('elbow', 'manual', 'threshold').
        **method_params: Additional parameters specific to the pruning method.

    Returns: None
        The function does not return anything. It saves the pruned results to
        specified file paths.
    """
    # Load the job posting mappings from the mapping file
    mapping = load_mapping_file(mapping_file)

    # Iterate through each job posting and its associated file paths
    for job_url, paths in mapping.items():
        try:
            # Extract the path for similarity metrics and the output file for
            # pruned responsibilities
            similarity_metrics_file = paths["sim_metrics"]
            pruned_resps_file = paths["pruned_responsibilities_flat"]

            # Run the pruning process for each job posting based on its
            # similarity metrics
            pruned_responsibilities = prune_responsibilities_for_jobposting(
                similarity_metrics_file=similarity_metrics_file,
                pruning_method=pruning_method,
                **method_params,
            )

            # Debugging: print the pruned responsibilities for verification
            logger.debug(
                f"Pruned responsibilities for {job_url}: {pruned_responsibilities}"
            )

            # Save the pruned responsibilities to the specified output file
            save_to_json_file(pruned_responsibilities, pruned_resps_file)

        except Exception as e:
            # Log any errors encountered during the pruning process for this job posting
            logger.error(f"Error processing job posting {job_url}: {str(e)}")
