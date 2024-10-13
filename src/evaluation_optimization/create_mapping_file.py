"""create_file_mapping.py"""

import os
import json
from pathlib import Path
import logging
import logging_config
from utils.generic_utils import save_to_json_file, read_from_json_file
from evaluation_optimization.evaluation_optimization_utils import create_file_name

logger = logging.getLogger(__name__)


def load_existing_or_create_new_mapping(
    mapping_file: str,
    job_descriptions: dict,
    reqs_output_dir: str,
    resps_output_dir: str,
    metrics_output_dir: str,
    pruned_resps_output_dir: str,
) -> dict:
    """
    Load the existing mapping file:
    - if it exists, compare URLs with job descriptions, and update if necessary.
    - if no mapping file exists, create a new mapping and save it.

    Args:
        -mapping_file (str or Path): Path to the mapping file.
        -job_descriptions (dict): Job descriptions that contain information for generating
        new mappings.
        -reqs_output_dir (str or Path): Directory where requirements files will be saved.
        -resps_output_dir (str or Path): Directory where responsibilities files will be saved.
        -metrics_output_dir (str or Path): Directory where similarity metrics files will be saved.
        -pruned_resps_output_dir (str or Path): Directory where pruned responsibilities files
        will be saved.

    Returns:
        dict: The updated or newly created mapping file.

    Raises:
        ValueError: If the mapping file contains more URLs than the job descriptions.

    Example:
        >>> job_descriptions = {
                "https://www.amazon.jobs/...": {
                    "company": "Amazon",
                    "job_title": "Product Manager, Artificial General Intelligence - Data Services"
                },
                "https://jobs.microsoft.com/...": {
                    "company": "Microsoft",
                    "job_title": "Head of Partner Intelligence and Strategy"
                }
            }
        >>> reqs_output_dir = "path/to/requirements_flat/"
        >>> resps_output_dir = "path/to/responsibilities_flat/"
        >>> metrics_output_dir = "path/to/similarity_metrics/"
        >>> pruned_resps_output_dir = "path/to/pruned_responsibilities_flat/"
        >>> mapping_file_path = "path/to/mapping_file.json"
        >>> mapping = load_existing_or_create_new_mapping(
                mapping_file_path,
                job_descriptions,
                reqs_output_dir,
                resps_output_dir
            )
        >>> print(mapping)
        {
            "https://www.amazon.jobs/...": {
                "reqs_flat": "path/to/requirements_flat/Amazon_Product_Manager__Artificial_General_Intelligence_-_Data_Services_reqs_flat.json",
                "resps_flat": "path/to/responsibilities_flat/Amazon_Product_Manager__Artificial_General_Intelligence_-_Data_Services_resps_flat.json",
                "sim_metrics": "path/to/requirements_flat/Amazon_Product_Manager__Artificial_General_Intelligence_-_Data_Services_sim_metrics.csv",
                "pruned_resps_flat": "path/to/pruned_responsibilities_flat/Amazon_Product_Manager__Artificial_General_Intelligence_-_Data_Services_resps_flat.json"
            },
            "https://jobs.microsoft.com/...": {
                "reqs_flat": "path/to/requirements_flat/Microsoft_Head_of_Partner_Intelligence_and_Strategy_reqs_flat.json",
                "resps_flat": "path/to/responsibilities_flat/Microsoft_Head_of_Partner_Intelligence_and_Strategy_resps_flat.json",
                "sim_metrics": "path/to/requirements_flat/Microsoft_Head_of_Partner_Intelligence_and_Strategy_sim_metrics.csv",
                "pruned_resps_flat": "path/to/requirempruned_responsibilities_flat/Microsoft_Head_of_Partner_Intelligence_and_Strategy_resps_flat.json"
            }
        }
    """

    def create_mapping_entry(company, job_title):
        """Helper function to create file paths using the create_file_name function."""
        return {
            "reqs_flat": str(
                Path(reqs_output_dir)  # assigns to requirements directory
                / create_file_name(company, job_title, "reqs_flat")
            ),
            "resps_flat": str(
                Path(resps_output_dir)  # assigns to responsibilities directory
                / create_file_name(company, job_title, "resps_flat")
            ),
            "sim_metrics": str(
                Path(metrics_output_dir)
                / create_file_name(company, job_title, "sim_metrics", ext="csv")
            ),  # set ext to "csv" b/c create_file_name function defaults to JSON for file type
            "pruned_resps_flat": str(
                Path(
                    pruned_resps_output_dir
                )  # assigns to pruned responsibilities directory
                / create_file_name(company, job_title, "resps_flat")
            ),
        }

    job_urls = set(job_descriptions.keys())

    if os.path.exists(mapping_file):
        # Load existing mapping
        existing_mapping = read_from_json_file(mapping_file)
        logger.info(f"Loading existing mapping file: {mapping_file}")

        mapped_urls = set(existing_mapping.keys())
        # No need to list(set(..)) b/c "==" works on sets: they are designed for set operations
        # such as comparing for equality or finding differences.

        # Check for discrepancies
        if mapped_urls == job_urls:
            logger.info("The mapping file is up-to-date. No new URLs to add.")
            return existing_mapping
        elif mapped_urls > job_urls:
            logger.error(
                "The mapping file contains more URLs than the job descriptions. Exiting."
            )
            raise ValueError(
                "The mapping file contains URLs that do not exist in the job descriptions."
            )
        else:
            logger.info("The mapping file has fewer URLs. Adding new entries...")
            missing_urls = job_urls - mapped_urls

            # Add missing URLs to the mapping
            for url in missing_urls:
                company = job_descriptions[url].get("company")
                job_title = job_descriptions[url].get("job_title")
                existing_mapping[url] = create_mapping_entry(company, job_title)

            logger.info(f"Added {len(missing_urls)} new entries to the mapping file.")
            with open(mapping_file, "w") as f:
                json.dump(existing_mapping, f, indent=4)
                logger.info(f"Updated mapping file saved to {mapping_file}")

            return existing_mapping

    else:
        logger.info(
            f"No existing mapping file found at {mapping_file}. Creating a new one."
        )
        new_mapping = {}

        # Create a new mapping for all job descriptions
        for url, info in job_descriptions.items():
            company = info.get("company")
            job_title = info.get("job_title")
            new_mapping[url] = create_mapping_entry(company, job_title)

        # Save the newly created mapping file
        save_to_json_file(new_mapping, mapping_file)
        logger.info(f"New mapping file saved to {mapping_file}")

        return new_mapping
