"""create_file_mapping.py"""

import os
import json
from pathlib import Path
import logging
import logging_config
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, cast
from pydantic import ValidationError, HttpUrl
from models.resume_job_description_io_models import JobFileMappings
from utils.generic_utils import (
    save_to_json_file,
    read_from_json_file,
    get_company_and_job_title,
)
from evaluation_optimization.evaluation_optimization_utils import create_file_name
from models.resume_job_description_io_models import JobFileMappings, JobFilePaths


logger = logging.getLogger(__name__)


@dataclass
class MappingConfig:
    """
    Exmaple config:

    mapping_config = MappingConfig(
        mapping_file="path/to/mapping_file.json",
        base_output_dir=".",
        reqs_dir_name="custom_requirements",
        resps_dir_name="custom_responsibilities",
        metrics_dir_name="custom_similarity_metrics",
        pruned_resps_dir_name="custom_pruned_responsibilities",
        reqs_suffix="_custom_reqs",
        resps_suffix="_custom_resps",
        metrics_suffix="_custom_sim_metrics",
        pruned_resps_suffix="_custom_pruned_resps",
        )
    """

    mapping_file: str
    base_output_dir: str = "."
    reqs_dir_name: str = "requirements"
    resps_dir_name: str = "responsibilities"
    metrics_dir_name: str = "similarity_metrics"
    pruned_resps_dir_name: str = "pruned_responsibilities"
    reqs_suffix: str = "reqs"
    resps_suffix: str = "resps"
    metrics_suffix: str = "sim_metrics"
    pruned_resps_suffix: str = "pruned_resps"
    metrics_ext: str = "csv"


def create_mapping_entry(
    config: MappingConfig, company: str, job_title: str, job_url: str
) -> JobFilePaths:
    """
    Create a mapping entry for a job description with Pydantic validation.

    Args:
        config (MappingConfig): Configuration for mapping file.
        company (str): Company name.
        job_title (str): Job title.
        job_url (str): Job URL (for logging)

    Returns:
        JobFilePaths: Validated mapping entry (Pydantic object).

    Raises:
        ValueError: If filename generation fails or validation fails.
    """

    # Defining file dir structure
    base_dir = Path(config.base_output_dir)
    reqs_dir = (base_dir) / config.reqs_dir_name
    resps_dir = (base_dir) / config.resps_dir_name
    metrics_dir = (base_dir) / config.metrics_dir_name
    pruned_resps_dir = (base_dir) / config.pruned_resps_dir_name

    # Create directories if they don't exist
    for directory in [reqs_dir, resps_dir, metrics_dir, pruned_resps_dir]:
        os.makedirs(
            directory, exist_ok=True
        )  # This will create the directory if it doesn't exist

    # Handle None for file names -> to create_file_name() (add new files)

    # Check and log: if company and job_title are not None before calling create_file_name
    if company is None or job_title is None:
        logger.error(
            f"Missing required values: company={company}, job_title={job_title}, job_url={job_url}"
        )
        raise ValueError(
            f"Missing required values for job: URL={job_url}, company={company}, job_title={job_title}"
        )

    reqs_file = create_file_name(company, job_title, config.reqs_suffix)
    resps_file = create_file_name(company, job_title, config.resps_suffix)
    metrics_file = create_file_name(
        company, job_title, config.metrics_suffix, ext=config.metrics_ext
    )
    pruned_resps_file = create_file_name(company, job_title, config.pruned_resps_suffix)

    # If any of the file names are None, handle that scenario
    if not reqs_file or not resps_file or not metrics_file or not pruned_resps_file:
        logger.error("File name generation failed, received None.")
        raise ValueError("File name generation failed, received None.")

    try:
        # Create a validated JobFilePaths Pydantic model
        job_file_paths_model = JobFilePaths(
            reqs=reqs_dir / reqs_file,
            resps=resps_dir / resps_file,
            sim_metrics=metrics_dir / metrics_file,
            pruned_resps=pruned_resps_dir / pruned_resps_file,
        )

        logger.debug(
            f"Creating mapping entry for {company}, {job_title}: {job_file_paths_model}"
        )

    except ValidationError as ve:
        logger.error(
            f"Pydantic validation error for entry {job_file_paths_model}: {ve}"
        )
        raise ValueError(f"Pydantic validation error: {ve}")

    return job_file_paths_model  # Return the validated JobFilePaths instance


def load_mappings_model_from_json(
    mapping_file: Union[str, Path]
) -> Optional[JobFileMappings]:
    """
    Load job file mappings from a JSON file using the JobFileMappings model.

    Args:
        mapping_file (str | Path): Path to the JSON mapping file.

    Returns:
        Optional[JobFileMappings]: Job file mappings model or None if validation fails.

    Logs:
        - Information about loading and validation success.
        - Errors encountered during validation or file processing.
    """
    try:
        mapping_file = Path(mapping_file)  # Change to Path obj. if str
        # Read the JSON file without specifying a key to get the entire data
        file_mapping = read_from_json_file(mapping_file, key=None)

        # Ensure the data is a dictionary
        if not isinstance(file_mapping, dict):
            logger.error(
                f"Mapping file {mapping_file} does not contain a valid JSON object."
            )
            return None

        # Initialize the Pydantic model with the entire mapping
        job_file_mappings_model = JobFileMappings.model_validate(file_mapping)

        logger.info(f"Loaded and validated mapping file from {mapping_file}")
        return job_file_mappings_model

    except ValidationError as e:
        logger.error(f"Validation error in mapping file {mapping_file}: {e}")
        return None
    except FileNotFoundError as e:
        logger.error(f"Mapping file not found: {mapping_file}. Error: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding failed for file {mapping_file}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading mapping file {mapping_file}: {e}")
        return None


def load_existing_or_create_new_mapping(
    job_descriptions: dict,
    config: MappingConfig,
) -> JobFileMappings:
    """
    Load the existing mapping file, validate it, and update it with new entries if necessary.
    If the mapping file does not exist, create a new one.

    Each iteration has a mapping file - it serves as the ultimate reference file for
    I/O (input/output) folders and files during each responsibilities to requirements
    matching/comparison, modification & pruning.

    Args:
        -job_descriptions (dict): Job descriptions is the source JSON file that
        contains job posting information (keys are posting urls).
        -config (dataclass): configuration of folder and file names.

    Returns:
        JobFileMappings: The updated or newly created mapping file, returned as
        a Pydantic model.

        However, the function is mainly used to create or update the mapping file in
        each iteration folder.

    Raises:
        ValueError: If the mapping file contains more URLs than the job descriptions.

    Example:
        >>> job_descriptions = {
                "https://www.amazon.jobs/...": {
                    "company": "Amazon",
                    "job_title": "Product Manager, Artificial General Intelligence - Data Services",
                    ...
                },
                "https://jobs.microsoft.com/...": {
                    "company": "Microsoft",
                    "job_title": "Head of Partner Intelligence and Strategy",
                    ...
                }
            }
        >>> mapping_config = MappingConfig(
                mapping_file=str(ITERATE_0_DIR / mapping_file_name),
                # base_output_dir=".",
                reqs_dir_name="requirements",
                resps_dir_name="responsibilities",
                metrics_dir_name="similarity_metrics",
                pruned_resps_dir_name="pruned_responsibilities",
                reqs_suffix="_reqs_iter0",
                resps_suffix="_resps_iter0",
                metrics_suffix="_sim_metrics_iter0",
                pruned_resps_suffix="_pruned_resps_iter0",
            )
        >>> mapping = load_existing_or_create_new_mapping(job_descriptions)
        >>> print(mapping)
        {
            "https://www.amazon.jobs/...": {
                "reqs": "path/to/requirements/Amazon_Product_Manager__Artificial_General_Intelligence_-_Data_Services_reqs_iter0.json",
                "resps": "path/to/responsibilities/Amazon_Product_Manager__Artificial_General_Intelligence_-_Data_Services_resps_iter0.json",
                "sim_metrics": "path/to/requirements_flat/Amazon_Product_Manager__Artificial_General_Intelligence_-_Data_Services_sim_metrics_iter0.csv",
                "pruned_resps": "path/to/pruned_responsibilities_flat/Amazon_Product_Manager__Artificial_General_Intelligence_-_Data_Services_pruned_resps_iter0.json"
            },
            "https://jobs.microsoft.com/...": {
                "reqs_flat": "path/to/requirements_flat/Microsoft_Head_of_Partner_Intelligence_and_Strategy_reqs_iter0.json",
                "resps_flat": "path/to/responsibilities_flat/Microsoft_Head_of_Partner_Intelligence_and_Strategy_resps_iter0.json",
                "sim_metrics": "path/to/requirements_flat/Microsoft_Head_of_Partner_Intelligence_and_Strategy_sim_metrics_iter0.csv",
                "pruned_resps_flat": "path/to/requirempruned_responsibilities_flat/Microsoft_Head_of_Partner_Intelligence_and_Strategy_resps_iter0.json"
            }
        }
    """

    mapping_file = Path(config.mapping_file)
    job_urls_set = set(job_descriptions.keys())

    if mapping_file.exists():
        # Load existing mapping
        file_mappings_model = load_mappings_model_from_json(mapping_file)
        if not file_mappings_model:
            raise ValueError(
                "❌ Failed to load and validate the existing mapping file."
            )
        logger.info(f"✅ Existing mapping file '{mapping_file}' loaded successfully.")

        # Convert mapped URLs to set
        mapped_urls: list[str] = [str(url) for url in file_mappings_model.root.keys()]
        mapped_urls_set = set(mapped_urls)

        logger.info(f"🔍 Comparing mapped URLs with job descriptions...")

        if mapped_urls_set == job_urls_set:
            logger.info("✅ The mapping file is up-to-date. No new URLs to add.")
            return file_mappings_model

        elif mapped_urls_set > job_urls_set:
            logger.error(
                "❌ The mapping file contains extra URLs not in job descriptions."
            )
            raise ValueError(
                "Mapping file has URLs that do not exist in job descriptions."
            )

        else:
            missing_urls = job_urls_set - mapped_urls_set
            logger.info(
                f"🔹 Adding {len(missing_urls)} new entries to the mapping file..."
            )

            # Extract existing mapping dictionary
            existing_mapping = file_mappings_model.root

            for url in missing_urls:
                job_info = get_company_and_job_title(url, job_descriptions)

                company = job_info["company"]
                job_title = job_info["job_title"]

                if not company or not job_title:
                    logger.warning(f"⚠️ Skipping {url} - Missing company or job title.")
                    continue  # Skip this entry

                # Create mapping entry
                new_entry = create_mapping_entry(
                    config=config, company=company, job_title=job_title, job_url=url
                )
                existing_mapping[url] = new_entry

            logger.info(f"✅ Added {len(missing_urls)} new entries.")

            # Revalidate and save
            try:
                file_mappings_model = JobFileMappings.model_validate(existing_mapping)
                logger.info(
                    "✅ Pydantic validation successful for the updated mapping."
                )
            except ValidationError as ve:
                logger.error(f"❌ Pydantic validation error: {ve}")
                raise ValueError(f"Validation error: {ve}")

            save_to_json_file(
                data=file_mappings_model.model_dump(), file_path=mapping_file
            )
            logger.info(f"✅ Updated mapping file saved to '{mapping_file}'.")

            return file_mappings_model

    else:
        logger.info(
            f"⚠️ No existing mapping file found at '{mapping_file}'. Creating a new one..."
        )
        new_mapping = {}

        for url in job_descriptions.keys():
            job_info = get_company_and_job_title(url, job_descriptions)

            company = job_info["company"]
            job_title = job_info["job_title"]

            if not company or not job_title:
                logger.warning(f"⚠️ Skipping {url} - Missing company or job title.")
                continue

            new_mapping[url] = create_mapping_entry(config, company, job_title, url)

        # Validate new mapping
        try:
            file_mappings_model = JobFileMappings.model_validate(new_mapping)
            logger.info("✅ Validation successful for the new mapping.")
        except ValidationError as ve:
            logger.error(f"❌ Validation error in new mapping: {ve}")
            raise ValueError(f"Validation error: {ve}")

        save_to_json_file(data=file_mappings_model.model_dump(), file_path=mapping_file)
        logger.info(f"✅ New mapping file saved to '{mapping_file}'.")

        return file_mappings_model
