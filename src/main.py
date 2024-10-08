# main.py

import os
import logging
from pathlib import Path
import logging_config
import asyncio
from tqdm import tqdm
from utils.generic_utils import fetch_new_urls
from pipelines.preprocessing_pipeline_async import (
    run_pipeline_async as run_preprocessing_pipeline_async,
)
from pipelines.preprocessing_pipeline import (
    run_pipeline as run_preprocessing_pipeline,
)
from pipelines.generating_flat_resps_reqs_mini_pipeline import (
    run_pipeline as run_flat_requirements_and_responsibilities_mini_pipeline,
)
from pipelines.resume_eval_pipeline import (
    metrics_processing_pipeline as run_resume_comparison_pipeline,
)
from pipelines.resume_eval_pipeline import (
    multivariate_indices_processing_mini_pipeline as run_adding_multivariate_indices_mini_pipeline,
)
from pipelines.resume_eval_pipeline import generate_matching_metrics

from pipelines.resume_eval_pipeline_async import (
    metrics_processing_pipeline_async as run_resume_comparison_pipeline_async,
)
from pipelines.resume_eval_pipeline_async import (
    multivariate_indices_processing_mini_pipeline_async as run_adding_multivariate_indices_mini_pipeline_async,
)
from pipelines.resume_editing_pipeline import (
    multi_files_modification_pipeline as run_resume_editting_pipeline,
)
from pipelines.resume_eval_pipeline import (
    re_processing_metrics_pipeline as re_run_resume_comparison_pipeline,
)

from config import (
    resume_json_file,
    job_posting_urls_file,
    job_descriptions_json_file,
    job_requirements_json_file,
    modified_resps_flat_iter_1_json_file,
    modified_resps_flat_iter_2_json_file,
    mapping_file_name,
    ITERATE_0_DIR,
    FLAT_REQS_FILES_ITERATE_0_DIR,
    FLAT_RESPS_FILES_ITERATE_0_DIR,
    PRUNED_FLAT_RESPS_FILES_ITERATE_0_DIR,
    SIMILARITY_METRICS_ITERATE_0_DIR,
)


# Setup logger
logger = logging.getLogger(__name__)


def run_pipeline_1():
    """
    Pipeline for preprocessing job posting webpage(s).
    This is the sync version.

    If there are multiple URL links, iterate through them and
    run the `run_preprocessing_pipeline` function for each.

    The pipeline will skip processing if no new URLs are found.
    """
    # Fetch new URLs to process
    new_urls = fetch_new_urls(
        existing_url_list_file=job_descriptions_json_file,
        url_list_file=job_posting_urls_file,
    )

    # Check if new_urls is empty; if so, return early
    if not new_urls:
        logger.info("No new URLs found... Skipping this process.")
        return  # Early return

    logger.info("New URLs found:")
    for url in new_urls:  # debugging
        print(url)

    # Initialize tqdm for progress visualization;
    # Iterate through the list of urls and run the preprocessing pipeline for each
    for url in tqdm(new_urls, desc="Processing job postings", unit="job"):

        logger.info(f"Processing for URL: {url}")

        run_preprocessing_pipeline(
            job_description_url=url,
            job_descriptions_json_file=job_descriptions_json_file,
            requirements_json_file=job_requirements_json_file,
            # resume_json_file=resume_json_file,
            # text_file_holder=description_text_holder,
        )


def run_pipeline_2a():
    """Pipeline for creating flattened responsibilities and requirements files"""
    logger.info(
        "Running pipeline 2a: creating flattened responsibilities and requirements files."
    )

    run_flat_requirements_and_responsibilities_mini_pipeline(
        job_descriptions_file=job_descriptions_json_file,
        job_requirements_file=job_requirements_json_file,
        resume_json_file=resume_json_file,
        flat_reqs_output_files_dir=FLAT_REQS_FILES_ITERATE_0_DIR,
        flat_resps_output_files_dir=FLAT_RESPS_FILES_ITERATE_0_DIR,
        sim_metrics_output_files_dir=SIMILARITY_METRICS_ITERATE_0_DIR,
        mapping_file_dir=ITERATE_0_DIR,
        mapping_file_name=mapping_file_name,
    )

    logger.info(
        "Finished running pipeline 2a: creating flattened responsibilities and requirements files."
    )


def run_pipeline_2b():
    """Pipeline for resume evaluation"""
    logger.info(
        "Running pipeline 2b: match resume's responsibilities to job postings' Requirements \
                to generate similarity related metrics."
    )

    mapping_file = ITERATE_0_DIR / mapping_file_name

    if not os.path.exists(mapping_file):
        logger.error(f"Mapping file not found: {mapping_file}")
        raise FileNotFoundError(f"Mapping file not found at {mapping_file}")

    # Step 2: Run the resume comparison pipeline using the mapping file
    run_resume_comparison_pipeline(mapping_file)

    logger.info(
        "Finished running pipeline 2b: match resume's responsibilities to job postings' \
                Requirements to generate similarity related metrics"
    )


def run_pipeline_2c():
    """Mini-Pipeline for adding indices to metrics files"""
    logger.info("Running pipeline 2b: adding indices to metrics csv files")

    csv_files_dir = SIMILARITY_METRICS_ITERATE_0_DIR
    # Run pipeline for each url
    run_adding_multivariate_indices_mini_pipeline(csv_files_dir)
    logger.info("Finished running pipeline 2b: adding indices to metrics csv files.")


def run_pipeline_3():
    """Pipeline to modify all responsibility files in the given directory."""

    # Define your directories and file paths
    responsibilities_dir = FLAT_RESPS_FILES_ITERATE_0_DIR
    # requirements_flat_json_file = FLAT
    # modified_resps_output_dir = Path("path/to/modified_responsibilities")

    mapping_file_path = ITERATE_0_DIR / mapping_file_name

    # Run the pipeline for all responsibilities files
    run_resume_editting_pipeline(
        mapping_file=mapping_file_path, model="openai", model_id="gpt-4-turbo"
    )


def run_pipeline_4():
    """Pipeline for re-run resume evaluation"""
    logger.info("Running pipeline 4...")
    re_run_resume_comparison_pipeline(
        requirements_file=requirements_flat_json_file,
        responsibilities_file=modified_resps_flat_iter_1_json_file,
        csv_file=resp_req_sim_metrics_1_csv_file,
    )


# This should be the main pipeline b/c it's a lot faster
async def run_pipeline_1_async():
    """
    Asynchronous pipeline for preprocessing job posting webpage(s).
    - If there are multiple URL links, iterate through them and run the `run_pipeline_async`
    function for each.
    - However, the pipeline is for ONE JOB SITE ONLY.
    - Multiple job sites will be iterated through the pipeline multiple times.

    The pipeline will skip processing if no new URLs are found.
    """
    # Fetch new URLs to process
    new_urls = fetch_new_urls(
        existing_url_list_file=job_descriptions_json_file,
        url_list_file=job_posting_urls_file,
    )

    # Check if new_urls is empty; if so, return early
    if not new_urls:
        logger.info("No new URLs found... Skipping this process.")
        return  # Early return

    logger.info("New URLs found:")
    for url in new_urls:  # debugging
        print(url)

    # Initialize tqdm for progress visualization;
    # Iterate through the list of urls and run the preprocessing pipeline for each
    for url in tqdm(new_urls, desc="Processing job postings", unit="job"):
        logger.info(f"Processing for URL: {url}")
        await run_preprocessing_pipeline_async(
            job_description_url=url,
            job_descriptions_json_file=job_descriptions_json_file,
            requirements_json_file=job_requirements_json_file,
        )


async def run_pipeline_2b_async():
    """Asynchronous pipeline for resume evaluation"""
    logger.info(
        "Running pipeline 2b: match resume's responsibilities to job postings' Requirements \
                to generate similarity related metrics."
    )

    mapping_file = ITERATE_0_DIR / mapping_file_name

    if not os.path.exists(mapping_file):
        logger.error(f"Mapping file not found: {mapping_file}")
        raise FileNotFoundError(f"Mapping file not found at {mapping_file}")

    # Step 2: Run the resume comparison pipeline using the mapping file
    await run_resume_comparison_pipeline_async(mapping_file)

    logger.info(
        "Finished running pipeline 2b: match resume's responsibilities to job postings' \
                Requirements to generate similarity related metrics"
    )


async def run_pipeline_2c_async():
    """Asynchronous Mini-Pipeline for adding indices to metrics files."""
    logger.info("Running pipeline 2b: adding indices to metrics csv files")

    csv_files_dir = METRICS_OUTPUTS_CSV_FILES_DIR / "iteration_0"
    # Run pipeline for each url
    await run_adding_multivariate_indices_mini_pipeline_async(csv_files_dir)
    logger.info("Finished running pipeline 2b: adding indices to metrics csv files.")


def main():
    """main to run the pipelines"""
    run_pipeline_1()
    run_pipeline_2a()
    run_pipeline_2b()
    run_pipeline_2c()


async def main_async():
    """main to run the pipelines asynchronously"""
    await run_pipeline_1_async()
    run_pipeline_2a()
    await run_pipeline_2b_async()


if __name__ == "__main__":
    # asyncio.run(main_async())
    # main()
    run_pipeline_2c()
