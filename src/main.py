# main.py

import os
import logging
import logging_config
import asyncio
from tqdm import tqdm
from pipelines.preprocessing_pipeline_async import (
    run_pipeline_async as run_preprocessing_pipeline_async,
)
from pipelines.preprocessing_pipeline import (
    run_pipeline as run_preprocessing_pipeline,
)
from pipelines.resume_eval_pipeline import (
    metrics_processing_pipeline as run_resume_comparison_pipeline,
)
from pipelines.resume_eval_pipeline import (
    multivariate_indices_processing_mini_pipeline as run_adding_multivariate_indices_mini_pipeline,
)
from pipelines.resume_eval_pipeline_async import (
    metrics_processing_pipeline_async as run_resume_comparison_pipeline_async,
)
from pipelines.resume_eval_pipeline_async import (
    multivariate_indices_processing_mini_pipeline_async as run_adding_multivariate_indices_mini_pipeline_async,
)
from pipelines.resume_eval_pipeline_async import (
    metrics_preprocessing_mini_pipeline_async,
)
from pipelines.resume_editing_pipeline import (
    flat_json_files_processing_mini_pipeline as run_processing_reqs_resp_flat_files_mini_pipeline,
)
from pipelines.resume_editing_pipeline import (
    run_pipeline as run_resume_editting_pipeline,
)
from pipelines.resume_eval_pipeline import (
    re_processing_metrics_pipeline as re_run_resume_comparison_pipeline,
)
from utils.generic_utils import fetch_new_urls
from pipelines.resume_eval_pipeline import metrics_preprocessing_mini_pipeline

from config import (
    resume_json_file,
    job_posting_urls_file,
    job_descriptions_json_file,
    job_requirements_json_file,
    description_text_holder,
    responsibilities_flat_json_file,
    requirements_flat_json_file,
    resp_req_sim_metrics_0_csv_file,
    resp_req_sim_metrics_1_csv_file,
    modified_resps_flat_iter_1_json_file,
    modified_resps_flat_iter_2_json_file,
    METRICS_OUTPUTS_CSV_FILES_DIR,
    SIMILARITY_METRICS_ITERATE_0_DIR,
    FLAT_JSON_FILES_ITERATE_0_DIR,
)


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


def run_pipeline_2a():
    """Pipeline for resume evaluation"""
    logger.info("Running pipeline 2a: Compare Resume with Requirements")

    # Preprocessing: Get new urls and check which are new - need to processed
    new_urls_and_f_names = metrics_preprocessing_mini_pipeline(
        job_descriptions_file=job_descriptions_json_file,
        output_dir=METRICS_OUTPUTS_CSV_FILES_DIR / "iteration_0",
    )

    # Run pipeline for each url
    for url, filename in new_urls_and_f_names.items():
        run_resume_comparison_pipeline(
            url=url,
            requirements_json_file=job_requirements_json_file,
            resume_json_file=resume_json_file,
            csv_file=filename,
        )

    logger.info("Finished running pipeline 2a: Compare Resume with Requirements")


async def run_pipeline_2a_async():
    """Asynchronous pipeline for resume evaluation"""
    logger.info("Running pipeline 2a: Compare Resume with Requirements")

    # Preprocessing: Get new urls and check which are new - need to processed
    new_urls_and_f_names = await metrics_preprocessing_mini_pipeline_async(
        job_descriptions_file=job_descriptions_json_file,
        output_dir=METRICS_OUTPUTS_CSV_FILES_DIR / "iteration_0",
    )

    # Run the pipeline concurrently for each URL
    tasks = [
        run_resume_comparison_pipeline_async(
            url=url,
            requirements_json_file=job_requirements_json_file,
            resume_json_file=resume_json_file,
            csv_file=filename,
        )
        for url, filename in new_urls_and_f_names.items()
    ]

    # Use asyncio.gather() process them concurrently
    await asyncio.gather(*tasks)
    logger.info("Finished running pipeline 2a: Compare Resume with Requirements")


def run_pipeline_2b():
    """Mini-Pipeline for adding indices to metrics files"""
    logger.info("Running pipeline 2b: adding indices to metrics csv files")

    csv_files_dir = METRICS_OUTPUTS_CSV_FILES_DIR / "iteration_0"
    # Run pipeline for each url
    run_adding_multivariate_indices_mini_pipeline(csv_files_dir)
    logger.info("Finished running pipeline 2b: adding indices to metrics csv files.")


async def run_pipeline_2b_async():
    """Asynchronous Mini-Pipeline for adding indices to metrics files."""
    logger.info("Running pipeline 2b: adding indices to metrics csv files")

    csv_files_dir = METRICS_OUTPUTS_CSV_FILES_DIR / "iteration_0"
    # Run pipeline for each url
    await run_adding_multivariate_indices_mini_pipeline_async(csv_files_dir)
    logger.info("Finished running pipeline 2b: adding indices to metrics csv files.")


def run_pipeline_3a():
    """Pipeline for creating flattened responsibilities and requirements files"""
    logger.info(
        "Running pipeline 3a: creating flattened responsibilities and requirements files."
    )
    logger.info(f"Job descriptions file: {job_descriptions_json_file}")
    run_processing_reqs_resp_flat_files_mini_pipeline(
        job_descriptions_file=job_descriptions_json_file,
        job_requirements_file=job_requirements_json_file,
        resume_json_file=resume_json_file,
        flat_json_output_files_dir=FLAT_JSON_FILES_ITERATE_0_DIR,
    )
    logger.info(
        "Finished running pipeline 3a: creating flattened responsibilities and requirements files."
    )


def run_pipeline_3b():
    """Pipeline for editing responsibility text from resume"""
    logger.info("Running pipeline 3b: modifying (resume) responsibilities.")
    run_resume_editting_pipeline(
        responsibilities_flat_json_file=responsibilities_flat_json_file,
        requirements_flat_json_file=requirements_flat_json_file,
        modified_resps_flat_json_file=modified_resps_flat_iter_1_json_file,
    )
    logger.info("Finished running pipeline 3b: modifying (resume) responsibilities.")


def run_pipeline_4():
    """Pipeline for re-run resume evaluation"""
    logger.info("Running pipeline 4...")
    re_run_resume_comparison_pipeline(
        requirements_file=requirements_flat_json_file,
        responsibilities_file=modified_resps_flat_iter_1_json_file,
        csv_file=resp_req_sim_metrics_1_csv_file,
    )


def main():
    """main to run the pipelines"""
    run_pipeline_1()
    run_pipeline_2a()
    run_pipeline_2b()
    run_pipeline_3a()


async def main_async():
    """main to run the pipelines asynchronously"""
    # asyncio.run(run_pipeline_1_async())
    await run_pipeline_2a_async()
    await run_pipeline_2b_async()


if __name__ == "__main__":
    # asyncio.run(main_async())
    # main()
    run_pipeline_3a()
