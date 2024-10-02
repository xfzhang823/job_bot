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
    run_pipeline as run_resume_comparison_pipeline,
)
from pipelines.resume_editing_pipeline import (
    run_pipeline as run_resume_editting_pipeline,
)
from pipelines.resume_eval_pipeline import (
    re_run_pipeline as re_run_resume_comparison_pipeline,
)
from utils.generic_utils import fetch_new_urls
from pipelines.resume_eval_pipeline import preprocess_for_eval

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
            resume_json_file=resume_json_file,
            text_file_holder=description_text_holder,
            responsibilities_flat_json_file=responsibilities_flat_json_file,
            requirements_flat_json_file=requirements_flat_json_file,
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
            resume_json_file=resume_json_file,
            # text_file_holder=description_text_holder,
            responsibilities_flat_json_file=responsibilities_flat_json_file,
            requirements_flat_json_file=requirements_flat_json_file,
        )


def run_pipeline_2():
    """Pipeline for resume evaluation"""
    logger.info("Running pipeline 2: Compare Resume with Requirements")

    # Preprocessing: Get new urls and check which are new - need to processed
    new_urls_and_f_names = preprocess_for_eval(
        job_descriptions_file=job_descriptions_json_file,
        output_dir=METRICS_OUTPUTS_CSV_FILES_DIR / "iteration_0",
    )

    logger.info("Preprocessing finished.")

    # Run pipeline for each url
    for url, filename in new_urls_and_f_names.items():
        run_resume_comparison_pipeline(
            url=url,
            requirements_json_file=job_requirements_json_file,
            resume_json_file=resume_json_file,
            csv_file=filename,
        )


def run_pipeline_3():
    """Pipeline for editing responsibility text from resume"""
    logger.info("Running pipeline 3...")
    run_resume_editting_pipeline(
        responsibilities_flat_json_file=responsibilities_flat_json_file,
        requirements_flat_json_file=requirements_flat_json_file,
        modified_resps_flat_json_file=modified_resps_flat_iter_1_json_file,
    )
    logger.info("Finished running pipeline 3.")


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
    asyncio.run(run_pipeline_1_async())
    run_pipeline_2()
    run_pipeline_3()


if __name__ == "__main__":
    run_pipeline_2()
