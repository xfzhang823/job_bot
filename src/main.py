# main.py

import os
import logging
from pathlib import Path
import logging_config
import asyncio
from tqdm import tqdm
from utils.generic_utils import fetch_new_urls, read_from_json_file, save_to_json_file
from pipelines.preprocessing_pipeline_async import (
    run_pipeline_async as run_preprocessing_pipeline_async,
)
from pipelines.preprocessing_pipeline import (
    run_pipeline as run_preprocessing_pipeline,
)
from pipelines.upserting_mapping_mini_pipeline_iter0 import (
    run_pipeline as run_upserting_mapping_file_pipeline_iter0,
)
from pipelines.flattened_resps_reqs_processing_mini_pipeline import (
    run_pipeline as run_flat_requirements_and_responsibilities_mini_pipeline,
)
from pipelines.resume_eval_pipeline import (
    metrics_processing_pipeline as run_resume_comparison_pipeline,
)
from pipelines.resume_eval_pipeline import (
    multivariate_indices_processing_mini_pipeline as run_adding_multivariate_indices_mini_pipeline,
)
from pipelines.exclude_responsibilities_mini_pipeline import (
    run_pipeline as run_excluding_responsibilities_mini_pipeline,
)
from pipelines.resume_pruning_pipeline import (
    run_pipeline as run_resume_pruning_pipeline,
)
from pipelines.pruned_resps_files_duplication_mini_pipeline import (
    run_pipe_line as run_pruned_resps_files_duplication_mini_pipeline,
)
from pipelines.upserting_mapping_mini_pipeline_iter1 import (
    run_pipeline as run_upserting_mapping_file_pipeline_iter1,
)

from pipelines.resume_editing_pipeline import (
    run_pipeline as run_resume_editting_pipeline,
)
from pipelines.resume_eval_pipeline import (
    metrics_re_processing_pipeline as re_run_resume_comparison_pipeline,
)
from pipelines.resume_eval_pipeline_async import (
    metrics_processing_pipeline_async as run_resume_comparison_pipeline_async,
)
from pipelines.resume_eval_pipeline_async import (
    multivariate_indices_processing_mini_pipeline_async as run_adding_multivariate_indices_mini_pipeline_async,
)
from pipelines.resume_editing_pipeline_async import (
    run_pipeline_async as run_resume_editting_pipeline_async,
)

# Add logging before imports
logging.info("Trying to import config...")
from config import (
    resume_json_file,
    job_posting_urls_file,
    job_descriptions_json_file,
    job_requirements_json_file,
    # modified_resps_flat_iter_1_json_file,
    # modified_resps_flat_iter_2_json_file,
    mapping_file_name,
    elbow_curve_plot_file_name,
    ITERATE_0_DIR,
    REQS_FILES_ITERATE_0_DIR,
    RESPS_FILES_ITERATE_0_DIR,
    PRUNED_RESPS_FILES_ITERATE_0_DIR,
    SIMILARITY_METRICS_ITERATE_0_DIR,
    ITERATE_1_DIR,
    REQS_FILES_ITERATE_1_DIR,
    RESPS_FILES_ITERATE_1_DIR,
    SIMILARITY_METRICS_ITERATE_1_DIR,
    PRUNED_RESPS_FILES_ITERATE_1_DIR,
)

logging.info("Successfully imported config")

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
    """
    Pipeline for creating/updating the mapping file:
    - Copy requirements files from previous iteration directory
    to the current iteration directory
    - Create the mapping file for this iteration
    """
    logger.info("Running pipeline 2a: creating/updating the mapping file.")

    run_upserting_mapping_file_pipeline_iter0(
        job_descriptions_file=job_descriptions_json_file,
        iteration=0,
        iteration_dir=ITERATE_0_DIR,
        mapping_file_name=mapping_file_name,
    )
    logger.info("Finished running pipeline 2a: creating/updating the mapping file.")


def run_pipeline_2b():
    """Pipeline for creating flattened responsibilities and requirements files"""
    logger.info(
        "Running pipeline 2b: creating flattened responsibilities and requirements files."
    )
    mapping_file_path = ITERATE_0_DIR / mapping_file_name
    logger.info(f"mapping file path: {mapping_file_path}")

    # Run pipeline
    run_flat_requirements_and_responsibilities_mini_pipeline(
        mapping_file=mapping_file_path,
        job_requirements_file=job_requirements_json_file,
        resume_json_file=resume_json_file,
    )

    logger.info(
        "Finished running pipeline 2b: creating flattened responsibilities and requirements files."
    )


def run_pipeline_2c():
    """Pipeline for resume evaluation"""
    logger.info(
        "Running pipeline 2c: match resume's responsibilities to job postings' requirements \
                to generate similarity related metrics."
    )

    mapping_file = ITERATE_0_DIR / mapping_file_name

    if not os.path.exists(mapping_file):
        logger.error(f"Mapping file not found: {mapping_file}")
        raise FileNotFoundError(f"Mapping file not found at {mapping_file}")

    # Step 2: Run the resume comparison pipeline using the mapping file
    run_resume_comparison_pipeline(mapping_file)

    logger.info(
        "Finished running pipeline 2c: match resume's responsibilities to job postings' \
                requirements to generate similarity related metrics"
    )


def run_pipeline_2d():
    """Mini-Pipeline for adding indices to metrics files"""
    logger.info("Running pipeline 2d: adding indices to metrics csv files")

    csv_files_dir = SIMILARITY_METRICS_ITERATE_0_DIR
    # Run pipeline for each url
    run_adding_multivariate_indices_mini_pipeline(csv_files_dir)
    logger.info("Finished running pipeline 2d: adding indices to metrics csv files.")


def run_pipeline_2e():
    """Pipeline to copy files & exlcude:
    - copy fles in responsilities folder to pruned_responsibilities folder, and
    - exclude certain responsibilities (not to be modified/analyzed but need
    to add back in the final stage)
    """

    logger.info(
        "Running pipeline 2e: copying fles to pruned_responsibilities folder and exclude certain responsibilities."
    )

    mapping_file = ITERATE_0_DIR / mapping_file_name

    # Copy files to the folder
    run_pruned_resps_files_duplication_mini_pipeline(mapping_file)

    # Exclude factual responsibilities text
    run_excluding_responsibilities_mini_pipeline(mapping_file)
    logger.info(
        "Finished running pipeline 2e: copying fles to pruned_responsibilities folder and exclude certain responsibilities."
    )


def run_pipeline_2f():  # for now do not run this - skip!!!!
    """Prune responsibilities"""
    logger.info(
        "Running pipeline 2e: prune resume responsibilities based on its alignment scores \
            with requirements"
    )

    # Run pipeline

    # File location of the mapping file (url: dir name: file names...)
    mapping_file = ITERATE_0_DIR / mapping_file_name

    #
    elbow_curve_plot_file = ITERATE_0_DIR / elbow_curve_plot_file_name
    elbow_method_specific_params = {
        "max_k": 15,
        "S": 12.0,
        "elbow_curve_plot_file": elbow_curve_plot_file,
    }
    run_resume_pruning_pipeline(
        mapping_file=str(mapping_file),
        pruning_method="elbow",
        group_by_responsibility=False,
        **elbow_method_specific_params,
    )

    logger.info(
        "Finished running pipeline 2e: prune resume responsibilities based on its alignment scores \
        with requirements."
    )
    return


def run_pipeline_3a():
    pipe_num = "3a"
    """
    Pipeline for creating/updating the mapping file.
    """
    logger.info(f"Running pipeline {pipe_num}: creating/updating the mapping file.")

    # Run pipeline
    run_upserting_mapping_file_pipeline_iter1(
        job_descriptions_file=job_descriptions_json_file,
        iteration=1,
        iteration_dir=ITERATE_1_DIR,
        mapping_file_name=mapping_file_name,
    )

    logger.info(
        f"Finished running pipeline {pipe_num}: creating/updating the mapping file."
    )


def run_pipeline_3b():
    """
    Pipeline for modifying responsibilities text based on requirements using LLM.
    """
    pipe_num = "3b"
    logger.info(
        f"Running pipeline {pipe_num}: modifying responsibilities text based on requirements \
                with OpenAI API"
    )

    # Run the pipeline for all responsibilities files

    mapping_file_curr_path = ITERATE_1_DIR / mapping_file_name
    mapping_file_prev_path = ITERATE_0_DIR / mapping_file_name

    run_resume_editting_pipeline(
        mapping_file_prev=mapping_file_prev_path,
        mapping_file_curr=mapping_file_curr_path,
        model="openai",
        model_id="gpt-4-turbo",
    )

    logger.info(
        f"Finished running pipeline {pipe_num}: modifying responsibilities based on requirments."
    )


def run_pipeline_3c():
    pipe_num = "3c"
    logger.info(
        f"Running pipeline {pipe_num}: match resume's responsibilities to job postings' requirements \
                to generate similarity related metrics."
    )
    mapping_file = ITERATE_1_DIR / mapping_file_name

    logger.info("Running pipeline {pipe_num}: ...")
    re_run_resume_comparison_pipeline(mapping_file)

    logger.info(
        f"Finish running pipeline {pipe_num}: match resume's responsibilities to job postings' requirements \
                to generate similarity related metrics."
    )


def run_pipeline_3d():
    pipe_num = "3d"
    logger.info(
        f"Running pipeline {pipe_num}: adding multivariate indices to metrics files"
    )

    # Set data_directory
    csv_files_dir = SIMILARITY_METRICS_ITERATE_1_DIR

    # Run pipeline
    run_adding_multivariate_indices_mini_pipeline(csv_files_dir)

    logger.info(
        f"Finish running pipeline {pipe_num}: adding multivariate indices to metrics files"
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


async def run_pipeline_2c_async():
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


async def run_pipeline_2d_async():
    """Asynchronous Mini-Pipeline for adding indices to metrics files."""
    logger.info("Running pipeline 2b: adding indices to metrics csv files")

    csv_files_dir = SIMILARITY_METRICS_ITERATE_0_DIR
    # Run pipeline for each url
    await run_adding_multivariate_indices_mini_pipeline_async(csv_files_dir)
    logger.info("Finished running pipeline 2b: adding indices to metrics csv files.")


async def run_pipeline_3b_async():
    """
    Pipeline for modifying responsibilities text based on requirements using LLM.
    """
    pipe_num = "3b async"
    logger.info(
        f"Running pipeline {pipe_num}: modifying responsibilities text based on requirements \
                with OpenAI API"
    )

    # Run the pipeline for all responsibilities files

    mapping_file_curr_path = ITERATE_1_DIR / mapping_file_name
    mapping_file_prev_path = ITERATE_0_DIR / mapping_file_name

    await run_resume_editting_pipeline_async(
        mapping_file_prev=mapping_file_prev_path,
        mapping_file_curr=mapping_file_curr_path,
        model="openai",
        model_id="gpt-4-turbo",
    )

    logger.info(
        f"Finished running pipeline {pipe_num}: modifying responsibilities based on requirments."
    )


def main():
    """main to run the pipelines"""
    # run_pipeline_1()
    # run_pipeline_2a()
    # run_pipeline_2b()
    # run_pipeline_2c()
    # run_pipeline_2d()
    # run_pipeline_2e()
    # run_pipeline_3a()
    asyncio.run(run_pipeline_3b_async())
    run_pipeline_3c()
    run_pipeline_3d()


async def main_async():
    """main to run the pipelines asynchronously"""
    await run_pipeline_1_async()
    run_pipeline_2a()
    await run_pipeline_2c_async()


if __name__ == "__main__":
    # asyncio.run(main_async())
    # run_pipeline_3a()
    main()
