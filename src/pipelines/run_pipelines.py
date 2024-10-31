# pipelines/run_pipelines.py
import os
import logging
from tqdm import tqdm
import asyncio

from utils.generic_utils import fetch_new_urls
from pipeline_config import PIPELINE_CONFIG

# Pipeline functions
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
from pipelines.copying_resps_to_pruned_resps_dir_mini_pipeline import (
    run_pipe_line as run_copying_resps_to_pruned_resps_mini_pipeline,
)
from pipelines.copying_reqs_to_next_iter_mini_pipeline import (
    run_pipeline as run_copying_requirements_to_next_iteration_mini_pipeline,
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
    metrics_re_processing_pipeline_async as re_run_resume_comparison_pipeline_async,
)
from pipelines.resume_eval_pipeline_async import (
    multivariate_indices_processing_mini_pipeline_async as run_adding_multivariate_indices_mini_pipeline_async,
)
from pipelines.resume_editing_pipeline_async import (
    run_pipeline_async as run_resume_editting_pipeline_async,
)

logger = logging.getLogger(__name__)


def run_pipeline(pipeline_id: str, llm_provider: str = "openai"):
    """
    Runs the specified pipeline based on configuration in PIPELINE_CONFIG.

    Args:
        pipeline_id (str): The identifier of the pipeline to execute.
        provider (str): The LLM provider to use ('openai' or 'claude').
    """
    config = PIPELINE_CONFIG[pipeline_id]
    func_name = config["function"]
    io_config = config["io"][llm_provider]

    logger.info(
        f"Running pipeline '{config['description']}' for provider '{llm_provider}'"
    )

    globals()[func_name](**io_config)


def run_pipeline_1():
    """
    Synchronous pipeline for preprocessing job posting webpages.

    This pipeline identifies and processes new URLs. For each new URL,
    it invokes the `run_preprocessing_pipeline` function to extract relevant
    data and save it. If no new URLs are found, the function logs and exits.
    """
    config = PIPELINE_CONFIG["1"]
    io_config = config["io"]["openai"]

    new_urls = fetch_new_urls(
        existing_url_list_file=io_config["existing_url_list_file"],
        url_list_file=io_config["url_list_file"],
    )
    if not new_urls:
        logger.info("No new URLs found. Skipping...")
        return

    for url in tqdm(new_urls, desc="Processing job postings", unit="job"):
        logger.info(f"Processing for URL: {url}")
        run_preprocessing_pipeline(
            job_description_url=url,
            job_descriptions_json_file=io_config["existing_url_list_file"],
            requirements_json_file=io_config["url_list_file"],
        )


def run_pipeline_2a():
    """
    Pipeline to create/update the mapping file for iteration 0.

    This function updates the mapping file and copies the requirements files from
    the previous directory to the current one.
    """
    config = PIPELINE_CONFIG["2a"]
    io_config = config["io"]["openai"]

    run_upserting_mapping_file_pipeline_iter0(
        job_descriptions_file=io_config["job_descriptions_file"],
        iteration_dir=io_config["iteration_dir"],
        iteration=io_config["iteration"],
        mapping_file_name=io_config["mapping_file_name"],
    )


def run_pipeline_2b():
    """
    Pipeline to flatten responsibilities and requirements files for iteration 0.
    """
    config = PIPELINE_CONFIG["2b"]
    io_config = config["io"]["openai"]

    run_flat_requirements_and_responsibilities_mini_pipeline(
        mapping_file=io_config["mapping_file"],
        job_requirements_file=io_config["job_requirements_file"],
        resume_json_file=io_config["resume_json_file"],
    )


def run_pipeline_2c():
    """
    Pipeline to evaluate resumes against job requirements and generate similarity metrics.
    """
    config = PIPELINE_CONFIG["2c"]
    io_config = config["io"]["openai"]

    if not os.path.exists(io_config["mapping_file"]):
        logger.error(f"Mapping file not found: {io_config['mapping_file']}")
        raise FileNotFoundError(
            f"Mapping file not found at {io_config['mapping_file']}"
        )

    run_resume_comparison_pipeline(io_config["mapping_file"])


def run_pipeline_2d():
    """
    Pipeline to add indices to metrics files based on similarity metrics.
    """
    config = PIPELINE_CONFIG["2d"]
    io_config = config["io"]["openai"]

    run_adding_multivariate_indices_mini_pipeline(io_config["csv_files_dir"])


def run_pipeline_2e():
    """
    Pipeline to copy files in responsibilities folder to pruned_responsibilities folder,
    and exclude certain responsibilities.
    """
    config = PIPELINE_CONFIG["2e"]
    io_config = config["io"]["openai"]

    run_copying_resps_to_pruned_resps_mini_pipeline(io_config["mapping_file"])
    run_excluding_responsibilities_mini_pipeline(io_config["mapping_file"])


def run_pipeline_3a():
    """
    Pipeline to create or upsert the mapping file for iteration 1.
    """
    config = PIPELINE_CONFIG["3a"]
    io_config = config["io"]["openai"]

    run_upserting_mapping_file_pipeline_iter1(
        job_descriptions_file=io_config["job_descriptions_file"],
        iteration_dir=io_config["iteration_dir"],
        iteration=io_config["iteration"],
        mapping_file_name=io_config["mapping_file_name"],
    )


def run_pipeline_3b(llm_provider: str):
    """
    Pipeline to modify responsibilities text based on requirements using LLM.

    Args:
        llm_provider (str): The LLM provider, e.g., 'openai' or 'claude'.
    """
    config = PIPELINE_CONFIG["3b_async"]
    io_config = config["io"][llm_provider]

    # Retrieve and validate model_id
    model_id = config.get("model_id")
    if model_id is None:
        raise ValueError(
            f"Model ID is not defined for pipeline '3b_async' using provider '{llm_provider}'."
        )

    # Validate other essential inputs
    mapping_file_prev = io_config.get("mapping_file_prev")
    mapping_file_curr = io_config.get("mapping_file_curr")
    if not mapping_file_prev or not mapping_file_curr:
        raise ValueError(
            f"One or more required I/O configurations are missing for provider '{llm_provider}'."
        )

    run_resume_editting_pipeline(
        mapping_file_prev=mapping_file_prev,
        mapping_file_curr=mapping_file_curr,
        llm_provider=llm_provider,
        model_id=model_id,
    )


def run_pipeline_3c():
    """
    Pipeline to copy requirements files from iteration 0 to iteration 1.
    """
    config = PIPELINE_CONFIG["3c"]
    io_config = config["io"]["openai"]

    run_copying_requirements_to_next_iteration_mini_pipeline(
        mapping_file_prev=io_config["mapping_file_prev"],
        mapping_file_curr=io_config["mapping_file_curr"],
    )


def run_pipeline_3d():
    """
    Pipeline to match resume's responsibilities to job postings' requirements
    and generate similarity metrics.
    """
    config = PIPELINE_CONFIG["3d"]
    io_config = config["io"]["openai"]

    re_run_resume_comparison_pipeline(io_config["mapping_file"])


def run_pipeline_3e():
    """
    Pipeline to add multivariate indices to metrics files in iteration 1.
    """
    config = PIPELINE_CONFIG["3e"]
    io_config = config["io"]["openai"]

    run_adding_multivariate_indices_mini_pipeline(io_config["csv_files_dir"])


async def run_pipeline_1_async():
    """
    Async pipeline for preprocessing job posting webpage(s).
    """
    config = PIPELINE_CONFIG["1_async"]
    io_config = config["io"]["openai"]

    new_urls = fetch_new_urls(
        existing_url_list_file=io_config["existing_url_list_file"],
        url_list_file=io_config["url_list_file"],
    )
    if not new_urls:
        logger.info("No new URLs found. Skipping...")
        return

    for url in tqdm(new_urls, desc="Processing job postings", unit="job"):
        await run_preprocessing_pipeline_async(
            job_description_url=url,
            job_descriptions_json_file=io_config["existing_url_list_file"],
            requirements_json_file=io_config["url_list_file"],
        )


async def run_pipeline_2c_async():
    """
    Async pipeline for resume evaluation in iteration 0.
    """
    config = PIPELINE_CONFIG["2c_async"]
    io_config = config["io"]["openai"]

    if not os.path.exists(io_config["mapping_file"]):
        raise FileNotFoundError(
            f"Mapping file not found at {io_config['mapping_file']}"
        )

    await run_resume_comparison_pipeline_async(io_config["mapping_file"])


async def run_pipeline_2d_async():
    """
    Async pipeline for adding indices to metrics files.
    """
    config = PIPELINE_CONFIG["2d_async"]
    io_config = config["io"]["openai"]

    await run_adding_multivariate_indices_mini_pipeline_async(
        io_config["csv_files_dir"]
    )


async def run_pipeline_3b_async(llm_provider):
    """
    Async pipeline for modifying responsibilities text based on requirements using LLM.

    Args:
        llm_provider (str): The LLM provider, e.g., 'openai' or 'claude'.
    """
    config = PIPELINE_CONFIG["3b_async"]
    io_config = config["io"][llm_provider]
    model_id = config.get("model_id")
    if model_id is None:
        raise ValueError(
            f"Model ID is not defined for pipeline '3b_async' using provider '{llm_provider}'."
        )

    # Validate other essential inputs
    mapping_file_prev = io_config.get("mapping_file_prev")
    mapping_file_curr = io_config.get("mapping_file_curr")
    if not mapping_file_prev or not mapping_file_curr:
        raise ValueError(
            f"One or more required I/O configurations are missing for provider '{llm_provider}'."
        )
    await run_resume_editting_pipeline_async(
        mapping_file_prev=mapping_file_prev,
        mapping_file_curr=mapping_file_curr,
        llm_provider=llm_provider,
        model_id=model_id,
    )


async def run_pipeline_3d_async():
    """
    Async version of the resume comparison pipeline for iteration 1.
    """
    config = PIPELINE_CONFIG["3d_async"]
    io_config = config["io"]["openai"]

    await re_run_resume_comparison_pipeline_async(io_config["mapping_file"])
