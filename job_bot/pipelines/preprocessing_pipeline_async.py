"""
The pipeline module contains all high-level functions.

This module is to be called by main.py.
"""

# Import libraries
import os
import json
import logging
from dotenv import load_dotenv
from typing import List, Union
from pathlib import Path
import asyncio
import openai
from llm_providers.llm_api_utils_async import (
    call_anthropic_api_async,
    call_openai_api_async,
)
from utils.generic_utils import (
    fetch_new_urls,
    pretty_print_json,
    load_or_create_json,
    read_from_json_file,
    add_to_json_file,
    save_to_json_file,
)
from utils.webpage_reader_async import process_webpages_to_json_async
from utils.generic_utils_async import add_new_data_to_json_file_async
from prompts.prompt_templates import EXTRACT_JOB_REQUIREMENTS_PROMPT
from preprocessing.resume_preprocessor import ResumeParser
from preprocessing.requirements_preprocessor import JobRequirementsParser
from preprocessing.extract_requirements_with_llms_async import (
    extract_job_requirements_with_anthropic_async,
    extract_job_requirements_with_openai_async,
)
from preprocessing.preprocessing_utils import find_new_urls
from models.llm_response_models import JobSiteResponse, RequirementsResponse
from models.resume_job_description_io_models import (
    JobPostingsBatch,
    ExtractedRequirementsBatch,
)
from project_config import (
    OPENAI,
    ANTHROPIC,
    GPT_4_1_NANO,
    GPT_35_TURBO,
    CLAUDE_HAIKU,
    CLAUDE_SONNET_3_5,
)

# Set up logging
logger = logging.getLogger(__name__)

# Define the maximum number of concurrent tasks (limit this to not hit the API rate limit)
MAX_CONCURRENT_REQUESTS = 3  # Based on your rate limit or desired concurrency level


async def process_single_url_async(
    job_description_url: str,
    job_descriptions_json_file: Path,
    job_requirements_json_file: Path,
    lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,  # limit # of concurrent worker (avoid rate limit)
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_1_NANO,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    web_scrape: bool = True,  # Flag to control whether to need to webscrape the content or not
) -> None:
    """
    Asynchronously processes a single job posting URL, fetches the job description
    (with or without web scraping), and extracts job requirements using GPT or other LLMs.

    Args:
        - job_description_url (str): The job description URL to process.
        - job_descriptions_json_file (Path): Path to the job descriptions JSON file.
        - job_requirements_json_file (Path): Path to the job requirements JSON file.
        - lock (asyncio.Lock): A lock to ensure thread safety during critical sections.
        #* The lock is put in the function b/c each URL processing must be done before
        #* it is saved to file.
        - semaphore (asyncio.Semaphore): To limit the number of concurrent tasks.
        - llm_provider (str): The LLM provider to use (default is "openai").
        - model_id (str): The model ID to use for GPT (default is "gpt-4-turbo").
        - max_tokens (int): The maximum number of tokens to generate. Defaults to 2048.
        - temperature (float): The temperature value. Defaults to 0.3.
        - web_scrape (bool): Flag to control whether to scrape the content or not.

    Returns:
        None
    """
    logger.info(f"Processing URL: {job_description_url}")

    # Apply the semaphore to limit concurrent execution
    async with semaphore:

        if web_scrape:
            # Step 1a: If scraping, Webpage -> job description JSON format
            job_description_model = await process_webpages_to_json_async(
                urls=job_description_url,
                llm_provider=llm_provider,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )  # Returns a dict: Dict[str, Dict[str, Any]]
            logger.info(f"Job description fetched for URL: {job_description_url}")
        else:
            # Step 1b: If not scraping, read existing data from the job_description_url
            # (assumed to be available)
            job_description_model = JobSiteResponse.model_validate(
                read_from_json_file(job_descriptions_json_file).get(
                    job_description_url, {}
                )
            )
            logger.info(f"Job description fetched for URL: {job_description_url}")

        logger.info(job_description_model)  # todo: debug; delete later

        # Gatekeeper: if no meaningful content was extracted, skip further processing.
        # Unwrap the outer dict: expect a single URL key
        if isinstance(job_description_model, dict) and len(job_description_model) == 1:
            first_val = next(iter(job_description_model.values()))
            data_section = first_val.get("data", {})

            content = data_section.get("content")

            logger.debug(f"data keys: {list(data_section.keys())}")
            logger.debug(
                f"content preview: {json.dumps(content, indent=2) if content else 'None'}"
            )

            # Check that content is a dict and has at least one non-empty string value
            if not isinstance(content, dict) or not any(
                v.strip() for v in content.values() if isinstance(v, str)
            ):
                logger.error(
                    f"Content is missing or has no meaningful values: {job_description_url}"
                )
                return
        else:
            logger.error(
                f"Unexpected format for job_description_dict: {job_description_model}"
            )
            return

        logger.debug(
            f"🔍 content preview: {json.dumps(content, indent=2) if content else 'None'}"
        )

        content = data_section.get("content")
        if not content:
            logger.error(f"No content found for URL: {job_description_url}, skipping.")
            return
        #! Early return (do not save or continute to extract requirements)

        #! According to OpenAI, you shouldn't passing JSON str into LLM;
        #! howeever, according to Anthropic and DeepSeek, JSON input is fine.
        # Step 2: Converst job_description from dict to json string
        job_description_json = json.dumps(data_section, indent=2)

        # Extract job requirements using openai or anthropic
        if llm_provider.lower() == OPENAI:
            requirements_model = await extract_job_requirements_with_openai_async(
                job_description=job_description_json, model_id=model_id
            )
        elif llm_provider.lower() == ANTHROPIC:
            requirements_model = await extract_job_requirements_with_anthropic_async(
                job_description=job_description_json, model_id=model_id
            )
        else:
            raise ValueError(f"{llm_provider} is not a supported API.")

        logger.info(f"Job Requirements: \n{requirements_model}")

        # Use the lock when writing to files to avoid race conditions
        async with lock:
            # 🔄 Load and validate both JSON files into Pydantic batch models
            existing_job_descriptions_dict = read_from_json_file(
                job_descriptions_json_file
            )
            existing_job_requirements_dict = read_from_json_file(
                job_requirements_json_file
            )

            job_desc_batch = JobPostingsBatch.model_validate(
                existing_job_descriptions_dict
            )  # type: ignore[attr-defined]
            job_reqs_batch = ExtractedRequirementsBatch.model_validate(
                existing_job_requirements_dict
            )  # type: ignore[attr-defined]

            # ✅ Add new entries directly to the .root dicts (preserves model integrity)
            if job_description_url not in job_desc_batch.root:
                job_desc_batch.root[job_description_url] = job_description_model

            if job_description_url not in job_reqs_batch.root:
                job_reqs_batch.root[job_description_url] = requirements_model

            # 💾 Dump full models back to disk
            await add_new_data_to_json_file_async(
                job_desc_batch.model_dump(), job_descriptions_json_file
            )
            await add_new_data_to_json_file_async(
                job_reqs_batch.model_dump(), job_requirements_json_file
            )

        logger.info(
            f"Job description and requirements processed for URL: {job_description_url}"
        )


async def process_urls_async(
    urls: List[str],
    job_descriptions_json_file: Path,
    job_requirements_json_file: Path,
    llm_provider: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    web_scrape: bool,
    lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
) -> List[asyncio.Task]:
    """
    Creates a list of asyncio tasks for processing URLs asynchronously.

    Returns:
        - List of `asyncio.Task` objects that can be awaited.
    """
    tasks = [
        asyncio.create_task(
            process_single_url_async(
                job_description_url=url,
                job_descriptions_json_file=job_descriptions_json_file,
                job_requirements_json_file=job_requirements_json_file,
                lock=lock,
                semaphore=semaphore,
                llm_provider=llm_provider,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                web_scrape=web_scrape,
            )
        )
        for url in urls
    ]
    return tasks  # Returns a list of `asyncio.Task` instead of coroutines


async def run_preprocessing_pipeline_async(
    job_posting_urls_file: Union[Path, str],
    job_descriptions_json_file: Union[Path, str],
    job_requirements_json_file: Union[Path, str],
    llm_provider: str = "openai",  # llm_provider is passed from the orchestrating function
    model_id: str = GPT_4_1_NANO,  # default to gpt 4 turbo
    max_tokens: int = 4096,
    temperature: float = 0.3,
):
    """
    Asynchronous pipeline for preprocessing job posting webpage(s):
    - Orchestrates the entire pipeline for modifying a resume based on a job description.
    - If there are multiple URL links, iterate through them and run the `process_single_url`
    function for each.
    - The pipeline processes job descriptions and extracts job requirements for each URL.

    Args:
        - job_posting_urls_file (str or Path): Path to the job postings JSON file with URLs.
        - job_descriptions_json_file (str or Path): Path to the job description JSON file.
        - job_requirements_json_file (str or Path): Path to the extracted job requirements
        JSON file.
        - llm_provider (str): The LLM provider to use (default is "openai").
        - model_id (str): The model ID to use for GPT (default is "gpt-4-turbo").
        - max_tokens (int): The maximum number of tokens to generate. Defaults to 2048.
        - temperature (float): The temperature value. Defaults to 0.3.

    Returns:
        None
    """
    logger.info("Starting preprocessing pipeline async...")

    # * Step 0: Set up and error handling

    # Convert file paths to Path obj. if not already
    job_descriptions_json_file, job_posting_urls_file, job_requirements_json_file = (
        Path(job_descriptions_json_file),
        Path(job_posting_urls_file),
        Path(job_requirements_json_file),
    )

    # check if job posting json file exists or not
    if not job_posting_urls_file.exists():
        raise FileNotFoundError(
            f"Job posting URLs file not found: {job_posting_urls_file}"
        )

    # * Step 1: Load job descriptions & requirements and find new URLs
    new_urls, missing_requirements_urls = find_new_urls(
        job_posting_urls_file=job_posting_urls_file,
        job_descriptions_file=job_descriptions_json_file,
        job_requirements_file=job_requirements_json_file,
    )

    # If no new URLs, exit early
    if not new_urls and not missing_requirements_urls:
        logger.info("No new or semi-new URLs found. Skipping processing...")
        return

    logger.info(
        f"Processing {len(new_urls)} new URLs and {len(missing_requirements_urls)} semi-new URLs."
    )

    # * Step 2: Process URLs asynchronously

    # Set semaphore (limit no. of concurrent workers) & lock
    semaphore = asyncio.Semaphore(
        MAX_CONCURRENT_REQUESTS
    )  # Limit concurrency to max requests
    lock = (
        asyncio.Lock()
    )  # Create an async lock for file writes to avoid race conditions

    new_url_tasks = await process_urls_async(
        new_urls,
        job_descriptions_json_file,
        job_requirements_json_file,
        llm_provider,
        model_id,
        max_tokens,
        temperature,
        web_scrape=True,
        lock=lock,
        semaphore=semaphore,
    )

    semi_new_url_tasks = await process_urls_async(
        missing_requirements_urls,
        job_descriptions_json_file,
        job_requirements_json_file,
        llm_provider,
        model_id,
        max_tokens,
        temperature,
        web_scrape=False,
        lock=lock,
        semaphore=semaphore,
    )

    # Await the completion of all tasks (URLs processed)
    await asyncio.gather(*new_url_tasks, *semi_new_url_tasks)

    logger.info("Preprocessing pipeline finished.")
