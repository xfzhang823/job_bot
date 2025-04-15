"""
The pipeline module contains all high-level functions.
"""

# Import libraries
from pathlib import Path
import json
import logging
from typing import List, Union, Optional
from pydantic import RootModel
import asyncio
import pandas as pd
import nest_asyncio
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

from db_io.db_insert import insert_df_dedup
from db_io.db_transform import flatten_model_to_df, add_ingestion_metadata
from db_io.schema_definitions import PipelineStage, PipelineStatus, TableName

from fsm.fsm_utils import advance_fsm_for_url
from fsm.pipeline_fsm_manager import PipelineFSMManager

from models.resume_job_description_io_models import (
    JobSiteResponse,
    RequirementsResponse,
    JobPostingsBatch,
    ExtractedRequirementsBatch,
)
from models.llm_response_models import (
    JobSiteData,
    NestedRequirements,
    JobSiteResponse,
    RequirementsResponse,
)


# from utils.pydantic_model_loaders_from_files import load_job_posting_urls_file_model
from utils.pydantic_model_loaders_from_db import (
    load_job_postings_for_url_from_db,
    load_all_extracted_requirements_model_from_db,
    load_all_job_postings_file_model_from_db,
)
from project_config import (
    OPENAI,
    ANTHROPIC,
    GPT_4_TURBO,
    GPT_35_TURBO,
    CLAUDE_HAIKU,
    CLAUDE_SONNET,
)

# Set up logging
logger = logging.getLogger(__name__)

# Define the maximum number of concurrent tasks (limit this to not hit the API rate limit)
MAX_CONCURRENT_REQUESTS = 3  # Based on your rate limit or desired concurrency level

PipelineStage.JOB_POSTINGS  # web scrape + json parse
PipelineStage.EXTRACTED_REQUIREMENTS  # requirement extraction


async def fetch_job_description(
    url: str,
    fallback_file: Optional[Path],
    llm_provider: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    web_scrape: bool,
) -> dict:
    if web_scrape:
        return await process_webpages_to_json_async(
            urls=url,
            llm_provider=llm_provider,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif fallback_file and fallback_file.exists():
        return read_from_json_file(fallback_file).get(url, {}) or {}
    else:
        logger.warning(f"⚠️ No fallback file found or file does not exist for {url}")
        return {}


async def extract_persist_requirements_async(
    url: str,
    job_posting_model: JobSiteResponse,
    fsm_manager: PipelineFSMManager,
    model_id: str = GPT_4_TURBO,
    llm_provider: str = OPENAI,
) -> Optional[RequirementsResponse]:
    """
    Extracts structured job requirements from a validated JobSiteResponse model
    using an LLM, and persists the result to DuckDB.

    This function performs:
    - JSON serialization of the job posting
    - Requirement extraction via OpenAI or Anthropic
    - Validation + enrichment of the response model
    - Insertion into the `extracted_requirements` DuckDB table
    - Pipeline FSM status updates

    Args:
        - url (str): Job posting URL.
        - job_posting_model (JobSiteResponse): Parsed job description model.
        - fsm_manager (PipelineFSMManager): Pipeline state manager.
        - model_id (str): LLM model ID (e.g., "gpt-4-turbo").
        - llm_provider (str): LLM provider name ("openai" or "anthropic").

    Returns:
        Optional[RequirementsResponse]: The validated response model if successful,
        or None if an LLM or persistence error occurred.
    """
    logger.info(f"📄 Extracting requirements for URL: {url}")
    fsm = fsm_manager.get_fsm(url)

    try:
        job_desc_json = json.dumps(job_posting_model.data, indent=2)

        if llm_provider == OPENAI:
            req_model = await extract_job_requirements_with_openai_async(
                job_desc_json, model_id
            )
        elif llm_provider == ANTHROPIC:
            req_model = await extract_job_requirements_with_anthropic_async(
                job_desc_json, model_id
            )
        else:
            raise ValueError(f"Unsupported provider: {llm_provider}")

    except Exception as e:
        logger.exception(f"❌ LLM failed to extract requirements for {url}")
        fsm.mark_status(PipelineStatus.ERROR, notes="LLM extraction failed")
        return None

    # ✅ Validate the response content
    if not req_model.data or not any(dict(req_model.data).values()):
        req_model.status = "error"
        req_model.message = "No requirements extracted from LLM output"
        logger.warning(f"⚠️ Empty requirements extracted for {url}")
    else:
        req_model.status = "success"
        req_model.message = "Requirements extracted successfully"

    try:
        req_batch = ExtractedRequirementsBatch(root={url: req_model})  # type: ignore[arg-type]
        req_df = flatten_model_to_df(
            model=req_batch,
            table_name=TableName.EXTRACTED_REQUIREMENTS,
            stage=PipelineStage.EXTRACTED_REQUIREMENTS,
        )
        insert_df_dedup(req_df, TableName.EXTRACTED_REQUIREMENTS.value)

        # ✅ Update pipeline control
        fsm.mark_status(
            PipelineStatus.IN_PROGRESS,
            notes=req_model.message,
        )
        fsm.step()

        logger.info(
            f"✅ Requirements pipeline complete for {url} — status: {req_model.status}"
        )
        return req_model

    except Exception as e:
        logger.exception(f"💾 Failed to persist requirements for URL: {url}")
        fsm.mark_status(
            PipelineStatus.ERROR,
            notes="Failed to persist extracted requirements",
        )
        return None


async def run_extracted_requirements_stage_async(
    url: str,
    fsm: PipelineFSM,
    job_posting_model: JobSiteResponse,
    **kwargs,
) -> Optional[RequirementsResponse]:
    """
    FSM wrapper for extracted_requirements stage. Uses existing extract_requirements_from_job_posting_async().
    """
    try:
        # 🔁 Use your existing per-URL processor
        requirements = await extract_requirements_from_job_posting_async(
            job_description_url=url,
            job_description_model=job_posting_model,
            **kwargs,
        )

        fsm.step()
        save_pipeline_state_to_duckdb(fsm.state_model)
        return requirements

    except Exception as e:
        logger.warning(f"❌ Failed to process requirements for {url}: {e}")
        return None


async def process_single_url_async(
    job_description_url: str,
    lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,  # limit # of concurrent worker (avoid rate limit)
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_TURBO,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    web_scrape: bool = True,  # Flag to control whether to need to webscrape the content or not
) -> None:
    """
    Asynchronously processes a single job posting URL, fetches the job description
    (with or without web scraping), and extracts job requirements using GPT or other LLMs.

    Args:
        - job_description_url (str): The job description URL to process.
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
            job_description_dict = await process_webpages_to_json_async(
                urls=job_description_url,
                llm_provider=llm_provider,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )  # Returns a dict: Dict[str, Dict[str, Any]]
            logger.info(f"Job description fetched for URL: {job_description_url}")
            job_postings_model = JobPostingsBatch.model_validate(job_description_dict)  # type: ignore[attr-defined]

        else:
            # Step 1b: If not scraping, read existing data from the job_description_url
            # (assumed to be available)
            job_description_model = load_job_postings_for_url_from_db(
                url=job_description_url
            )
            job_description_dict = job_description_model.model_dump()  # type: ignore[attr-defined]
            logger.info(f"Job description fetched for URL: {job_description_url}")

        logger.info(job_description_dict)  # debug

        # Gatekeeper: if no meaningful content was extracted, skip further processing.
        # Unwrap the outer dict: expect a single URL key
        if isinstance(job_description_dict, dict) and len(job_description_dict) == 1:
            first_val = next(iter(job_description_dict.values()))
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
                f"Unexpected format for job_description_dict: {job_description_dict}"
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
            requirements_dict = await extract_job_requirements_with_openai_async(
                job_description=job_description_json, model_id=model_id
            )
        elif llm_provider.lower() == ANTHROPIC:
            requirements_dict = await extract_job_requirements_with_anthropic_async(
                job_description=job_description_json, model_id=model_id
            )
        else:
            raise ValueError(f"{llm_provider} is not a supported API.")

        logger.info(f"Job Requirements: \n{requirements_dict}")

        # Use the lock when inserting to DuckDB to avoid race conditions
        async with lock:
            # Load existing records to check for duplicates
            existing_job_descriptions = load_all_job_postings_file_model_from_db().root  # type: ignore[attr-defined]
            existing_job_requirements = load_all_extracted_requirements_model_from_db().root  # type: ignore[attr-defined]

            # Step 1: Insert job posting if new
            if job_description_url not in existing_job_descriptions:
                jobsite_model = job_postings_model.root[job_description_url]
                job_df = flatten_model_to_df(
                    model=jobsite_model,
                    table_name=TableName.JOB_POSTINGS,
                    stage=PipelineStage.PREPROCESSING,
                )
                insert_df_dedup(job_df, TableName.JOB_POSTINGS.value)

            # Step 2: Insert requirements if new
            if job_description_url not in existing_job_requirements:

                # Validate the raw dict into a RequirementsResponse
                req_model = RequirementsResponse.model_validate(requirements_dict)

                # Wrap it in a batch model
                req_batch = ExtractedRequirementsBatch(
                    root={job_description_url: req_model}
                )  # type: ignore[arg-type]

                req_df = flatten_model_to_df(
                    model=req_batch,
                    table_name=TableName.EXTRACTED_REQUIREMENTS,
                    stage=PipelineStage.PREPROCESSING,
                )

                insert_df_dedup(req_df, TableName.EXTRACTED_REQUIREMENTS.value)

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


async def run_preprocessing_pipeline_async_fsm(
    job_posting_urls_file: Union[Path, str],
    job_descriptions_json_file: Union[Path, str],
    job_requirements_json_file: Union[Path, str],
    llm_provider: str = "openai",  # llm_provider is passed from the orchestrating function
    model_id: str = GPT_4_TURBO,  # default to gpt 4 turbo
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

    # Step 3: Convert to validated models and insert into DuckDB
    jobsite_model = validate_model_or_none(
        JobSiteResponse, job_description_dict[job_description_url]
    )
    if jobsite_model:
        job_postings_df = flatten_model_to_df(
            model=jobsite_model,
            table_name=TableName.JOB_POSTINGS,
            source_file="pipeline",
            stage=PipelineStage.PREPROCESSING,
        )
        job_postings_df = add_ingestion_metadata(
            df=job_postings_df,
            source_file="pipeline",
            stage=PipelineStage.PREPROCESSING,
            table=TableName.JOB_POSTINGS,
            iteration=0,
            version="original",
            llm_provider=llm_provider,
        )
        insert_df_dedup(job_postings_df, table_name=TableName.JOB_POSTINGS.value)

    reqs_model = validate_model_or_none(RequirementsResponse, requirements_dict)
    if reqs_model:
        extracted_df = flatten_model_to_df(
            model=reqs_model,
            table_name=TableName.EXTRACTED_REQUIREMENTS,
            source_file="pipeline",
            stage=PipelineStage.PREPROCESSING,
        )
        extracted_df = add_ingestion_metadata(
            df=extracted_df,
            source_file="pipeline",
            stage=PipelineStage.PREPROCESSING,
            table=TableName.EXTRACTED_REQUIREMENTS,
            iteration=0,
            version="original",
            llm_provider=llm_provider,
        )
        insert_df_dedup(extracted_df, table_name=TableName.EXTRACTED_REQUIREMENTS.value)

        # Step 4: Advance FSM state for this URL (FSM-aware version)
    advance_fsm_for_url(
        url=job_description_url,
        llm_provider=llm_provider,
        iteration=0,
        version="original",
    )

    logger.info("Preprocessing pipeline finished.")
