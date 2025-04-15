"""
job_postings_pipeline_async_fsm.py

This module implements the asynchronous FSM-aware pipeline for scraping,
parsing, validating, and persisting job posting webpages into DuckDB.

It is responsible for:
- Querying the pipeline_control table for jobs at the `job_urls` stage
- Concurrency-controlled processing of each job URL (scrape + LLM + validation)
- Advancing the pipeline state via a Finite State Machine (FSM)
- Storing parsed job data into the `job_postings` DuckDB table

Pipeline Stage:
    PipelineStage.JOB_URLS → PipelineStage.JOB_POSTINGS

FSM Controls:
    - Each job (URL) has an associated PipelineState
    - PipelineFSM manages allowed transitions and persists progress
    - FSM steps forward only after successful job scrape + persist

Concurrency:
    - Semaphore used to limit the number of concurrent scraping + LLM operations
    - Ensures safe and resource-efficient processing across batches

Dependencies:
    - process_webpages_to_json_async(): Core scraping + parsing logic
    - flatten_model_to_df(): Converts parsed model to DuckDB-ready DataFrame
    - insert_df_dedup(): Inserts deduplicated records into DuckDB
    - PipelineFSMManager: Controls per-URL state and stage progression

---
📦 Example Usage (in orchestrator):

    await run_job_postings_pipeline_async_fsm()

---
🔁 ASCII Flowchart:

+---------------------------+
| pipeline_control table    |
| (stage = 'job_urls')      |
+------------+--------------+
             |
             v
+---------------------------+
|  process_job_posting_batch_async_fsm |
|  - uses FSM                        |
|  - limits concurrency              |
+------------+--------------+
             |
             v
+---------------------------+
| scrape_parse_persist_job_posting_async |
| - scrape + LLM parse         |
| - validate content           |
| - flatten + insert to DB     |
| - fsm.step() if success      |
+------------+--------------+
             |
             v
+----------------------------+
| stage = 'job_postings'     |
| status = 'in_progress'     |
| persisted in DuckDB        |
+----------------------------+

---
"""

# Standard libraries
import asyncio
import logging
from typing import List, Optional

# Project-level imports
from db_io.db_insert import insert_df_dedup
from db_io.db_transform import flatten_model_to_df
from db_io.schema_definitions import PipelineStage, PipelineStatus, TableName
from db_io.state_sync import load_pipeline_state
from db_io.db_utils import get_urls_by_stage_and_status
from fsm.pipeline_fsm_manager import PipelineFSMManager

from models.resume_job_description_io_models import (
    JobPostingsBatch,
    JobSiteResponse,
)
from project_config import (
    OPENAI,
    GPT_4_TURBO,
)

from utils.webpage_reader_async import process_webpages_to_json_async


# Set logger
logger = logging.getLogger(__name__)


async def scrape_parse_persist_job_posting_async(
    url: str,
    fsm_manager: PipelineFSMManager,
    semaphore: asyncio.Semaphore,
    model_id: str = GPT_4_TURBO,
    llm_provider: str = OPENAI,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> Optional[JobSiteResponse]:
    """
    Scrapes, parses, validates, and persists a single job posting from the web.

    Args:
        url (str): Job posting URL.
        fsm_manager (PipelineFSMManager): Manager to track and control pipeline state.
        semaphore (asyncio.Semaphore): Concurrency limiter for I/O-intensive steps.
        model_id (str): LLM model to use.
        llm_provider (str): LLM provider ("openai", "anthropic", etc.).
        max_tokens (int): Max tokens for LLM generation.
        temperature (float): Sampling temperature.

    Returns:
        Optional[JobSiteResponse]: Parsed job posting model, or None on failure.
    """
    logger.info(f"🌐 Starting scrape/parse for: {url}")
    fsm = fsm_manager.get_fsm(url)

    # Step 1: Scrape + LLM parse (inside semaphore)
    try:
        async with semaphore:
            job_postings_batch_model = await process_webpages_to_json_async(
                urls=url,
                llm_provider=llm_provider,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )
    except Exception as e:
        logger.exception(f"❌ Scraping or LLM failed at URL: {url}")
        fsm.mark_status(PipelineStatus.ERROR, notes="Scraping or LLM call failed")
        return None

    # Step 2: Validation + Flattening + Persistence
    try:
        jobposting_model = job_postings_batch_model.root[url]  # type: ignore[attr-defined]
        data_section = jobposting_model.data
        content = data_section.get("content")

        # Validate content
        if not isinstance(content, dict) or not any(
            isinstance(v, str) and v.strip() for v in content.values()
        ):
            jobposting_model.status = "error"
            jobposting_model.message = "Empty or uninformative content extracted"
            logger.warning(f"🚫 Skipping URL due to empty content: {url}")
        else:
            jobposting_model.status = "success"
            jobposting_model.message = "Job description parsed successfully"

        # Flatten model and insert to DuckDB
        job_posting_batch = JobPostingsBatch(root={url: jobposting_model})  # type: ignore[arg-type]
        job_df = flatten_model_to_df(
            model=job_posting_batch,
            table_name=TableName.JOB_POSTINGS,
            stage=PipelineStage.JOB_POSTINGS,
        )
        insert_df_dedup(job_df, TableName.JOB_POSTINGS.value)

        # FSM Step + Status update
        fsm.step()
        fsm.mark_status(
            PipelineStatus.IN_PROGRESS,
            notes="Job posting scraped, parsed, and persisted to db.",
        )

        logger.info(
            f"✅ Scrape pipeline complete for {url} — status: {jobposting_model.status}"
        )
        return jobposting_model

    except Exception as e:
        logger.exception(f"💾 Failed to post-process or persist job posting: {url}")
        fsm.mark_status(PipelineStatus.ERROR, notes="Post-processing/persist failed")
        return None


async def process_job_posting_batch_async_fsm(
    urls: List[str],
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_TURBO,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    no_of_concurrent_workers: int = 5,
) -> List[asyncio.Task]:
    """
    Prepares a list of asyncio Tasks to scrape, parse, and persist job postings
    using FSM tracking.

    This function handles a batch of job posting URLs in parallel (with concurrency
    limits), verifies that each job is at the correct stage (`job_urls`) via
    a finite state machine (FSM), and delegates execution to a lower-level
    async function if eligible.

    Args:
        - urls (List[str]): List of job posting URLs to process.
        - llm_provider (str): LLM provider to use (e.g., "openai", "anthropic").
        - model_id (str): The model identifier (e.g., "gpt-4", "claude-haiku").
        - max_tokens (int): Maximum tokens for the LLM output.
        - temperature (float): Sampling temperature for LLM generation.
        - no_of_concurrent_workers (int): Maximum number of concurrent workers
        (semaphore limit).

    Returns:
        List[asyncio.Task]: A list of asyncio Tasks ready to be awaited via
        `asyncio.gather`.
    """
    semaphore = asyncio.Semaphore(no_of_concurrent_workers)

    async def run_one(url: str):
        state = load_pipeline_state(url)
        if state is None:
            logger.warning(f"⚠️ No pipeline state for URL: {url}")
            return

        fsm_manager = PipelineFSMManager()
        fsm = fsm_manager.get_fsm(url)

        if fsm.state != PipelineStage.JOB_URLS.value:
            logger.info(f"⏩ Skipping {url} — current stage: {fsm.state}")
            return

        await scrape_parse_persist_job_posting_async(
            url=url,
            fsm_manager=fsm_manager,
            semaphore=semaphore,
            llm_provider=llm_provider,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return [asyncio.create_task(run_one(url)) for url in urls]


async def run_job_postings_pipeline_async_fsm(
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_TURBO,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    max_concurrent_tasks: int = 5,
    filter_keys: Optional[list[str]] = None,
):
    """
    FSM-aware pipeline to scrape and parse job postings asynchronously.

    This function:
    - Retrieves job URLs with 'new' status at the 'job_urls' stage.
    - Filters by `filter_keys` if provided.
    - Uses a concurrency-controlled async batch runner to scrape and
    parse job content.
    - Updates FSM stages and status fields per job URL on success or failure.

    Args:
        - llm_provider (str): LLM provider ("openai" or "anthropic").
        - model_id (str): Model ID to use for parsing.
        - max_tokens (int): Max tokens for LLM call.
        - temperature (float): Sampling temperature.
        - max_concurrent_tasks (int): Max number of parallel workers.
        - filter_keys (Optional[list[str]]): Subset of URLs to process.
        If None, all matching URLs are processed.
    """

    urls = get_urls_by_stage_and_status(
        stage=PipelineStage.JOB_URLS,
        status=PipelineStatus.NEW,
    )

    if filter_keys:
        urls = [url for url in urls if url in filter_keys]

    if not urls:
        logger.info("📭 No job URLs to process at 'job_urls' stage.")
        return

    logger.info(f"🚀 Processing {len(urls)} job URLs at stage 'job_urls'...")

    tasks = await process_job_posting_batch_async_fsm(
        urls=urls,
        llm_provider=llm_provider,
        model_id=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        no_of_concurrent_workers=max_concurrent_tasks,
    )

    await asyncio.gather(*tasks)

    logger.info("✅ Finished job postings FSM pipeline.")
