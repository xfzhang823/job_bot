"""
pipelines_with_fsm/extract_job_requirements_pipeline_async_fsm.py

* Job Description (DB) → Requirements (DB)

FSM-aware, DuckDB-native pipeline to extract structured job requirements from
job_postings using LLMs. No filesystem dependency.

Stage transition intent:
    JOB_POSTINGS  --(extract)-->  EXTRACTED_REQUIREMENTS

Control-table semantics per URL:
- When starting this stage from JOB_POSTINGS, mark that stage IN_PROGRESS.
- On success: mark JOB_POSTINGS COMPLETE, step() to EXTRACTED_REQUIREMENTS,
  then mark NEW for the new stage (so the next stage runner can pick it up).
- On failure: mark ERROR and do not advance.
"""

from __future__ import annotations

# Standard libraries
import asyncio
import json
import logging
from typing import List, Optional

# Project-level imports (keep paths consistent with your repo layout)
from job_bot.db_io.db_inserters import insert_df_with_config
from job_bot.db_io.flatten_and_rehydrate import flatten_model_to_df
from job_bot.db_io.pipeline_enums import PipelineStage, PipelineStatus, TableName
from job_bot.db_io.state_sync import load_pipeline_state
from job_bot.db_io.db_utils import get_urls_ready_for_transition
from job_bot.db_io.db_loaders import load_table

from job_bot.fsm.pipeline_fsm_manager import PipelineFSMManager

from job_bot.models.llm_response_models import JobSiteResponse  # data: JobSiteData
from job_bot.models.resume_job_description_io_models import (
    ExtractedRequirementsBatch,
    RequirementsResponse,
)
from job_bot.preprocessing.extract_requirements_with_llms_async import (
    extract_job_requirements_with_openai_async,
    extract_job_requirements_with_anthropic_async,
)

# Control-plane sync after each URL
from job_bot.fsm.pipeline_control_sync import (
    sync_job_postings_to_pipeline_control,
)

from job_bot.config.project_config import OPENAI, GPT_4_1_NANO, ANTHROPIC, CLAUDE_HAIKU


logger = logging.getLogger(__name__)


async def extract_and_persist_requirements_for_url(
    url: str,
    *,
    fsm_manager: PipelineFSMManager,
    semaphore: asyncio.Semaphore,
    llm_provider: str = OPENAI,  # Needed for operations
    model_id: str = GPT_4_1_NANO,  # Needed for operations
) -> Optional[RequirementsResponse]:
    """
    Orchestrate the requirements extraction for a single job URL (FSM + DuckDB,
    no filesystem).

    This function:
      Marks the control row `IN_PROGRESS`.
      Loads the structured job description for `url` from DuckDB (`job_postings`).
      Validates/serializes the description into a JSON string for LLM input.
      Uses a bounded semaphore to call the LLM (via `_extract_job_requirements`) and
         returns a typed `RequirementsResponse`.
      Flattens and inserts the result into DuckDB (`extracted_requirements`) with
         de-duplication.
      Advances the FSM: mark current stage `COMPLETED` → `step()` to
         `EXTRACTED_REQUIREMENTS` → mark new stage `NEW`.
      Optionally syncs control-plane views (`sync_job_postings_to_pipeline_control`)
         if available.

    Side effects:
      - Writes to DuckDB table: `extracted_requirements`.
      - Updates `pipeline_control` status and performs a stage transition.
      - Makes a network call to the selected LLM provider.
      - Logs progress and errors.

    Concurrency:
      - LLM calls are rate-limited by `semaphore`.
      - Safe to run across many URLs concurrently when each task shares a
        process-wide `PipelineFSMManager`.

    Idempotency & behavior:
      - If the URL is no longer at `JOB_POSTINGS`, the function no-ops and
        returns `None`.
      - On any error (load/validate/LLM/DB/FSM), the function logs, marks `ERROR`,
        **does not advance** the FSM, and returns `None`.
      - On success, returns the `RequirementsResponse`.

    Args:
      url: Canonical job URL key.
      fsm_manager: Manager used to fetch and update the URL's FSM state.
      semaphore: Concurrency limiter for the LLM call.
      model_id: LLM model identifier (e.g., `GPT_4_1_NANO`).
      llm_provider: LLM provider name (e.g., `OPENAI`, `ANTHROPIC`).

    Returns:
      RequirementsResponse on success; otherwise `None`.
    """
    logger.info(f"🧠 Starting requirements extraction for: {url}")
    fsm = fsm_manager.get_fsm(url)

    # Mark the current stage as in-progress in the control table
    fsm.mark_status(PipelineStatus.IN_PROGRESS, notes="Extracting requirements…")

    # Resolve iteration once for loading the source posting (stamping is handled at insert time)
    state = load_pipeline_state(url)
    iter_value = getattr(state, "iteration", 0)

    # Load job posting from DuckDB
    try:
        postings_batch = load_table(
            TableName.JOB_POSTINGS, url=url, iteration=iter_value
        )
        job_posting: JobSiteResponse = postings_batch.root[url]  # type: ignore[attr-defined]
    except Exception:
        logger.exception(f"❌ Failed to load job posting for {url}")
        fsm.mark_status(PipelineStatus.ERROR, notes="Load job_postings failed")
        return None

    # Validate content (extra guard)
    job_description_json = _validate_job_content(job_posting)
    if not job_description_json:
        logger.warning(f"🚫 Skipping {url} — empty or invalid job content.")
        fsm.mark_status(PipelineStatus.ERROR, notes="Empty/invalid job content")
        return None

    # LLM extraction (concurrency limited)
    try:
        async with semaphore:
            requirements_model = await _extract_job_requirements(
                job_description_json=job_description_json,
                llm_provider=llm_provider,
                model_id=model_id,
            )
    except Exception:
        logger.exception(f"❌ LLM extraction failed for {url}")
        fsm.mark_status(PipelineStatus.ERROR, notes="LLM extraction failed")
        return None

    # Flatten → insert (config stamps metadata + applies dedup)
    try:
        batch = ExtractedRequirementsBatch({url: requirements_model})
        df = flatten_model_to_df(
            model=batch, table_name=TableName.EXTRACTED_REQUIREMENTS
        )

        if df is None:
            raise ValueError("Flatten failed, got None instead of DataFrame")

        insert_df_with_config(
            df,
            TableName.EXTRACTED_REQUIREMENTS,
            url=url,  # lets the inserter resolve/stamp iteration per YAML
            llm_provider=llm_provider,  # stamped only if YAML requests it
            model_id=model_id,  # stamped only if YAML requests it
        )
    except Exception:
        logger.exception(f"❌ Failed to insert extracted requirements for {url}")
        fsm.mark_status(PipelineStatus.ERROR, notes="DB insert failed")
        return None

    # Transition control-plane
    try:
        # Complete current stage, then step to the next stage
        fsm.mark_status(
            PipelineStatus.COMPLETED, notes="Requirements saved to DB"
        )  # Not needed; keeping only for the notes.
        fsm.step()  # JOB_POSTINGS -> EXTRACTED_REQUIREMENTS

        # Optional sync pass to keep pipeline_control denormalized views fresh
        if sync_job_postings_to_pipeline_control:
            try:
                sync_job_postings_to_pipeline_control()
            except Exception:
                logger.warning(f"⚠️ Control sync failed for {url} (non-fatal)")

        logger.info(f"✅ Completed extraction for {url}")
        return requirements_model
    except Exception:
        logger.exception(f"❌ Failed to step/mark FSM for {url}")
        fsm.mark_status(PipelineStatus.ERROR, notes="FSM transition failed")
        return None


async def process_extracted_requirements_batch_async_fsm(
    urls: List[str],
    *,
    llm_provider: str,
    model_id: str,
    no_of_concurrent_workers: int = 5,
) -> list[asyncio.Task]:
    """Kick off tasks for a list of URLs with a bounded semaphore."""
    semaphore = asyncio.Semaphore(no_of_concurrent_workers)
    fsm_manager = PipelineFSMManager()  # reuse across tasks

    async def run_one(url: str) -> None:
        # Having a state row isn't strictly required for operation, but warn
        if load_pipeline_state(url) is None:
            logger.warning(f"⚠️ No pipeline_state row for {url}")
        await extract_and_persist_requirements_for_url(
            url,
            fsm_manager=fsm_manager,
            semaphore=semaphore,
            llm_provider=llm_provider,
            model_id=model_id,
        )

    return [asyncio.create_task(run_one(u)) for u in urls]


async def run_extract_job_requirements_pipeline_async_fsm(
    *,
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_1_NANO,
    max_concurrent_tasks: int = 5,
    filter_keys: Optional[list[str]] = None,
    # iteration: int = 0,
) -> None:
    """Entry point: process all URLs at JOB_POSTINGS stage with status NEW.

    filter_keys: if provided, limit to this subset of URLs.
    """
    urls = get_urls_ready_for_transition(
        stage=PipelineStage.EXTRACTED_REQUIREMENTS,
    )

    logger.info(f"urls from get_urls_ready: {urls}")

    if filter_keys:
        urls = [u for u in urls if u in filter_keys]
        logger.info(f"urls to process: {urls}")

    if not urls:
        logger.info("📭 No job URLs to process at 'extracted_requirement' stage.")
        return

    logger.info(f"🧪 Extracting requirements for {len(urls)} URL(s)…")

    tasks = await process_extracted_requirements_batch_async_fsm(
        urls=urls,
        llm_provider=llm_provider,
        model_id=model_id,
        no_of_concurrent_workers=max_concurrent_tasks,
    )

    await asyncio.gather(*tasks)

    logger.info("✅ Finished job requirements FSM pipeline.")


async def _extract_job_requirements(
    *, job_description_json: str, llm_provider: str, model_id: str
) -> RequirementsResponse:
    """Dispatch to provider-specific async extraction."""
    provider = llm_provider.lower()
    if provider == OPENAI:
        return await extract_job_requirements_with_openai_async(
            job_description=job_description_json, model_id=model_id
        )
    if provider == ANTHROPIC:
        return await extract_job_requirements_with_anthropic_async(
            job_description=job_description_json, model_id=model_id
        )
    raise ValueError(f"Unsupported LLM provider: {llm_provider}")


def _validate_job_content(job_posting: JobSiteResponse) -> Optional[str]:
    """
    Return pretty JSON payload if content exists and is non-empty; else None.

    NOTE:
      - `job_posting.data` is a Pydantic model (JobSiteData), not a dict.
      - We serialize the whole `data` section so the LLM sees title/company/etc.
    """
    data_model = job_posting.data  # JobSiteData
    content = data_model.content

    if not isinstance(content, dict):
        return None

    # Non-empty if any string field has non-blank content
    any_text = any((v.strip() for v in content.values() if isinstance(v, str)))
    if not any_text:
        return None

    # Pretty JSON string for LLM input
    return json.dumps(data_model.model_dump(), indent=2)
