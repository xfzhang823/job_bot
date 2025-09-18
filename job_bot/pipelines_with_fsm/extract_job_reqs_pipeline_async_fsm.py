"""
pipelines_with_fsm/job_requirements_pipeline_async_fsm.py

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
from db_io.db_insert import insert_df_dedup
from db_io.db_transform import flatten_model_to_df
from db_io.pipeline_enums import PipelineStage, PipelineStatus, TableName
from db_io.state_sync import load_pipeline_state
from db_io.db_utils import get_urls_by_stage_and_status
from fsm.pipeline_fsm_manager import PipelineFSMManager
from utils.pydantic_model_loaders_from_db import load_job_postings_for_url_from_db

from models.llm_response_models import JobSiteResponse
from models.resume_job_description_io_models import (
    ExtractedRequirementsBatch,
    RequirementsResponse,
)
from project_config import OPENAI, GPT_4_1_NANO, ANTHROPIC, CLAUDE_HAIKU
from preprocessing.extract_requirements_with_llms_async import (
    extract_job_requirements_with_openai_async,
    extract_job_requirements_with_anthropic_async,
)

# Optional (best-effort) control-plane sync after each URL
# keep import optional so this file is self-contained if sync module moves
from pipelines_with_fsm.state_orchestration_pipeline import sync_job_postings_to_pipeline_control  # type: ignore


logger = logging.getLogger(__name__)


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
    """Return pretty JSON payload if content exists and is non-empty; else None."""
    data_section = job_posting.data.model_dump()
    content = data_section.get("content")

    if not isinstance(content, dict):
        return None
    # Non-empty if any string field has non-blank content
    any_text = any((v.strip() for v in content.values() if isinstance(v, str)))
    if not any_text:
        return None

    return json.dumps(data_section, indent=2)


async def extract_and_persist_requirements_for_url(
    url: str,
    *,
    fsm_manager: PipelineFSMManager,
    semaphore: asyncio.Semaphore,
    model_id: str = GPT_4_1_NANO,
    llm_provider: str = OPENAI,
) -> Optional[RequirementsResponse]:
    """
    Orchestrate the requirements extraction for a single job URL (FSM + DuckDB, no filesystem).

    This function:
      1) Guards on FSM: only runs when the URL's current stage is `JOB_POSTINGS`.
      2) Marks the control row `IN_PROGRESS`.
      3) Loads the structured job description for `url` from DuckDB (`job_postings`).
      4) Validates/serializes the description into a JSON string for LLM input.
      5) Uses a bounded semaphore to call the LLM (via `_extract_job_requirements`) and
         returns a typed `RequirementsResponse`.
      6) Flattens and inserts the result into DuckDB (`extracted_requirements`) with
         de-duplication.
      7) Advances the FSM: mark current stage `COMPLETED` → `step()` to
         `EXTRACTED_REQUIREMENTS` → mark new stage `NEW`.
      8) Optionally syncs control-plane views (`sync_job_postings_to_pipeline_control`)
         if available.

    Side effects:
      - Writes to DuckDB table: `extracted_requirements`.
      - Updates `pipeline_control` statuses and performs a stage transition.
      - Makes a network call to the selected LLM provider.
      - Logs progress and errors.

    Concurrency:
      - LLM calls are rate-limited by `semaphore`.
      - Safe to run across many URLs concurrently when each task shares a
        process-wide `PipelineFSMManager`.

    Idempotency & behavior:
      - If the URL is no longer at `JOB_POSTINGS`, the function no-ops and returns `None`.
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
    logger.info("🧠 Starting requirements extraction for: %s", url)
    fsm = fsm_manager.get_fsm(url)

    # Guard: only operate when current stage is JOB_POSTINGS
    if fsm.get_current_stage() != PipelineStage.JOB_POSTINGS.value:
        logger.info("⏩ Skipping %s; current stage is %s", url, fsm.state)
        return None

    # Mark the current stage as in-progress in the control table
    fsm.mark_status(PipelineStatus.IN_PROGRESS, notes="Extracting requirements…")

    # Load job posting from DuckDB
    try:
        job_description_model = load_job_postings_for_url_from_db(
            url=url,
            status=None,
            iteration=0,
        )
        job_posting: JobSiteResponse = job_description_model.root[url]  # type: ignore[attr-defined]
    except Exception:
        logger.exception("❌ Failed to load job posting for %s", url)
        fsm.mark_status(PipelineStatus.ERROR, notes="Load job_postings failed")
        return None

    job_description_json = _validate_job_content(job_posting)
    if not job_description_json:
        logger.warning("🚫 Skipping %s — empty or invalid job content.", url)
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
        logger.exception("❌ LLM extraction failed for %s", url)
        fsm.mark_status(PipelineStatus.ERROR, notes="LLM extraction failed")
        return None

    # Persist to DuckDB
    try:
        df = flatten_model_to_df(
            model=ExtractedRequirementsBatch(root={url: requirements_model}),  # type: ignore[attr-defined]
            table_name=TableName.EXTRACTED_REQUIREMENTS,
            stage=PipelineStage.EXTRACTED_REQUIREMENTS,
        )
        insert_df_dedup(df, TableName.EXTRACTED_REQUIREMENTS.value)
    except Exception:
        logger.exception("❌ Failed to insert extracted requirements for %s", url)
        fsm.mark_status(PipelineStatus.ERROR, notes="DB insert failed")
        return None

    # Transition control-plane
    try:
        # Complete current stage, then step to the next stage
        fsm.mark_status(PipelineStatus.COMPLETED, notes="Requirements saved to DB")
        fsm.step()  # JOB_POSTINGS -> EXTRACTED_REQUIREMENTS
        fsm.mark_status(PipelineStatus.NEW, notes="Ready for next stage")

        # Optional sync pass to keep pipeline_control denormalized views fresh
        if sync_job_postings_to_pipeline_control:
            try:
                sync_job_postings_to_pipeline_control([url])  # type: ignore[misc]
            except Exception:
                logger.warning("⚠️ Control sync failed for %s (non-fatal)", url)

        logger.info("✅ Completed extraction for %s", url)
        return requirements_model
    except Exception:
        logger.exception("❌ Failed to step/mark FSM for %s", url)
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
            logger.warning("⚠️ No pipeline_state row for %s", url)
        await extract_and_persist_requirements_for_url(
            url,
            fsm_manager=fsm_manager,
            semaphore=semaphore,
            llm_provider=llm_provider,
            model_id=model_id,
        )

    return [asyncio.create_task(run_one(u)) for u in urls]


async def run_job_requirements_pipeline_async_fsm(
    *,
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_1_NANO,
    max_concurrent_tasks: int = 5,
    filter_keys: Optional[list[str]] = None,
) -> None:
    """Entry point: process all URLs at JOB_POSTINGS stage with status NEW.

    filter_keys: if provided, limit to this subset of URLs.
    """
    urls = get_urls_by_stage_and_status(
        stage=PipelineStage.JOB_POSTINGS,
        status=PipelineStatus.NEW,
    )

    if filter_keys:
        urls = [u for u in urls if u in filter_keys]

    if not urls:
        logger.info("📭 No job URLs to process at 'job_postings' stage.")
        return

    logger.info("🧪 Extracting requirements for %d URL(s)…", len(urls))

    tasks = await process_extracted_requirements_batch_async_fsm(
        urls=urls,
        llm_provider=llm_provider,
        model_id=model_id,
        no_of_concurrent_workers=max_concurrent_tasks,
    )

    await asyncio.gather(*tasks)

    logger.info("✅ Finished job requirements FSM pipeline.")
