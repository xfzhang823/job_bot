"""
pipelines_with_fsm/flatten_requirements_pipeline_async_fsm.py

FSM-aware, DuckDB-native stage:
    EXTRACTED_REQUIREMENTS  --(flatten)-->  FLATTENED_REQUIREMENTS

Goal
-----
For each URL at stage EXTRACTED_REQUIREMENTS/NEW:
  * Load structured `RequirementsResponse` from DuckDB.
  * Flatten nested categories/lists into a tidy, row-per-requirement table.
  * Insert into DuckDB `flattened_requirements` with standard metadata.
  * Advance FSM: mark current COMPLETED → step() to FLATTENED_REQUIREMENTS → mark NEW.

No filesystem dependency. Safe for high concurrency (no LLM/network calls).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional, List

# Enums / pipeline metadata
from job_bot.db_io.pipeline_enums import (
    PipelineStage,
    PipelineStatus,
    TableName,
    Version,
)

# FSM
from job_bot.fsm.pipeline_fsm_manager import PipelineFSMManager

# Worklist + IO
from job_bot.db_io.db_utils import get_urls_by_stage_and_status
from job_bot.db_io.db_transform import add_metadata
from job_bot.db_io.db_inserters import insert_df_with_config
from job_bot.db_io.flatten_and_rehydrate import flatten_extracted_requirements_to_table

# Optional control-plane sync (best-effort, zero-arg whole-table sync)
from job_bot.fsm.pipeline_control_sync import (
    sync_job_urls_to_pipeline_control,  # type: ignore
)


# from job_bot.db_io.db_loaders import (
#     load_extracted_requirements_for_url_from_db as _load_requirements_single,
# )
# from job_bot.db_io.db_loaders import (
#     load_all_extracted_requirements_from_db as _load_requirements_all,
# )

# Models
from job_bot.models.resume_job_description_io_models import (
    RequirementsResponse,
    ExtractedRequirementsBatch,
)


logger = logging.getLogger(__name__)


async def flatten_and_persist_requirements_for_url(
    url: str,
    *,
    fsm_manager: PipelineFSMManager,
    semaphore: asyncio.Semaphore,
    iteration: int | None = None,
) -> bool:
    """
    Flatten extracted requirements for a single URL and persist them (DuckDB + FSM).

    Overview
    --------
    This function performs the "flatten" stage of the pipeline for one job posting:
      - Guards on FSM stage: only runs if the URL is at `EXTRACTED_REQUIREMENTS`.
      - Marks the control row as IN_PROGRESS.
      - Loads the URL’s `RequirementsResponse` from DuckDB.
      - Validates that at least one non-empty requirement exists.
      - Flattens nested categories into a row-per-requirement DataFrame via
        `flatten_extracted_requirements_to_table`.
      - Stamps metadata (stage, table, version, iteration, etc.) with `add_metadata`.
      - Inserts rows into `flattened_requirements` with de-duplication.
      - Advances the FSM: `COMPLETED` → `step()` to `FLATTENED_REQUIREMENTS` → `NEW`.
      - Optionally performs best-effort control sync.

    Concurrency
    -----------
    The entire operation is wrapped by the provided `semaphore`. Since this stage
    does no LLM or network I/O, higher concurrency (default 8) is generally safe.

    Error Handling
    --------------
    - Any failure (load/validate/insert/FSM) logs, marks the URL `ERROR`, does **not**
      advance the FSM, and returns False.
    - On success, the FSM advances, and the function returns True.

    Args:
        url: Canonical job posting URL.
        fsm_manager: Shared FSM manager used to read and update pipeline state.
        semaphore: Async semaphore used to bound concurrency.

    Returns:
        bool: True on success; False if skipped or failed.
    """
    async with semaphore:
        fsm = fsm_manager.get_fsm(url)

        if fsm.get_current_stage() != PipelineStage.EXTRACTED_REQUIREMENTS.value:
            logger.info("⏩ Skipping %s; current stage is %s", url, fsm.state)
            return False

        # IN_PROGRESS at current stage
        fsm.mark_status(PipelineStatus.IN_PROGRESS, notes="Flattening requirements…")

        # Load extracted requirements
        try:
            reqs_model = _load_extracted_requirements_for_url(url)
            if reqs_model is None:
                raise ValueError("No extracted requirements found for URL")
        except Exception:
            logger.exception("❌ Failed to load extracted requirements for %s", url)
            fsm.mark_status(
                PipelineStatus.ERROR, notes="Load extracted_requirements failed"
            )
            return False

        # Validate content
        if not _validate_requirements_model(reqs_model):
            logger.warning("🚫 Skipping %s — empty/invalid requirements.", url)
            fsm.mark_status(PipelineStatus.ERROR, notes="Empty/invalid requirements")
            return False

        # Flatten + persist
        try:
            batch = ExtractedRequirementsBatch({url: reqs_model})
            df = flatten_extracted_requirements_to_table(batch)

            insert_df_with_config(df, TableName.FLATTENED_REQUIREMENTS)
        except Exception:
            logger.exception("❌ Failed to persist flattened requirements for %s", url)
            fsm.mark_status(PipelineStatus.ERROR, notes="DB insert failed")
            return False

        # Advance FSM
        try:
            fsm.mark_status(PipelineStatus.COMPLETED, notes="Flattened → DB")
            fsm.step()  # EXTRACTED_REQUIREMENTS → FLATTENED_REQUIREMENTS
            fsm.mark_status(PipelineStatus.NEW, notes="Ready for next stage")

            # Optional whole-table sync; zero-arg call is correct
            if sync_job_urls_to_pipeline_control:
                try:
                    sync_job_urls_to_pipeline_control()  # ✅ no args
                except Exception:
                    logger.warning("⚠️ Control sync failed for %s (non-fatal)", url)

            logger.info("✅ Flatten requirements complete for %s", url)
            return True
        except Exception:
            logger.exception("❌ FSM transition failed for %s", url)
            fsm.mark_status(PipelineStatus.ERROR, notes="FSM transition failed")
            return False


async def process_flattened_requirements_batch_async_fsm(
    urls: List[str],
    *,
    max_concurrent: int = 8,
) -> list[asyncio.Task]:
    """Kick off bounded-concurrency tasks for a list of URLs."""
    semaphore = asyncio.Semaphore(max_concurrent)
    fsm_manager = PipelineFSMManager()

    async def _run_one(u: str) -> None:
        await flatten_and_persist_requirements_for_url(
            u, fsm_manager=fsm_manager, semaphore=semaphore
        )

    return [asyncio.create_task(_run_one(u)) for u in urls]


async def run_flattened_requirements_pipeline_async_fsm(
    *,
    filter_urls: Optional[list[str]] = None,
    max_concurrent: int = 8,
) -> None:
    """
    FSM-aware entrypoint for flattening extracted job requirements and persisting
    them into DuckDB.

    What It Does
    ------------
    - Queries the `pipeline_control` table for all job URLs currently at
      stage = `EXTRACTED_REQUIREMENTS` with status = `NEW`.
    - Optionally filters this list to a caller-provided subset (`filter_urls`).
    *- For each eligible URL:
        * Loads the structured `RequirementsResponse` from DuckDB
        *   (`extracted_requirements`).
        * Validates that at least one requirement string exists.
        * Flattens the nested requirement categories into a row-per-requirement table
        *  via `flatten_extracted_requirements_to_table`.
        * Stamps standard metadata (stage, iteration, version, etc.).
        * Inserts deduplicated rows into `flattened_requirements`.
        * Advances the FSM: mark current stage `COMPLETED` →
        *  step() to `FLATTENED_REQUIREMENTS` →
        *  mark new stage `NEW`.
    - Runs all URLs concurrently, bounded by `max_concurrent`.

    Concurrency
    -----------
    - No network/LLM calls; safe to set higher concurrency (default = 8).
    - A shared `PipelineFSMManager` ensures consistent state transitions across tasks.

    Error Handling
    --------------
    - If no URLs match, the function logs and returns early.
    - For individual URLs:
        * Any load/validation/insert/FSM error logs, marks status = `ERROR`,
          does not advance, and continues with other URLs.
    - The pipeline completes even if some URLs fail.

    Args:
        filter_urls (Optional[list[str]]): Subset of URLs to process. If None,
            all matching URLs at the stage are processed.
        max_concurrent (int): Maximum number of concurrent tasks. Defaults to 8.

    Returns:
        None. Side effects include writing rows to DuckDB and updating the
        `pipeline_control` FSM state.

    Example:
        >>> await run_flattened_requirements_pipeline_async_fsm(
        ...     filter_urls=["https://job.com/abc123"],
        ...     max_concurrent=16,
        ... )
        # Flattens and persists requirements for the provided URL(s).
    """

    urls = get_urls_by_stage_and_status(
        stage=PipelineStage.EXTRACTED_REQUIREMENTS,
        status=PipelineStatus.NEW,
    )

    if filter_urls:
        urls = [u for u in urls if u in filter_urls]

    if not urls:
        logger.info("📭 No URLs to flatten at 'extracted_requirements' stage.")
        return

    logger.info("🧱 Flattening requirements for %d URL(s)…", len(urls))

    tasks = await process_flattened_requirements_batch_async_fsm(
        urls=urls,
        max_concurrent=max_concurrent,
    )

    await asyncio.gather(*tasks)

    logger.info("✅ Finished flattened requirements FSM pipeline.")


def _validate_requirements_model(model: RequirementsResponse) -> bool:
    """Return True if the model contains at least one requirement string."""
    try:
        data_dict = model.data.model_dump()  # categories -> list[str]
        if not isinstance(data_dict, dict) or not data_dict:
            return False
        for items in data_dict.values():
            if isinstance(items, list) and any(
                isinstance(x, str) and x.strip() for x in items
            ):
                return True
        return False
    except Exception:
        return False


def _load_extracted_requirements_for_url(url: str) -> Optional[RequirementsResponse]:
    """
    Load the extracted requirements for a single URL.
    Contract: return ONE `RequirementsResponse` (or None on error/miss).
    """
    if _load_requirements_single is None:
        raise RuntimeError("Per-URL loader is not available")
    try:
        return _load_requirements_single(url=url)
        # If needed later:
        # return _load_requirements_single(url=url, status=PipelineStatus.NEW, iteration=0)
    except Exception:
        return None
