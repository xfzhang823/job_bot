"""
pipelines_with_fsm/resume_editing_pipeline_async_fsm.py

DB-native Resume Editing Pipeline (FSM-driven; async; no filesystem I/O)

Inputs (per URL):
  - Worklist:  FLATTENED_RESPONSIBILITIES / NEW as default
  - Tables:    flattened_requirements        (requirement_key, requirement, url, …)
               flattened_responsibilities    (responsibility_key, responsibility, url, …)

Output (per URL):
    • edited_responsibilities / NEW
        Columns: url, responsibility_key, requirement_key, responsibility,
            iteration, version, llm_provider, stage, status, created_at, updated_at

Per-URL flow:
  1) FSM: mark IN_PROGRESS for source stage (EDITED_RESPONSIBILITIES).
  2) Load flattened responsibilities + flattened requirements (DuckDB).
  3) Run async LLM editor (→ ResponsibilityMatches).
  4) Flatten to row-per-(responsibility_key, requirement_key).
  5) INSERT into edited_responsibilities with standard metadata.
  6) FSM: mark COMPLETE on source; step() → SIM_METRICS_REVAL; mark NEW.

Notes:
  • Strictly DB-native: no JSON/CSV file I/O.
  • Uses your pydantic_model_loaders_from_db utilities for
    rehydration & validation.
  • Mirrors the structure of your similarity-metrics FSM pipeline
    (semaphore, gather).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence
import pandas as pd

# Pipeline enums / metadata
from job_bot.db_io.pipeline_enums import (
    PipelineStage,
    PipelineStatus,
    TableName,
    Version,
)

# FSM
from job_bot.fsm.pipeline_fsm_manager import PipelineFSMManager

# Worklist + IO
from job_bot.db_io.db_utils import get_urls_ready_for_transition
from job_bot.db_io.db_inserters import insert_df_with_config
from job_bot.db_io.db_loaders import load_table

# The async editor that rewrites responsibilities against requirements
from job_bot.evaluation_optimization.resumes_editing_async import (
    modify_multi_resps_based_on_reqs_async,
)

# Pyd models (for types)
from job_bot.models.resume_job_description_io_models import (
    ResponsibilityMatches,
    Responsibilities,
    Requirements,
)

# Configs
from job_bot.config.project_config import OPENAI, GPT_4_1_NANO, ANTHROPIC, CLAUDE_HAIKU

logger = logging.getLogger(__name__)


def get_resps_dict_strict(obj) -> Dict[str, str]:
    if isinstance(obj, Responsibilities):
        return dict(obj.responsibilities)
    raise TypeError(f"Expected Responsibilities, got {type(obj).__name__}")


def get_reqs_dict_strict(obj) -> Dict[str, str]:
    if isinstance(obj, Requirements):
        return dict(obj.requirements)
    raise TypeError(f"Expected Requirements, got {type(obj).__name__}")


# =============================================================================
# Per-URL worker
# =============================================================================
async def edit_and_persist_responsibilities_for_url(
    url: str,
    *,
    fsm_manager: PipelineFSMManager,
    semaphore: asyncio.Semaphore,
    # iteration: int | None = None,
    llm_provider: str,
    model_id: str,
    no_of_concurrent_workers_for_llm: int = 3,
) -> bool:
    """
    Rewrite (edit) responsibilities for a single URL and persist them
    (DuckDB + FSM).

    Overview
    --------
    This function performs the "editing" stage for one job posting:
      - Guards on FSM stage: only runs if the URL is at
        `EDITED_RESPONSIBILITIES`.
      - Marks the control row as IN_PROGRESS.
      - Loads the URL’s flattened responsibilities and flattened requirements
        from DuckDB.
      - Calls the async LLM editor (`modify_multi_resps_based_on_reqs`)
        to produce edited responsibilities aligned to requirements.
      - Stamps metadata (stage/table/version/iteration/llm_provider)
        with `add_metadata`.
      - Inserts rows into `edited_responsibilities` with de-duplication.
      - Advances the FSM: `COMPLETED` → step() to `EDITED_RESPONSIBILITIES` → `NEW`.

    Concurrency
    -----------
    LLM calls are bounded by the provided `semaphore` and an internal
    `no_of_concurrent_workers_for_llm` inside the editor.

    Error Handling
    --------------
    - Any failure (load/validate/insert/FSM/LLM) logs, marks the URL `ERROR`,
      does **not** advance the FSM, and returns False.
    - On success, the FSM advances, and the function returns True.

    Args:
        url: Canonical job posting URL.
        fsm_manager: Shared FSM manager used to read and update pipeline state.
        semaphore: Async semaphore used to bound URL-level concurrency.
        iteration: Iteration stamp for auditing/reruns.
        llm_provider: LLM provider label (e.g., "openai", "anthropic").
        model_id: LLM model identifier for editing.
        no_of_concurrent_workers_for_llm: Internal concurrency inside the editor.

    Returns:
        bool: True on success; False if skipped or failed.
    """
    async with semaphore:
        fsm = fsm_manager.get_fsm(url)

        # Guard: only operate when current stage is EDITED_RESPONSIBILITIES
        if fsm.get_current_stage() != PipelineStage.EDITED_RESPONSIBILITIES.value:
            logger.info("⏩ Skipping %s; current stage is %s", url, fsm.state)
            return False

        # IN_PROGRESS at current stage
        fsm.mark_status(PipelineStatus.IN_PROGRESS, notes="Editing responsibilities…")

        try:
            resps_model = load_table(TableName.FLATTENED_RESPONSIBILITIES, url=url)
            reqs_model = load_table(TableName.FLATTENED_REQUIREMENTS, url=url)

            # Validate loader return types up front (great errors when config drifts)
            resps_dict = get_resps_dict_strict(resps_model)
            reqs_dict = get_reqs_dict_strict(reqs_model)

            if not resps_dict or not reqs_dict:
                raise ValueError(
                    "Empty responsibilities or requirements after rehydration"
                )

        except Exception:
            logger.exception("❌ Failed to load/rehydrate inputs for %s", url)
            fsm.mark_status(PipelineStatus.ERROR, notes="Load inputs failed")
            return False

        # Editor call stays simple (dict[str,str], dict[str,str])
        try:
            matches: ResponsibilityMatches = (
                await modify_multi_resps_based_on_reqs_async(
                    responsibilities=resps_dict,
                    requirements=reqs_dict,
                    llm_provider=llm_provider,
                    model_id=model_id,
                    no_of_concurrent_workers=no_of_concurrent_workers_for_llm,
                )
            )
        except Exception:
            logger.exception("❌ Editor failed for %s", url)
            fsm.mark_status(PipelineStatus.ERROR, notes="LLM editor failed")
            return False

        # --- Flatten ResponsibilityMatches → rows
        try:

            # Expected shape: { resp_key: { req_key: "edited text", ... }, ... }
            rows: List[Dict[str, Any]] = []
            for resp_key, by_req in matches.responsibilities.items():
                for req_key, optimized_text in by_req.optimized_by_requirements.items():
                    rows.append(
                        {
                            "responsibility_key": resp_key,
                            "requirement_key": req_key,
                            "responsibility": optimized_text.optimized_text,  # real string, not a dict repr
                        }
                    )
            logger.debug("Edited sample: %s", rows[:2])

            edited_df = pd.DataFrame(rows)
            if edited_df.empty:
                raise ValueError("Editor produced no rows")

            # Single call -> stamping (iteration/LLM), alignment, and dedup are handled by inserter/YAML
            insert_df_with_config(
                edited_df,
                TableName.EDITED_RESPONSIBILITIES,
                url=url,
                llm_provider=llm_provider,  # ensures the actual provider is recorded
                model_id=model_id,  # ensures the actual provider is recorded
            )

        except Exception:
            logger.exception("❌ Persist failed for %s", url)
            fsm.mark_status(PipelineStatus.ERROR, notes="DB insert failed")
            return False

        # --- Advance FSM to EDITED_RESPONSIBILITIES → NEW
        try:
            fsm.mark_status(
                PipelineStatus.COMPLETED, notes="Edited → DB"
            )  # not needed but kept for notes
            fsm.step()  # EDITED_RESPONSIBILITIES → SIM_METRICS_REVAL

            logger.info("✅ Editing complete for %s", url)
            return True
        except Exception:
            logger.exception("❌ FSM transition failed for %s", url)
            fsm.mark_status(PipelineStatus.ERROR, notes="FSM transition failed")
            return False


# =============================================================================
# Batch runner (bounded concurrency)
# =============================================================================
async def process_resume_editing_batch_async_fsm(
    urls: List[str],
    *,
    iteration: int = 0,
    llm_provider: str,
    model_id: str,
    max_concurrent_urls: int = 4,
    no_of_concurrent_workers_for_llm: int = 3,
) -> list[asyncio.Task]:
    """
    Kick off bounded-concurrency editing tasks for a list of URLs.
    """
    semaphore = asyncio.Semaphore(max_concurrent_urls)
    fsm_manager = PipelineFSMManager()

    async def _run_one(u: str) -> None:
        await edit_and_persist_responsibilities_for_url(
            u,
            fsm_manager=fsm_manager,
            semaphore=semaphore,
            # iteration=iteration,
            llm_provider=llm_provider,
            model_id=model_id,
            no_of_concurrent_workers_for_llm=no_of_concurrent_workers_for_llm,
        )

    return [asyncio.create_task(_run_one(u)) for u in urls]


# =============================================================================
# Entrypoint (stage worklist → tasks → await)
# =============================================================================
async def run_resume_editing_pipeline_async_fsm(
    *,
    iteration: int = 0,
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_1_NANO,
    max_concurrent_urls: int = 4,
    no_of_concurrent_workers_for_llm: int = 3,
    filter_urls: Optional[Sequence[str]] = None,
    limit_urls: Optional[int] = None,
) -> None:
    """
    FSM-aware entrypoint for **editing** flattened responsibilities and persisting
    them into DuckDB.

    What It Does
    ------------
    - Queries the `pipeline_control` table for all job URLs currently at
      stage = `FLATTENED_RESPONSIBILITIES` with status = `NEW`.
    - Optionally filters this list to a caller-provided subset (`filter_urls`) and
      trims with `limit_urls`.
    *- For each eligible URL:
        * Loads flattened responsibilities + flattened requirements (DuckDB).
        * Runs the async editor to produce LLM-edited responsibilities.
        * Stamps standard metadata (stage, iteration, version='edited', llm_provider).
        * Inserts deduplicated rows into `edited_responsibilities`.
        * Advances the FSM: mark current stage `COMPLETED` →
          step() to `EDITED_RESPONSIBILITIES` →
          mark new stage `NEW`.
    - Runs all URLs concurrently, bounded by `max_concurrent_urls`.

    Concurrency
    -----------
    - Uses a stage-level semaphore. The editor can also run multiple
        internal workers per-URL (`no_of_concurrent_workers_for_llm`).

    Error Handling
    --------------
    - If no URLs match, the function logs and returns early.
    - For individual URLs:
        * Any load/LLM/insert/FSM error logs, marks status = `ERROR`,
          does not advance, and continues with other URLs.
    - The pipeline completes even if some URLs fail.

    Args:
        iteration: Iteration stamp to apply on output rows.
        llm_provider: Provider label used for stamping (e.g., "openai").
        model_id: Model ID used by the editing function.
        max_concurrent_urls: Maximum number of concurrent URL tasks.
        no_of_concurrent_workers_for_llm: Internal concurrency per URL
            in the editor.
        filter_urls: Optional subset of URLs to process.
        limit_urls: Optional cap on how many URLs to pull.

    Returns:
        None. Side effects include writing rows to DuckDB and updating the
        `pipeline_control` FSM state.
    """
    # Select urls to process (stage / version)
    urls = get_urls_ready_for_transition(
        stage=PipelineStage.EDITED_RESPONSIBILITIES,
    )

    if filter_urls:
        urls = [u for u in urls if u in filter_urls]
    if not urls:
        logger.info(
            f"📭 No URLs to edit at {PipelineStage.EDITED_RESPONSIBILITIES.value} stage."
        )
        return
    if limit_urls:
        urls = urls[:limit_urls]

    logger.info("✏️ Editing responsibilities for %d URL(s)…", len(urls))

    tasks = await process_resume_editing_batch_async_fsm(
        urls=urls,
        iteration=iteration,
        llm_provider=llm_provider,
        model_id=model_id,
        max_concurrent_urls=max_concurrent_urls,
        no_of_concurrent_workers_for_llm=no_of_concurrent_workers_for_llm,
    )

    await asyncio.gather(*tasks)

    logger.info("✅ Finished resume editing FSM pipeline.")
