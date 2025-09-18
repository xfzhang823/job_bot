"""
pipelines_with_fsm/flatten_reqs_pipeline_async_fsm.py

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

# Standard libs
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional

# Project libs (align with your repo layout)
from fsm.pipeline_fsm import PipelineStage, PipelineStatus
from fsm.pipeline_fsm_manager import PipelineFSMManager
from db_io.db_utils import get_urls_by_stage_and_status
from db_io.db_transform import add_all_metadata
from db_io.db_insert import insert_df_dedup
from db_io.flatten_and_rehydrate import flatten_extracted_requirements_to_table
from db_io.pipeline_enums import TableName, Version

# Pydantic I/O models
from models.resume_job_description_io_models import (
    RequirementsResponse,
    ExtractedRequirementsBatch,
)

# Optional control-plane sync (best-effort)
# If you maintain a separate control sync module
from state_orchestration_pipeline import sync_job_urls_to_pipeline_control  # type: ignore


# Optional: loader(s) — support either a per-URL loader or a batch loader
from utils.pydantic_model_loaders_from_db import (
    load_extracted_requirements_for_url_from_db as _load_reqs_single,  # type: ignore
)

from utils.pydantic_model_loaders_from_db import (
    load_all_extracted_requirements_model_from_db as _load_reqs_all,
)


logger = logging.getLogger(__name__)


async def flatten_and_persist_reqs_for_url(
    url: str,
    *,
    fsm_manager: PipelineFSMManager,
    semaphore: asyncio.Semaphore,
) -> bool:
    """
    Flatten extracted requirements for a single URL and persist them (DuckDB + FSM).

    What this does
    --------------
    - Guard: only runs if the URL is at `EXTRACTED_REQUIREMENTS`.
    - Mark IN_PROGRESS in `pipeline_control`.
    - Load the URL’s `RequirementsResponse` from DuckDB.
    - Validate it has at least one non-empty requirement.
    - Flatten to a long-form table via `flatten_extracted_requirements_to_table`.
    - Stamp metadata with `add_all_metadata` (stage/table/version/iteration, etc.).
    - Insert rows into `flattened_requirements` (deduped).
    - Advance FSM: `COMPLETED` → `step()` to `FLATTENED_REQUIREMENTS` → mark `NEW`.
    - (Optional) best-effort control sync.

    Concurrency
    -----------
    Bounded by `semaphore` (wraps the whole operation). This stage does no LLM or
    network I/O, so you can set higher concurrency (default 8 in the entrypoint).

    Error handling
    --------------
    Any failure (load/validate/insert/FSM) logs, marks `ERROR`, does **not** advance,
    and returns False. Success returns True.

    Args:
        url: Canonical job URL key.
        fsm_manager: Shared FSM manager used to read/update control state.
        semaphore: Concurrency limiter for this stage.

    Returns:
        True on success; False if skipped or failed.
    """
    async with semaphore:  # ← ensure we actually use the semaphore
        fsm = fsm_manager.get_fsm(url)

        if fsm.get_current_stage() != PipelineStage.EXTRACTED_REQUIREMENTS.value:
            logger.info("⏩ Skipping %s; current stage is %s", url, fsm.state)
            return False

        # Update status to in progress
        fsm.mark_status(PipelineStatus.IN_PROGRESS, notes="Flattening requirements…")

        try:
            req_model = _load_extracted_reqs_for_url(url)
            if req_model is None:
                raise ValueError("No extracted requirements found for URL")
        except Exception:
            logger.exception("❌ Failed to load extracted requirements for %s", url)
            fsm.mark_status(
                PipelineStatus.ERROR, notes="Load extracted_requirements failed"
            )
            return False

        if not _validate_reqs_model(req_model):
            logger.warning("🚫 Skipping %s — empty/invalid requirements.", url)
            fsm.mark_status(PipelineStatus.ERROR, notes="Empty/invalid requirements")
            return False

        try:
            # Validate with pydantic model
            model = ExtractedRequirementsBatch({url: req_model})
            df = flatten_extracted_requirements_to_table(model)

            df = add_all_metadata(
                df=df,
                file_path=Path("fsm"),
                stage=PipelineStage.FLATTENED_REQUIREMENTS,
                table=TableName.FLATTENED_REQUIREMENTS,
                version=Version.ORIGINAL,  # optional; omit if you want the default
                llm_provider=None,  # or LLMProvider.OPENAI etc. if you carry it forward
                iteration=0,  # bump if you re-run this stage
            )
            insert_df_dedup(df, TableName.FLATTENED_REQUIREMENTS.value)
        except Exception:
            logger.exception("❌ Failed to persist flattened requirements for %s", url)
            fsm.mark_status(PipelineStatus.ERROR, notes="DB insert failed")
            return False

        try:
            # Update status to completed
            fsm.mark_status(PipelineStatus.COMPLETED, notes="Flattened → DB")
            fsm.step()  # EXTRACTED_REQUIREMENTS -> FLATTENED_REQUIREMENTS
            fsm.mark_status(PipelineStatus.NEW, notes="Ready for next stage")

            if sync_job_urls_to_pipeline_control:
                try:
                    sync_job_urls_to_pipeline_control([url])  # type: ignore[misc]
                except Exception:
                    logger.warning("⚠️ Control sync failed for %s (non-fatal)", url)

            # ✅ success path always returns True (even if sync helper is absent)
            logger.info("✅ Flatten requirements complete for %s", url)
            return True

        except Exception:
            logger.exception("❌ FSM transition failed for %s", url)
            fsm.mark_status(PipelineStatus.ERROR, notes="FSM transition failed")
            return False


async def process_flattened_reqs_batch_async_fsm(
    urls: list[str],
    *,
    max_concurrent: int = 8,
) -> list[asyncio.Task]:
    """Kick off bounded-concurrency tasks for a list of URLs."""
    semaphore = asyncio.Semaphore(max_concurrent)
    fsm_manager = PipelineFSMManager()

    async def _run_one(u: str) -> None:
        await flatten_and_persist_reqs_for_url(
            u, fsm_manager=fsm_manager, semaphore=semaphore
        )

    return [asyncio.create_task(_run_one(u)) for u in urls]


async def run_flattened_reqs_pipeline_async_fsm(
    *,
    filter_urls: Optional[list[str]] = None,
    max_concurrent: int = 8,
) -> None:
    """
    Entrypoint: process all URLs at stage `EXTRACTED_REQUIREMENTS/NEW` and
    persist flattened requirements to DuckDB, advancing FSM to
    `FLATTENED_REQUIREMENTS/NEW`.
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

    tasks = await process_flattened_reqs_batch_async_fsm(
        urls=urls,
        max_concurrent=max_concurrent,
    )

    await asyncio.gather(*tasks)

    logger.info("✅ Finished flattened requirements FSM pipeline.")


def _validate_reqs_model(model: RequirementsResponse) -> bool:
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


def _load_extracted_reqs_for_url(url: str) -> Optional[RequirementsResponse]:
    """Best-effort loader that prefers per-URL loader; falls back to batch loader."""
    if _load_reqs_single is not None:
        model = _load_reqs_single(url=url, status=None, iteration=0)  # type: ignore[misc]
        # Expect shape like: {url: RequirementsResponse}
        try:
            return model.root[url]  # type: ignore[attr-defined]
        except Exception:
            return None

    if _load_reqs_all is not None:
        batch = _load_reqs_all()
        try:
            return batch.root[url]  # type: ignore[attr-defined]
        except Exception:
            return None

    raise ImportError(
        "No loader available: expected utils.pydantic_model_loaders_from_db to expose either "
        "'load_extracted_requirements_for_url_from_db' or 'load_all_extracted_requirements_model_from_db'."
    )
