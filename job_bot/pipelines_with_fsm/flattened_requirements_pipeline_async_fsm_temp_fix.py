"""
pipelines_with_fsm/flattened_requirements_pipeline_async_fsm.py

FSM-aware, DuckDB-native stage:
    EXTRACTED_REQUIREMENTS  --(copy/transform)-->  FLATTENED_REQUIREMENTS

Operation
---------
Per-URL:
  • Ensure `flattened_requirements` table exists (DDL no-op if present)
  • DELETE existing flat rows for this url+iteration
  • INSERT ... SELECT from `extracted_requirements` with key transform
  • Advance FSM: COMPLETED → step() → NEW

No filesystem dependency. No LLM calls. Fast + explicit table writes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, List, Tuple

# Enums / pipeline metadata
from job_bot.db_io.pipeline_enums import (
    PipelineStage,
    PipelineStatus,
)

# FSM
from job_bot.fsm.pipeline_fsm_manager import PipelineFSMManager, PipelineFSM

# Worklist + IO
from job_bot.db_io.db_utils import get_urls_ready_for_transition
from job_bot.db_io.get_db_connection import get_db_connection

# Optional control-plane sync (best-effort, zero-arg whole-table sync)
from job_bot.fsm.pipeline_control_sync import (
    sync_job_urls_to_pipeline_control,  # type: ignore
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# SQL snippets
# ──────────────────────────────────────────────────────────────────────────────

# Make sure the physical table exists with the expected schema
ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS flattened_requirements AS
SELECT
  CAST(NULL AS VARCHAR) AS url,
  CAST(NULL AS INTEGER) AS iteration,
  CAST(NULL AS VARCHAR) AS requirement_key,
  CAST(NULL AS VARCHAR) AS requirement,
  CAST(NULL AS VARCHAR) AS source_file
WHERE 1=0;
"""

# For this URL, figure out which iteration we should flatten (use latest from extracted)
GET_MAX_ITER_FOR_URL_SQL = """
SELECT MAX(iteration) AS it
FROM extracted_requirements
WHERE url = ?
"""

# Remove any previous flattened rows for this url + iteration (idempotent)
DELETE_EXISTING_SQL = """
DELETE FROM flattened_requirements
WHERE url = ?
  AND iteration = ?
"""

# Copy rows from extracted → flattened (compose the flat key)
INSERT_FLAT_SQL = """
INSERT INTO flattened_requirements (url, iteration, requirement_key, requirement, source_file)
SELECT
  er.url,
  er.iteration,
  CONCAT(
    CAST(er.requirement_category_key AS VARCHAR),
    '.',
    er.requirement_category,
    '.',
    CAST(er.requirement_key AS VARCHAR)
  ) AS requirement_key,
  er.requirement,
  er.source_file
FROM extracted_requirements er
WHERE er.url = ?
  AND er.iteration = ?
"""

# Quick existence check so we don't advance URLs with no extracted rows
CHECK_URL_IN_EXTRACTED_SQL = """
SELECT 1 FROM extracted_requirements WHERE url = ? LIMIT 1
"""


def _copy_extracted_to_flattened_for_url(url: str) -> int:
    """
    Ensure table, delete old flat rows for (url, iter), and insert fresh rows.
    Returns number of rows inserted.
    """
    con = get_db_connection()
    try:
        # Ensure destination table
        con.execute(ENSURE_TABLE_SQL)

        # Do we have any extracted rows for this URL?
        has_row: Optional[Tuple] = con.execute(
            CHECK_URL_IN_EXTRACTED_SQL, [url]
        ).fetchone()
        if has_row is None:
            return -1  # signal "no data"

        # Latest iteration from extracted_requirements
        max_iter_row: Optional[Tuple] = con.execute(
            GET_MAX_ITER_FOR_URL_SQL, [url]
        ).fetchone()
        if max_iter_row is None or max_iter_row[0] is None:
            return -1

        (iteration_val,) = max_iter_row  # tuple-unpack avoids subscripting an Optional
        iteration: int = int(iteration_val)

        # Clean old rows
        con.execute(DELETE_EXISTING_SQL, [url, iteration])

        # Insert fresh projection
        con.execute(INSERT_FLAT_SQL, [url, iteration])

        # Count inserted rows (guard fetchone() again)
        count_row: Optional[Tuple] = con.execute(
            "SELECT COUNT(*) FROM flattened_requirements WHERE url = ? AND iteration = ?",
            [url, iteration],
        ).fetchone()
        if count_row is None or count_row[0] is None:
            return 0

        (count_val,) = count_row
        return int(count_val)

    finally:
        con.close()


# ──────────────────────────────────────────────────────────────────────────────
# Per-URL action
# ──────────────────────────────────────────────────────────────────────────────


async def flatten_and_persist_requirements_for_url(
    url: str,
    *,
    fsm_manager: PipelineFSMManager,
    semaphore: asyncio.Semaphore,
) -> bool:
    """
    Copy/transform 'flatten' for one URL:
      - Marks IN_PROGRESS
      - Ensures table, DELETE old rows, INSERT ... SELECT from extracted
      - Advances FSM (COMPLETED → step() → NEW)
    """
    async with semaphore:
        fsm: PipelineFSM = fsm_manager.get_fsm(url)

        # mark work start
        fsm.mark_status(PipelineStatus.IN_PROGRESS, notes="Flattening (copy/insert)…")

        try:
            inserted = _copy_extracted_to_flattened_for_url(url)
            if inserted < 0:
                logger.error("❌ No extracted_requirements rows for %s", url)
                fsm.mark_status(
                    PipelineStatus.ERROR,
                    notes="No extracted rows found for url; cannot flatten",
                )
                return False
            logger.info("📦 flattened %s → %d row(s)", url, inserted)
        except Exception:
            logger.exception("❌ Copy/insert failed for %s", url)
            fsm.mark_status(PipelineStatus.ERROR, notes="DB copy/insert failed")
            return False

        # Advance FSM (same semantics)
        try:
            fsm.mark_status(PipelineStatus.COMPLETED, notes="Flattened → DB")
            fsm.step()  # FLATTENED_REQUIREMENTS → next stage

            # Optional whole-table sync; zero-arg call is correct
            if sync_job_urls_to_pipeline_control:
                try:
                    sync_job_urls_to_pipeline_control()
                except Exception:
                    logger.warning("⚠️ Control sync failed for %s (non-fatal)", url)

            logger.info("✅ Flatten requirements complete for %s", url)
            return True
        except Exception:
            logger.exception("❌ FSM transition failed for %s", url)
            fsm.mark_status(PipelineStatus.ERROR, notes="FSM transition failed")
            return False


# ──────────────────────────────────────────────────────────────────────────────
# Batch runner + pipeline entrypoint (unchanged signatures)
# ──────────────────────────────────────────────────────────────────────────────


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
    FSM-aware entrypoint: copy/transform extracted → flattened for eligible URLs.
    """
    urls = get_urls_ready_for_transition(stage=PipelineStage.FLATTENED_REQUIREMENTS)

    if filter_urls:
        urls = [u for u in urls if u in filter_urls]

    if not urls:
        logger.info("📭 No URLs to flatten at 'extracted_requirements' stage.")
        return

    logger.info("🧱 Flattening (copy/insert) requirements for %d URL(s)…", len(urls))

    tasks = await process_flattened_requirements_batch_async_fsm(
        urls=urls,
        max_concurrent=max_concurrent,
    )

    await asyncio.gather(*tasks)

    logger.info("✅ Finished flattened requirements FSM pipeline.")
