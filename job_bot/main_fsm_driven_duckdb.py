"""
main_fsm_driven_duckdb.py

High-level orchestrator for the DuckDB-backed FSM pipelines.

This module coordinates the entire pipeline execution sequence:
1. Ensure all DuckDB tables exist (schema bootstrap).
2. Pre-flight updates:
   - Ingest `job_urls` from JSON and sync into `pipeline_control`.
   - Optionally refresh `flattened_responsibilities` if the resume has been updated.
3. Run stage pipelines in order (async for concurrent URL-level tasks).
4. Optionally run reporting/analytics at the end (e.g., crosstab export).

The design separates:
- Control-plane sync (`state_orchestration_pipeline`)
- Stage-specific pipelines (job_postings, requirements extraction/flattening,
  resume editing, similarity metrics, etc.)
- Reporting/export pipelines

Each pipeline is FSM-aware and uses the control table to determine its worklist.
This orchestrator simply drives them in a logical sequence.

Usage:
    python main_fsm_driven_duckdb.py
"""

# Dependencies
from __future__ import annotations

# * ✅ Force Transformers & BERTScore to use local cache
from job_bot.pipelines.hf_cache_refresh_and_lock_pipeline import (
    run_hf_cache_refresh_and_lock_pipeline,
)

# Setting environment variables for Transformers to force the library to work entirely
# from the local cache only!
run_hf_cache_refresh_and_lock_pipeline(refresh_cache=False)

# from standard/third-party
import os
import asyncio
import logging
import matplotlib

# User defined
from job_bot.db_io.create_db_tables import create_all_db_tables
import job_bot.config.logging_config

# 0) URL intake + control-plane seed
from job_bot.pipelines_with_fsm.update_job_urls_pipeline_fsm import (
    run_update_job_urls_pipeline_fsm,
)
from job_bot.fsm.pipeline_control_sync import (
    sync_job_urls_to_pipeline_control,
)


# 1) Stage pipelines
from job_bot.pipelines_with_fsm.pipe_control_auto_transition_pipeline_fsm import (
    run_pipe_control_auto_transition_pipeline_fsm,
)
from job_bot.pipelines_with_fsm.job_postings_pipeline_async_fsm import (
    run_job_postings_pipeline_async_fsm,
)
from job_bot.pipelines_with_fsm.extract_requirements_pipeline_async_fsm import (
    run_extract_job_requirements_pipeline_async_fsm,
)

# from job_bot.pipelines_with_fsm.flattened_requirements_pipeline_async_fsm import (
#     run_flattened_requirements_pipeline_async_fsm,
# )
from job_bot.pipelines_with_fsm.flattened_requirements_pipeline_async_fsm_temp_fix import (
    run_flattened_requirements_pipeline_async_fsm,
)  # temp fix to copy from the extracted req table
from job_bot.pipelines_with_fsm.flattened_responsibilities_pipeline_fsm import (
    run_flattened_responsibilities_pipeline_fsm,
)
from job_bot.pipelines_with_fsm.resume_editing_pipeline_async_fsm import (
    run_resume_editing_pipeline_async_fsm,
)
from job_bot.pipelines_with_fsm.similarity_metrics_pipeline_async_fsm import (
    run_similarity_metrics_eval_async_fsm,
    run_similarity_metrics_reval_async_fsm,
)

from job_bot.pipelines_with_fsm.alignment_review_pipeline_async_fsm import (
    run_alignment_review_pipeline_async_fsm,
)

logger = logging.getLogger(__name__)

matplotlib.use("Agg")  # Prevent interactive mode


async def run_all_fsm(*, append_only_urls: bool = True) -> None:
    """
    Run the full FSM-driven DuckDB pipeline in sequence.

    Steps
    -----
    0. Ensure DB schema exists (idempotent).
    1. Seed or refresh `pipeline_control` to align with job_urls/job_postings/etc.
       (initial run uses full=True to bootstrap).
    2. Execute the main pipelines in order:
       - Preprocessing
        (job_postings → extracted_requirements → flattened_requirements)
       - Metrics/evaluation stages
       - Resume editing stage
    3. Refresh `pipeline_control` with integrity checks enabled.
    4. Run optional reporting/export pipelines (e.g.,
        responsibilities–requirements crosstab).

    Notes
    -----
    - All heavy lifting is done inside the per-stage pipelines.
    - This function only orders their execution and ensures control-plane sync
      before and after.
    - Async/await is used so that stage runners can process many URLs concurrently.
    """
    # Ensure schema
    create_all_db_tables()

    # --- PRE-FLIGHT: update job_urls, then seed control-plane from it ---
    run_update_job_urls_pipeline_fsm(mode="append" if append_only_urls else "replace")

    sync_job_urls_to_pipeline_control()  # seeds JOB_URLS/NEW in pipeline_control
    logger.info(f"✅ Sync job urls to pipeline control table completed.")

    # sync pipeline control table: auto transition, etc. (idempotent)
    retried, advanced = run_pipe_control_auto_transition_pipeline_fsm(dry_run=False)

    logger.info(
        "🔄 Pipeline control auto-transition completed (retried=%s, advanced=%s)",
        retried,
        advanced,
    )

    # --- Pipelines in order (each pulls its own worklist from pipeline_control) ---

    # JOB_URLS → JOB_POSTINGS
    await run_job_postings_pipeline_async_fsm(max_concurrent_tasks=3, retry_errors=True)

    # JOB_POSTINGS → EXTRACTED_REQUIREMENTS
    await run_extract_job_requirements_pipeline_async_fsm()

    # EXTRACTED_REQUIREMENTS → FLATTENED_REQUIREMENTS
    await run_flattened_requirements_pipeline_async_fsm()

    # Upsert flattened resps: resume json -> FLATTENED_RESPONSIBILITIES
    run_flattened_responsibilities_pipeline_fsm()

    # run on FLATTENED_RESPONSIBILITIES
    await run_similarity_metrics_eval_async_fsm()

    # FLATTENED_RESPONSIBILITIES → EDITED_RESPONSIBILITIES
    await run_resume_editing_pipeline_async_fsm()

    # run on EDITED_RESPONSIBILITIES
    await run_similarity_metrics_reval_async_fsm()

    # run on EDITED_RESPONSIBILITIES
    await run_alignment_review_pipeline_async_fsm()


if __name__ == "__main__":
    asyncio.run(run_all_fsm())
