"""
pipelines_with_fsm/resume_editing_pipeline_async_fsm.py

DB-native Resume Editing Pipeline (FSM-driven; async; lease-aware claimables;
no filesystem I/O)

Worklist (from pipeline_control; human-gated + lease-aware):
  • EDIT: stage = EDITED_RESPONSIBILITIES,
    status ∈ {NEW[, ERROR if retry_errors=True]}

Inputs (per URL):
  • flattened_requirements        (requirement_key, requirement, url, …)
  • flattened_responsibilities    (responsibility_key, responsibility, url, …)

Output (per URL):
  • edited_responsibilities
      Columns (typical): url, responsibility_key, requirement_key,
        responsibility, iteration, version, llm_provider, stage, status,
        created_at, updated_at

Orchestration (claimables model):
  1) Build claimable worklist for EDITED_RESPONSIBILITIES with status {NEW[, ERROR]}.
  2) Generate a worker_id for this run.
  3) For each (url, iteration):
       a) try_claim_one(url, iteration, worker_id) → acquire lease
        or skip if already claimed.
       b) edit_and_persist_responsibilities_for_url(…) → pure compute/insert
        (no lease/FSM mutation).
       c) finalize_one_row_in_pipeline_control(url, iteration, worker_id, ok, notes)
          - Atomically sets final status (COMPLETED/ERROR)
          + clears lease iff ownership matches.
          - Returns True/False indicating whether finalize occurred
            (i.e., we still owned the lease).
       d) If ok and finalized → fsm.step() to advance to SIM_METRICS_REVAL
        (marks NEW there).

Notes:
  • Strictly DB-native: no JSON/CSV I/O.
  • Reuses Pydantic-validated loaders for rehydration & validation.
  • Mirrors the similarity-metrics FSM pipeline structure
    (bounded concurrency; gather).
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
)

# FSM
from job_bot.fsm.pipeline_fsm_manager import PipelineFSMManager

# Worklist + IO
from job_bot.db_io.db_inserters import insert_df_with_config
from job_bot.db_io.db_loaders import load_table

# Worklist + IO (lease-aware claimables)
from job_bot.db_io.db_utils import (
    get_claimable_worklist,
    try_claim_one,
    finalize_one_row_in_pipeline_control,
    generate_worker_id,
)

# async editor to rewrite responsibilities against requirements
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
    iteration: int,
    semaphore: asyncio.Semaphore,
    llm_provider: str,
    model_id: str,
    no_of_concurrent_workers_for_llm: int = 3,
) -> bool:
    """
    Rewrite (edit) responsibilities for a single URL and persist them
    (DuckDB + FSM).

    Steps
    -----
    1) Load flattened responsibilities + flattened requirements (DuckDB).
    2) Run async LLM editor:
         modify_multi_resps_based_on_reqs_async(responsibilities, requirements, ...)
    3) Flatten ResponsibilityMatches → row-per-(responsibility_key, requirement_key).
    4) INSERT into edited_responsibilities with standard metadata stamping
       (llm_provider/model_id via inserter args; iteration via param;
       version handled by config).

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
        semaphore: Async semaphore used to bound URL-level concurrency.
        iteration: Iteration stamp for auditing/reruns.
        llm_provider: LLM provider label (e.g., "openai", "anthropic").
        model_id: LLM model identifier for editing.
        no_of_concurrent_workers_for_llm: Internal concurrency inside
            the editor.

    Returns:
        bool: True on success; False if skipped or failed.
    """
    async with semaphore:
        # Load inputs
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
            return False

        # Run editor
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
            return False

        # Flatten → DataFrame → insert
        try:
            # Expected shape: { resp_key: { req_key: "edited text", ... }, ... }
            rows: List[Dict[str, Any]] = []
            for resp_key, by_req in matches.responsibilities.items():
                for req_key, optimized_text in by_req.optimized_by_requirements.items():
                    rows.append(
                        {
                            "url": url,  # 👈 add url here b/c the computation does not include url
                            "responsibility_key": resp_key,
                            "requirement_key": req_key,
                            "responsibility": optimized_text.optimized_text,  # real string, not a dict repr
                        }
                    )
            logger.debug("Edited sample: %s", rows[:2])

            edited_df = pd.DataFrame(rows)
            if edited_df.empty:
                raise ValueError("Editor produced no rows")

            # Single call -> stamping (iteration/LLM), alignment, and
            # dedup are handled by inserter/YAML
            insert_df_with_config(
                edited_df,
                TableName.EDITED_RESPONSIBILITIES,
                url=url,
                llm_provider=llm_provider,  # ensures the actual provider is recorded
                model_id=model_id,  # ensures the actual provider is recorded
                iteration=iteration,  # pass iteration
            )
            return True

        except Exception:
            logger.exception("❌ Persist failed for %s", url)
            return False


# =============================================================================
# Batch runner (bounded concurrency)
# =============================================================================
async def process_resume_editing_batch_async_fsm(
    url_iter_pairs: List[tuple[str, int]],
    *,
    worker_id: str,
    llm_provider: str,
    model_id: str,
    max_concurrent_urls: int = 4,
    no_of_concurrent_workers_for_llm: int = 3,
) -> list[asyncio.Task]:
    """
    Claim → run editing → finalize/step for a batch of (url, iteration) pairs.

    - Attempts to claim each (url, iteration) with `worker_id`.
    - If claimed, performs pure editing compute/insert.
    - Finalizes row with ok status and steps FSM on success (if we still own lease).
    """
    semaphore = asyncio.Semaphore(max_concurrent_urls)
    fsm_manager = PipelineFSMManager()

    async def _run_one(url: str, iteration: int) -> None:
        # Acquire lease or skip
        if not try_claim_one(url=url, iteration=iteration, worker_id=worker_id):
            logger.info("⏭️ Skipping %s@%s — already claimed.", url, iteration)
            return

        try:
            ok = await edit_and_persist_responsibilities_for_url(
                url,
                iteration=iteration,
                semaphore=semaphore,
                llm_provider=llm_provider,
                model_id=model_id,
                no_of_concurrent_workers_for_llm=no_of_concurrent_workers_for_llm,
            )

            finalized = finalize_one_row_in_pipeline_control(
                url=url,
                iteration=iteration,
                worker_id=worker_id,
                ok=ok,
                notes="Edited responsibilities saved to DB" if ok else "Editing failed",
            )

            if not finalized:
                logger.warning(
                    "[finalize] Lost lease for %s@%s; not stepping.", url, iteration
                )
                return

            if ok:
                try:
                    # EDITED_RESPONSIBILITIES → SIM_METRICS_REVAL
                    expected_source_stage = PipelineStage.EDITED_RESPONSIBILITIES

                    fsm = fsm_manager.get_fsm(url)
                    if fsm.get_current_stage() == expected_source_stage.value:
                        fsm.step()
                except Exception:
                    logger.exception("FSM step() failed for %s@%s", url, iteration)

        except Exception as e:
            logger.exception("❌ Failure in _run_one for %s@%s: %s", url, iteration, e)
            # Best-effort error finalize (still lease-validated)
            finalized = finalize_one_row_in_pipeline_control(
                url=url,
                iteration=iteration,
                worker_id=worker_id,
                ok=False,
                notes=f"editing failed: {e}",
            )
            if not finalized:
                logger.warning(
                    "[finalize] Could not mark ERROR for %s@%s (lease mismatch).",
                    url,
                    iteration,
                )

    return [asyncio.create_task(_run_one(u, it)) for (u, it) in url_iter_pairs]


# =============================================================================
# Entrypoint (stage worklist → tasks → await)
# =============================================================================
async def run_resume_editing_pipeline_async_fsm(
    *,
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_1_NANO,
    max_concurrent_urls: int = 4,
    no_of_concurrent_workers_for_llm: int = 3,
    filter_urls: Optional[Sequence[str]] = None,
    limit_urls: Optional[int] = None,
    retry_errors: bool = False,
) -> None:
    """
    FSM-aware entrypoint for **editing** flattened responsibilities and persisting
    them into DuckDB.

    Workflow (lease-aware, human-gated, claimables pattern)
    -------------------------------------------------------
    1) Build worklist (DB): call
         get_claimable_worklist(stage=EDITED_RESPONSIBILITIES, status={NEW[, ERROR]})
       • Enforces human gate (task_state='READY') and lease rules
        (unclaimed or lease expired).
       • Returns a list of (url, iteration) pairs.
    2) Optional filter: if `filter_urls` provided, restrict worklist to those URLs;
        if `limit_urls` is provided, truncate the list.
    3) Worker identity: generate a `worker_id` via generate_worker_id("resume_editing").
    4) Process batch (bounded concurrency):
         For each (url, iteration) in the worklist:
           a) try_claim_one(url, iteration, worker_id) — acquire a lease or skip
            if already claimed.
           b) edit_and_persist_responsibilities_for_url(...) — pure compute & insert,
            no lease/FSM mutation.
           c) finalize_one_row_in_pipeline_control(url, iteration, worker_id, ok, notes)
            — atomically
              writes final status and clears lease iff we still own it;
                returns bool (finalized or not).
           d) If ok and finalized → fsm.step() to advance to SIM_METRICS_REVAL.
    5) Await all tasks and log completion.

    Parameters
    ----------
        llm_provider : str
            Provider label used for stamping (e.g., "openai" or "anthropic").
        model_id : str
            Model ID used by the editing function.
        max_concurrent_urls : int
            Maximum number of concurrent URL tasks (outer-level semaphore).
        no_of_concurrent_workers_for_llm : int
            Internal concurrency inside the editor per URL.
        filter_urls : Optional[Sequence[str]]
            Optional subset of URLs to process.
        limit_urls : Optional[int]
            Optional cap on how many URLs to pull from the worklist
                (after filtering).
        retry_errors : bool
            If True, include ERROR rows in the claimable worklist in addition to NEW.

    Returns
    -------
    None

    Side effects:
        include writing rows to DuckDB and updating the `pipeline_control`
        FSM state.

    Notes
    -----
    • Idempotent per (url, iteration): inserter config should deduplicate
        on your chosen keys.
    • Keep concurrency modest to avoid long lease holds during LLM calls.

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
    """
    statuses = (
        (PipelineStatus.NEW, PipelineStatus.ERROR, PipelineStatus.IN_PROGRESS)
        if retry_errors
        else (PipelineStatus.NEW,)
    )
    worklist: List[tuple[str, int]] = get_claimable_worklist(
        stage=PipelineStage.EDITED_RESPONSIBILITIES,
        status=statuses,
        max_rows=max(1000, max_concurrent_urls * 4),
    )

    if filter_urls:
        filt = set(filter_urls)
        worklist = [(u, it) for (u, it) in worklist if u in filt]

    if not worklist:
        logger.info(
            f"📭 No claimable rows at {PipelineStage.EDITED_RESPONSIBILITIES.value}."
        )
        return

    if limit_urls:
        worklist = worklist[:limit_urls]

    worker_id = generate_worker_id("resume_editing")
    logger.info(
        "✏️ Starting resume editing | %d item(s) | worker_id=%s | provider=%s model=%s",
        len(worklist),
        worker_id,
        llm_provider,
        model_id,
    )

    tasks = await process_resume_editing_batch_async_fsm(
        url_iter_pairs=worklist,
        worker_id=worker_id,
        llm_provider=llm_provider,
        model_id=model_id,
        max_concurrent_urls=max_concurrent_urls,
        no_of_concurrent_workers_for_llm=no_of_concurrent_workers_for_llm,
    )

    await asyncio.gather(*tasks)

    logger.info("✅ Finished resume editing FSM pipeline.")
