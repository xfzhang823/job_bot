"""
pipelines_with_fsm/update_flattened_responsibilities_pipeline_fsm.py

DB-native mini-pipeline (sync): materialize `flattened_responsibilities` from the
current resume JSON for a worklist of (url, iteration) items.

Worklist (from pipeline_control; human-gated + lease-aware):
  • stage = FLATTENED_RESPONSIBILITIES
  • status ∈ {NEW[, ERROR if retry_errors=True]}

What it does (per (url, iteration)):
  1) try_claim_one(url, iteration, worker_id) → acquire lease or skip
  2) process_responsibilities_from_resume(...) → Responsibilities (validated)
  3) flatten_model_to_df(...) → aligned DataFrame
  4) insert_df_with_config(..., iteration=iteration)
     - mode="append": PK-scoped dedup (default)
     - mode="replace": wipe-by-URL before insert (key_cols=["url"])
  5) finalize_one_row_in_pipeline_control(url, iteration, worker_id, ok, notes)
  6) if ok and finalized and still at FLATTENED_RESPONSIBILITIES → fsm.step()
     (typically to SIM_METRICS_EVAL)

Notes:
  • Sync & simple on purpose (no network LLM calls here).
  • Idempotent per (url, iteration) via table PK + YAML dedup.
  • Optional control-plane sync is kept for back-compat.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd

# Pure processing → validated Responsibilities model
from job_bot.evaluation_optimization.evaluation_optimization_utils import (
    process_responsibilities_from_resume,
)

# DB utilities
from job_bot.db_io.flatten_and_rehydrate import flatten_model_to_df
from job_bot.db_io.db_inserters import insert_df_with_config
from job_bot.db_io.pipeline_enums import TableName, PipelineStage, PipelineStatus
from job_bot.db_io.db_utils import (
    get_claimable_worklist,
    try_claim_one,
    finalize_one_row_in_pipeline_control,
    generate_worker_id,
)

# FSM
from job_bot.fsm.pipeline_fsm_manager import PipelineFSMManager

# Optional control-plane sync helper
from job_bot.fsm.pipeline_control_sync import (
    sync_flattened_responsibilities_to_pipeline_control,
)

# Default nested resume JSON
from job_bot.config.project_config import RESUME_JSON_FILE as DEFAULT_RESUME_JSON_FILE

logger = logging.getLogger(__name__)


def _process_one_url(
    *,
    url: str,
    iteration: int,
    resume_json_path: Path,
    mode: Literal["append", "replace"],
) -> bool:
    """
    Pure operation: read/validate resume JSON → flatten → insert.
    No claim/lease/FSM mutations here. Returns True on success.
    """
    try:
        # 1) Produce validated Responsibilities
        model = process_responsibilities_from_resume(
            resume_json_file=resume_json_path,
            url=url,
        )
        if model is None:
            logger.error("❌ No responsibilities produced for %s", url)
            return False

        # 2) Flatten to table-aligned DataFrame
        df: pd.DataFrame = flatten_model_to_df(
            model=model,
            table_name=TableName.FLATTENED_RESPONSIBILITIES,
            source_file=resume_json_path,
        )
        if df.empty:
            logger.error("❌ Empty responsibilities DataFrame for %s", url)
            return False

        # 3) Insert with configured semantics
        kwargs: dict[str, Any] = dict(
            url=url,
            iteration=iteration,  # propagate iteration explicitly
            mode=mode,
        )
        # For "replace", wipe-by-URL first (broad reset for snapshot refresh)
        if mode == "replace":
            kwargs["key_cols"] = ["url"]

        insert_df_with_config(
            df,
            TableName.FLATTENED_RESPONSIBILITIES,
            **kwargs,
        )
        logger.info(
            "📦 flattened_responsibilities: inserted %d row(s) for %s", len(df), url
        )
        return True

    except Exception:
        logger.exception("❌ Flatten/insert failed for %s", url)
        return False


def run_flattened_responsibilities_pipeline_fsm(
    *,
    resume_json_file: str | Path = DEFAULT_RESUME_JSON_FILE,
    mode: Literal["append", "replace"] = "append",
    do_control_sync: bool = True,
    retry_errors: bool = False,
    filter_urls: Optional[list[str]] = None,
    limit_urls: Optional[int] = None,
) -> None:
    """
    Refresh `flattened_responsibilities` for a claimable worklist at FLATTENED_RESPONSIBILITIES.

    Workflow (lease-aware, human-gated, claimables pattern)
    -------------------------------------------------------
    1) Build worklist:
         get_claimable_worklist(stage=FLATTENED_RESPONSIBILITIES, status={NEW[, ERROR]})
       • Enforces human gate (task_state='READY') and lease rules
        (unclaimed or lease-expired)
       • Returns list[(url, iteration)]
    2) Optional filters: `filter_urls`, `limit_urls`
    3) Generate worker_id for this run.
    4) For each (url, iteration):
         a) try_claim_one(url, iteration, worker_id) — acquire lease or skip
         b) _process_one_url(...) — pure operation (validate, flatten, insert)
         c) finalize_one_row_in_pipeline_control(url, iteration, worker_id, ok, notes)
         d) if ok and finalized and still at FLATTENED_RESPONSIBILITIES → fsm.step()
    5) Optional control-plane sync (non-fatal)

    Parameters
    ----------
    resume_json_file : str | Path
        Path to the nested resume JSON (single-resume source of truth).
    mode : {"append","replace"}
        Insert behavior. See module docstring.
    do_control_sync : bool
        If True, run 'sync_flattened_responsibilities_to_pipeline_control()'
        at the end.
    retry_errors : bool
        If True, include ERROR rows in the claimable worklist.
    filter_urls : list[str] | None
        Restrict processing to this allowlist of URLs.
    limit_urls : int | None
        Optional cap on worklist size after filtering.

    Returns
    -------
    None
    """
    statuses = (
        (PipelineStatus.NEW, PipelineStatus.ERROR)
        if retry_errors
        else (PipelineStatus.NEW,)
    )
    worklist: list[tuple[str, int]] = get_claimable_worklist(
        stage=PipelineStage.FLATTENED_RESPONSIBILITIES,
        status=statuses,
        max_rows=1000,
    )

    if filter_urls:
        filt = set(filter_urls)
        worklist = [(u, it) for (u, it) in worklist if u in filt]
    if limit_urls:
        worklist = worklist[:limit_urls]

    if not worklist:
        logger.info(
            "📭 No claimable rows at %s.",
            PipelineStage.FLATTENED_RESPONSIBILITIES.value,
        )
        return

    resume_json_path = Path(resume_json_file)
    worker_id = generate_worker_id("flattened_responsibilities")
    fsm_manager = PipelineFSMManager()

    logger.info(
        "🧱 Starting flattened_responsibilities | items=%d | mode=%s | worker_id=%s",
        len(worklist),
        mode,
        worker_id,
    )

    for url, iteration in worklist:
        # Claim or skip
        if not try_claim_one(url=url, iteration=iteration, worker_id=worker_id):
            logger.info("⏭️ Skipping %s@%s — already claimed.", url, iteration)
            continue

        ok = _process_one_url(
            url=url,
            iteration=iteration,
            resume_json_path=resume_json_path,
            mode=mode,
        )

        finalized = finalize_one_row_in_pipeline_control(
            url=url,
            iteration=iteration,
            worker_id=worker_id,
            ok=ok,
            notes="Flattened responsibilities saved" if ok else "Flattening failed",
        )
        if not finalized:
            logger.warning(
                "[finalize] Lost lease for %s@%s; not stepping.", url, iteration
            )
            continue

        if ok:
            try:
                expected_source_stage = PipelineStage.FLATTENED_RESPONSIBILITIES
                fsm = fsm_manager.get_fsm(url)
                if fsm.get_current_stage() == expected_source_stage.value:
                    # Typically advance to SIM_METRICS_EVAL as the next stage
                    fsm.step()
                else:
                    logger.info(
                        "Not stepping %s@%s: stage moved from %s → %s elsewhere.",
                        url,
                        iteration,
                        expected_source_stage.value,
                        fsm.state,
                    )
            except Exception:
                logger.exception("FSM step() failed for %s@%s", url, iteration)

    if do_control_sync:
        try:
            sync_flattened_responsibilities_to_pipeline_control()
            logger.info(
                "🔄 Control-plane sync complete for flattened_responsibilities."
            )
        except Exception:
            logger.warning("⚠️ Control-plane sync failed (non-fatal).")

    logger.info("✅ Done. Processed %d item(s).", len(worklist))
