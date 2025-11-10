# pipelines_with_fsm/job_urls_pipeline_fsm.py

from __future__ import annotations

import logging
from typing import Dict, Optional

from job_bot.db_io.pipeline_enums import PipelineStage, PipelineStatus
from job_bot.db_io.db_utils import (
    generate_worker_id,
    get_claimable_worklist,
    try_claim_one,
    release_one,
)
from job_bot.fsm.pipeline_fsm_manager import PipelineFSMManager

logger = logging.getLogger(__name__)


def run_job_urls_pipeline_fsm(*, max_batch: Optional[int] = None) -> Dict[str, int]:
    """
    Move URLs from 'job_urls' to 'job_postings' in a **lease-aware**, FSM-driven way.

    Lease-aware behavior (idempotent per (url, iteration)):
      1) Worklist:
         - Query claimables via `get_claimable_worklist(stage='job_urls', status='NEW')`
           which enforces the human gate (`task_state='READY'`) and lease rules
           (unclaimed or expired lease).
      2) Atomic claim (per row):
         - `try_claim_one(url, iteration, worker_id)` sets `status=IN_PROGRESS`
           and stamps lease fields (`is_claimed`, `worker_id`, `lease_until`).
      3) Complete and step:
         - Using the FSM manager, mark current stage COMPLETED and `step()` to
           enqueue the next stage ('job_postings') as `status='NEW'`.
      4) Release lease (finalize):
         - `release_one(..., final_status=COMPLETED)` clears the lease and persists
           the final status for this stage.
         - On error, release with `final_status=ERROR`.

    Notes:
      - This runner is control-plane quick work; no long-running tasks, so no
        lease renewal is needed.
      - `decision_flag` is not used here.
      - If a row is not actually at `JOB_URLS` when claimed (race/logic drift),
        it is released with `ERROR` so a later pass can re-evaluate cleanly.

    Args:
        max_batch: Optional cap on how many rows to process in this run.

    Returns:
        dict counters for logging/metrics: {found, claimed, completed, enqueued_next}
    """
    fsm_manager = PipelineFSMManager()
    worker_id = generate_worker_id(prefix="joburls")

    logger.info("PipelineStage.JOB_URLS.value = %r", PipelineStage.JOB_URLS.value)

    # 1) Build claimable worklist (honors human gate + lease rules)
    claimables = get_claimable_worklist(
        stage=PipelineStage.JOB_URLS,
        status=PipelineStatus.NEW,
        max_rows=(max_batch if (max_batch is not None and max_batch >= 0) else None),
    )

    if not claimables:
        logger.info("[job_urls_pipeline_fsm] No claimable rows at 'job_urls'.")
        return {"found": 0, "claimed": 0, "completed": 0, "enqueued_next": 0}

    claimed = completed = enqueued = 0
    found = len(claimables)
    logger.info(
        "[job_urls_pipeline_fsm] Found %d claimable row(s) at 'job_urls' → processing...",
        found,
    )

    # 2) Atomic claim → 3) Complete + step → 4) Release
    for url, iter_ in claimables:
        row = try_claim_one(
            url=url, iteration=iter_, worker_id=worker_id, lease_minutes=5
        )
        if not row:
            # Lost the race; another worker claimed it
            continue
        claimed += 1

        try:
            fsm = fsm_manager.get_fsm(url=url)
            if fsm.get_current_stage() != PipelineStage.JOB_URLS:
                logger.warning(
                    "[job_urls_pipeline_fsm] URL not at 'job_urls'; releasing as ERROR: %s (stage=%s)",
                    url,
                    fsm.get_current_stage(),
                )
                release_one(
                    url=url,
                    iteration=iter_,
                    worker_id=worker_id,
                    final_status=PipelineStatus.ERROR,
                )
                continue

            # Complete current stage in FSM
            fsm.mark_status(status=PipelineStatus.COMPLETED, notes="job_urls done")
            completed += 1

            # Enqueue next stage; conventionally step() creates next stage with NEW
            fsm.step()
            enqueued += 1

            # Release lease with COMPLETED
            release_ok = release_one(
                url=url,
                iteration=iter_,
                worker_id=worker_id,
                final_status=PipelineStatus.COMPLETED,
            )
            if not release_ok:
                logger.warning(
                    "[job_urls_pipeline_fsm] Release failed (mismatched worker or lost lease?) url=%s iter=%s",
                    url,
                    iter_,
                )

        except Exception:
            logger.exception(
                "[job_urls_pipeline_fsm] Error completing/stepping url=%s iter=%s",
                url,
                iter_,
            )
            # Best-effort release with ERROR
            release_one(
                url=url,
                iteration=iter_,
                worker_id=worker_id,
                final_status=PipelineStatus.ERROR,
            )

    logger.info(
        "[job_urls_pipeline_fsm] done — found=%d, claimed=%d, completed=%d, enqueued_next=%d",
        found,
        claimed,
        completed,
        enqueued,
    )
    return {
        "found": found,
        "claimed": claimed,
        "completed": completed,
        "enqueued_next": enqueued,
    }
