"""pipelines_with_fsm/pipe_control_auto_transitions_pipeline.py"""

from __future__ import annotations
import logging

from job_bot.db_io.pipeline_enums import PipelineStage, TableName
from job_bot.fsm.pipeline_control_sync import (
    fsm_retry_errors_all,
    fsm_auto_advance_completed_all,
    sync_decision_flags_all,
)

logger = logging.getLogger(__name__)


def run_pipe_control_auto_transition_pipeline_fsm(
    *,
    table: TableName = TableName.PIPELINE_CONTROL,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Status-only control-plane housekeeping:
      - ERROR -> NEW (same stage)
      - COMPLETED -> next stage + NEW
      - recompute decision/transition flags (table-wide)

    Returns:
        (retried_count, advanced_count)
    """
    retried = fsm_retry_errors_all(dry_run=dry_run, table=table)
    advanced = fsm_auto_advance_completed_all(dry_run=dry_run, table=table)
    if not dry_run:
        sync_decision_flags_all(table_name=table)

    logger.info(
        "🧮 Status auto-transitions: retried=%s, advanced=%s", retried, advanced
    )
    return retried, advanced
