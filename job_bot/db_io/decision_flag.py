"""
db_io/decision_flag.py

Centralized utilities for updating and synchronizing the `decision_flag`
column in the pipeline_control table.

Semantics:
    • NULL = undecided (default; eligible if human gate allows)
    • 1    = approved / ready to proceed
    • 0    = no-go / explicitly blocked

This module provides safe, minimal primitives for:
    - setting or clearing the flag for a given (url, stage)
    - bulk sync helpers that respect the human gate (`task_state`)
It deliberately avoids touching `transition_flag` or other state columns.
"""

from __future__ import annotations
import logging
from typing import Optional
from duckdb import DuckDBPyConnection

from job_bot.db_io.db_utils import get_db_connection
from job_bot.db_io.pipeline_enums import (
    TableName,
    PipelineStatus,
    PipelineStage,
    PipelineTaskState,  # Enum now: READY, PAUSED, SKIP, HOLD
)

logger = logging.getLogger(__name__)
PC = TableName.PIPELINE_CONTROL


def set_decision_flag(
    url: str, stage: PipelineStage, value: int, con: Optional[DuckDBPyConnection] = None
) -> int:
    """
    Set `decision_flag` for a specific (url, stage) pair.

    Parameters
    ----------
    url : str
        Job posting or resume URL uniquely identifying the record.
    stage : PipelineStage
        Pipeline stage whose decision flag will be updated.
    value : int
        Integer 0 or 1. Values are coerced to bool → int.
        (Use a direct SQL UPDATE to set NULL if needed.)
    con : Optional[DuckDBPyConnection], default=None
        Existing DuckDB connection to reuse.
        If None, a new connection is opened.

    Returns
    -------
    int
        Number of rows updated (0 if no matching row).
    """
    owns = con is None
    if owns:
        con = get_db_connection()
    try:
        res = con.execute(
            f"""
            UPDATE {PC.value}
            SET decision_flag = ?,
                updated_at = now()
            WHERE url = ? AND {_stage_eq()}
            """,
            (int(bool(value)), url, stage.value),
        )
        return getattr(res, "rowcount", 0)
    finally:
        if owns:
            con.close()


def approve(
    url: str, stage: PipelineStage, con: Optional[DuckDBPyConnection] = None
) -> int:
    """Approve a URL for a given stage: sets `decision_flag = 1`."""
    return set_decision_flag(url, stage, 1, con=con)


def clear(
    url: str, stage: PipelineStage, con: Optional[DuckDBPyConnection] = None
) -> int:
    """
    Disapprove / block a URL for a given stage: sets `decision_flag = 0`.
    (Use direct SQL if you want to set NULL/undecided).
    """
    return set_decision_flag(url, stage, 0, con=con)


def mark_all_as_ready(
    *,
    stage: str | None = None,
    url: str | None = None,
    con: Optional[DuckDBPyConnection] = None,
) -> int:
    """
    Set decision_flag=1 and ensure task_state='READY' for rows that can run again.
    Used to re-approve or resume paused/held rows.

    Scope must be constrained by stage or url.
    """
    if stage is None and url is None:
        raise ValueError("Must provide at least stage or url scope.")

    owns = con is None
    if owns:
        con = get_db_connection()
    try:
        where = [
            "status = 'NEW'",
            # Only flip rows that are not actively leased and not permanently skipped.
            "task_state IN ('PAUSED', 'READY', 'HOLD')",
            # Only rows that are not already approved (includes NULL)
            "(decision_flag IS NULL OR decision_flag = 0)",
        ]

        params: list[str] = []
        if stage:
            where.append("stage = ?")
            params.append(stage)
        if url:
            where.append("url = ?")
            params.append(url)

        res = con.execute(
            f"""
            UPDATE {PC.value}
            SET decision_flag = 1,
                task_state   = 'READY',
                updated_at   = now()
            WHERE {" AND ".join(where)}
            """,
            params,
        )
        return getattr(res, "rowcount", 0) or 0
    finally:
        if owns:
            con.close()


def sync_all(con: Optional[DuckDBPyConnection] = None) -> int:
    """
    Normalize `decision_flag` across the entire pipeline_control table.

    Policy (minimal, human-gate–aware):
      • If task_state = 'SKIP' → force decision_flag = 0 (no-go)
      • Otherwise               → leave decision_flag as-is (including NULL/1/0)

    Rationale:
      - We keep decision_flag independent of machine `status`.
      - `NEW/IN_PROGRESS/COMPLETED/ERROR` do not force approvals to flip.
      - Worklists already gate on: (decision_flag IS NULL OR decision_flag = 1).
    """
    owns = con is None
    if owns:
        con = get_db_connection()
    try:
        res = con.execute(
            f"""
            UPDATE {PC.value}
            SET decision_flag = 0,
                updated_at   = now()
            WHERE task_state = 'SKIP'
              AND (decision_flag IS DISTINCT FROM 0)
            """
        )
        return getattr(res, "rowcount", 0)
    finally:
        if owns:
            con.close()


def _stage_eq() -> str:
    """
    Internal helper for stage comparison clause (case-insensitive).
    Example: "lower(stage) = lower(?)"
    """
    return "lower(stage) = lower(?)"
