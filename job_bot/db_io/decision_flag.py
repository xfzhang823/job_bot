"""
db_io/decision_flag.py

Centralized utilities for updating and synchronizing the `decision_flag`
column in the pipeline_control table.

`decision_flag` serves as the explicit approval or readiness gate for each
pipeline stage:
    • 0 = pending or not approved
    • 1 = approved / ready to proceed

This module provides safe, minimal primitives for:
    - setting or clearing the flag for a given (url, stage)
    - bulk recomputation of decision flags based on pipeline status
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
    PipelineProcessStatus,
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
    con : Optional[DuckDBPyConnection], default=None
        Existing DuckDB connection to reuse.
        If None, a new connection is opened.

    Returns
    -------
    int
        Number of rows updated (0 if no matching row).

    Notes
    -----
    - Does not modify `transition_flag` or `status`.
    - Automatically updates `updated_at` timestamp.
    - Closes the connection if it was created inside this function.
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
    """
    Approve a URL for a given stage.

    Sets `decision_flag = 1` for (url, stage).

    Parameters
    ----------
    url : str
        URL identifying the record.
    stage : PipelineStage
        Stage to approve.
    con : Optional[DuckDBPyConnection], default=None
        Reuse an open DuckDB connection if provided.

    Returns
    -------
    int
        Number of rows affected.
    """
    return set_decision_flag(url, stage, 1, con=con)


def clear(
    url: str, stage: PipelineStage, con: Optional[DuckDBPyConnection] = None
) -> int:
    """
    Clear (disapprove) a URL for a given stage.

    Sets `decision_flag = 0` for (url, stage).

    Parameters
    ----------
    url : str
        URL identifying the record.
    stage : PipelineStage
        Stage to clear.
    con : Optional[DuckDBPyConnection], default=None
        Reuse an open DuckDB connection if provided.

    Returns
    -------
    int
        Number of rows affected.
    """
    return set_decision_flag(url, stage, 0, con=con)


def sync_all(con: Optional[DuckDBPyConnection] = None) -> int:
    """
    Recompute `decision_flag` across the entire pipeline_control table.

    Logic
    -----
    decision_flag = 0 if status/process_status ∈ {COMPLETED, SKIPPED}
                    else COALESCE(decision_flag, 0)

    This ensures completed or skipped stages are always cleared,
    while preserving any existing approvals in active or
    pending stages.

    Parameters
    ----------
    con : Optional[DuckDBPyConnection], default=None
        Existing database connection. A new one is opened if None.

    Returns
    -------
    int
        Number of rows updated.

    Notes
    -----
    - This operation does not alter `transition_flag`.
    - Use this only for global synchronization or cleanup passes.
    - Automatically updates `updated_at` timestamps.
    """
    owns = con is None
    if owns:
        con = get_db_connection()
    try:
        st_completed = PipelineStatus.COMPLETED.value
        st_skipped = getattr(PipelineStatus, "SKIPPED", PipelineStatus.COMPLETED).value
        ps_completed = PipelineProcessStatus.COMPLETED.value
        ps_skipped = getattr(
            PipelineProcessStatus, "SKIPPED", PipelineProcessStatus.COMPLETED
        ).value

        res = con.execute(
            f"""
            UPDATE {PC.value}
            SET decision_flag = CASE
                                   WHEN status IN (?, ?) OR process_status IN (?, ?)
                                     THEN 0
                                   ELSE COALESCE(decision_flag, 0)
                                END,
                updated_at = now()
            """,
            (st_completed, st_skipped, ps_completed, ps_skipped),
        )
        return getattr(res, "rowcount", 0)
    finally:
        if owns:
            con.close()


def _stage_eq() -> str:
    """
    Internal helper for stage comparison clause.

    Returns
    -------
    str
        A SQL fragment that performs case-insensitive equality
                on `stage`.
        Example: "lower(stage) = lower(?)"
    """
    return "lower(stage) = lower(?)"
