"""
db_io/db_utils.py

Utility functions for querying DuckDB pipeline control metadata.

These help inspect the pipeline_control table for FSM stage tracking,
progress monitoring, and diagnostics.

Each function returns raw values or DataFrames depending on the context.
"""

from typing import Optional, List
import pandas as pd
from db_io.duckdb_adapter import get_duckdb_connection
from db_io.schema_definitions import PipelineStage, PipelineStatus


# ✅ Core FSM Worklist Search
from db_io.duckdb_adapter import get_duckdb_connection
from models.duckdb_table_models import PipelineStatus
from db_io.schema_definitions import PipelineStage
from typing import List, Optional


def get_urls_by_status(status: PipelineStatus) -> List[str]:
    """
    Return a list of all URLs matching the given pipeline status.

    Args:
        - status (PipelineStatus): The status to filter by
        (e.g., 'new', 'in_progress').

    Returns:
        List[str]: A list of matching URLs.
    """
    con = get_duckdb_connection()
    df = con.execute(
        """
        SELECT url FROM pipeline_control WHERE status = ?
    """,
        (status.value,),
    ).df()
    return df["url"].tolist()


def get_urls_by_stage(stage: PipelineStage) -> List[str]:
    """
    Return a list of all URLs currently in the specified pipeline stage.

    Args:
        - stage (PipelineStage): Enum value representing the pipeline stage
        (e.g., PipelineStage.JOB_POSTINGS).

    Returns:
        List[str]: A list of matching job posting URLs.
    """
    con = get_duckdb_connection()
    df = con.execute(
        """
        SELECT url FROM pipeline_control WHERE stage = ?
    """,
        (stage.value,),
    ).df()
    return df["url"].tolist()


def get_urls_by_stage_and_status(
    stage: PipelineStage,
    status: PipelineStatus = PipelineStatus.NEW,
    version: Optional[str] = None,
    iteration: Optional[int] = None,
) -> List[str]:
    """
    Return URLs from the pipeline_control table matching a specific stage and status,
    with optional filtering by version and iteration.

    Args:
        stage (PipelineStage): Pipeline stage (as Enum) to match.
        status (PipelineStatus): Status to filter on (e.g., new, in_progress).
        version (Optional[str]): Optional version filter (e.g., "original").
        iteration (Optional[int]): Optional iteration number.

    Returns:
        List[str]: A list of job posting URLs matching the criteria.
    """
    filters = ["stage = ?", "status = ?"]  # parameterized SQL query
    params: List[str | int] = [
        stage.value,
        status.value,
    ]  # Include int b/c iteration is int

    if version:
        filters.append("version = ?")
        params.append(version)
    if iteration is not None:
        filters.append("iteration = ?")
        params.append(iteration)

    sql = f"""
        SELECT DISTINCT url
        FROM pipeline_control
        WHERE {' AND '.join(filters)}
    """
    con = get_duckdb_connection()
    df = con.execute(sql, params).df()
    return df["url"].tolist()


# ✅ URL Lookup Utilities
def get_pipeline_state(url: str) -> pd.DataFrame:
    """
    Return the full pipeline_control row for a given URL.

    Args:
        url (str): The job posting URL.

    Returns:
        pd.DataFrame: A single-row DataFrame containing the pipeline state for the given URL.
    """
    con = get_duckdb_connection()
    return con.execute(
        """
        SELECT * FROM pipeline_control
        WHERE url = ?
    """,
        (url,),
    ).df()


def get_current_stage_for_url(url: str) -> Optional[str]:
    """
    Return the current pipeline stage for a given URL.

    Args:
        url (str): The job posting URL.

    Returns:
        Optional[str]: The current stage if found, else None.
    """
    df = get_pipeline_state(url)
    if df.empty:
        return None
    return df.iloc[0]["stage"]


# ✅ Summary Utilities
def get_stage_progress_counts() -> pd.DataFrame:
    """
    Return a count of records grouped by pipeline stage and status.

    Returns:
        pd.DataFrame: A summary table showing (stage, status, count).
    """
    con = get_duckdb_connection()
    return con.execute(
        """
        SELECT stage, status, COUNT(*) as count
        FROM pipeline_control
        GROUP BY stage, status
        ORDER BY stage, status
    """
    ).df()


def get_recent_urls(limit: int = 10) -> pd.DataFrame:
    """
    Return the most recently updated job URLs in the pipeline.

    Args:
        limit (int): Number of recent URLs to return (default = 10).

    Returns:
        pd.DataFrame: DataFrame with columns (url, stage, status, timestamp).
    """
    con = get_duckdb_connection()
    return con.execute(
        """
        SELECT url, stage, status, timestamp
        FROM pipeline_control
        ORDER BY timestamp DESC
        LIMIT ?
    """,
        (limit,),
    ).df()
