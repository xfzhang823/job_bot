"""
db_io/db_utils.py

Utility functions for querying DuckDB pipeline control metadata.

These help inspect the pipeline_control table for FSM stage tracking,
progress monitoring, and diagnostics.

Each function returns raw values or DataFrames depending on the context.
"""

# Standard
import logging
from typing import (
    Any,
    Iterable,
    Optional,
    List,
    get_origin,
    get_args,
    Union,
    Sequence,
    Type,
    TypeVar,
)
from datetime import datetime
import json
from enum import Enum
from pydantic import BaseModel, HttpUrl
from pydantic_core import PydanticUndefined
import pandas as pd

# Project level
from job_bot.db_io.get_db_connection import get_db_connection
from job_bot.db_io.pipeline_enums import (
    PipelineStage,
    PipelineStatus,
    PipelineProcessStatus,
    TableName,
)
from job_bot.db_io import decision_flag

logger = logging.getLogger(__name__)

E = TypeVar("E")  # enum type


def align_df_with_schema(
    df: pd.DataFrame, schema_columns: List[str], strict: bool = False
) -> pd.DataFrame:
    """
    Final validation and alignment step before inserting into DuckDB.

    This function:
    - Ensures all schema columns are present
    - Reorders columns to match DuckDB column order
    - Deduplicates fully identical rows (before timestamps are added)
    - Cleans and coerces problematic types (e.g., nested dicts/lists, NA values)
    - Optionally validates column types if schema typing is available
    (future enhancement)

    Args:
        - df (pd.DataFrame): Input DataFrame to validate and clean.
        - schema_columns (List[str]): Expected column names in final DuckDB order.
        - strict:
            if strict=True:
                Any missing columns (before step 1) → error.
            if False:
                just warnings;
                prune extras; create missing columns (w/t None)
                proceed.
    Returns:
        pd.DataFrame: A cleaned and schema-aligned DataFrame.
    """
    df = df.copy()

    # 1) dedupe
    original_len = len(df)
    df.drop_duplicates(inplace=True)
    deduped_len = len(df)
    if deduped_len < original_len:
        logger.info(f"Dropped {original_len - deduped_len} duplicate rows")

    # 2) detect mismatches against the incoming df (BEFORE adding)
    missing_before = [c for c in schema_columns if c not in df.columns]
    extra_in_df = [c for c in df.columns if c not in schema_columns]

    # 3) strict handling of missing
    if strict and missing_before:
        raise ValueError(f"Missing columns in DataFrame: {missing_before}")

    # 4) always drop extras (be explicit + observable)
    if extra_in_df:
        logger.warning(f"Dropping unexpected columns: {extra_in_df}")
        df = df.drop(columns=extra_in_df)

    # 5) add any still-missing schema columns (fill with None)
    for col in schema_columns:
        if col not in df.columns:
            df[col] = None
            logger.debug(f"Added missing column: {col}")

    # 6) reorder to schema
    df = df[[col for col in schema_columns]]

    # Coerce problematic types
    for col in df.columns:
        # Convert nested objects (dicts/lists) to JSON strings
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            logger.debug(f"Serializing nested structures in column: {col}")
            df[col] = df[col].apply(
                lambda x: (
                    json.dumps(x, ensure_ascii=False)
                    if isinstance(x, (dict, list))
                    else x
                )
            )

        # Convert Pydantic models or HttpUrls to string
        if df[col].apply(lambda x: isinstance(x, (BaseModel, HttpUrl))).any():
            logger.debug(f"Converting BaseModel or HttpUrl to string in column: {col}")
            df[col] = df[col].apply(
                lambda x: str(x) if isinstance(x, (BaseModel, HttpUrl)) else x
            )

        # Replace pandas NA or numpy NaN with Python None
        if df[col].isna().any():
            logger.debug(f"Replacing NA/nulls with None in column: {col}")
            df[col] = df[col].where(pd.notna(df[col]), None)

    logger.info(f"Aligned DataFrame with schema: {schema_columns}")
    return df


# ✅ Core FSM Worklist Search
def get_urls_by_status(status: PipelineStatus) -> List[str]:
    """
    Return a list of all URLs matching the given pipeline status.

    Args:
        - status (PipelineStatus): The status to filter by
        (e.g., 'new', 'in_progress').

    Returns:
        List[str]: A list of matching URLs.
    """
    con = get_db_connection()
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
    con = get_db_connection()
    df = con.execute(
        """
        SELECT url FROM pipeline_control WHERE stage = ?
    """,
        (stage.value,),
    ).df()
    return df["url"].tolist()


def get_urls_by_stage_and_status(
    stage: PipelineStage,
    *,
    status: Union[PipelineStatus, Sequence[PipelineStatus]] = PipelineStatus.NEW,
    version: Optional[str] = None,
    iteration: Optional[int] = None,
    active_urls_only: bool = True,
) -> List[str]:
    """
    Return URLs from `pipeline_control` matching a specific stage and one or more status,
    with optional filtering by version, iteration, and (by default) active lifecycle.

    Lifecycle gating:
        When active_urls_only=True, only rows with process_status IN ('new','running')
        are considered. Set to False to ignore lifecycle.

    Args:
        stage: Pipeline stage (Enum) to match.
        status: Single status or a sequence of statuses to include (default = NEW).
        version: Optional version filter (e.g., "original").
        iteration: Optional iteration number.
        active_urls_only: If True (default), include only NEW/RUNNING lifecycle rows.

    Returns:
        List[str]: DISTINCT URLs matching the criteria, ordered by recency.
    """
    # Normalize to a non-empty list of PipelineStatus
    if isinstance(status, PipelineStatus):
        status_list: List[PipelineStatus] = [status]
    else:
        status_list = list(status)
        if not status_list:
            # Defensive: don't emit invalid SQL
            return []

    filters = ["stage = ?"]
    params: List[object] = [stage.value]

    # Stage-local status filter
    filters.append(f"status IN ({', '.join(['?'] * len(status_list))})")
    params.extend(s.value for s in status_list)

    # Optional lifecycle (process_status) gating
    if active_urls_only:
        process_allowed = (PipelineProcessStatus.NEW, PipelineProcessStatus.RUNNING)
        filters.append(f"process_status IN ({', '.join(['?'] * len(process_allowed))})")
        params.extend(p.value for p in process_allowed)

    # Optional version/iteration filters
    if version is not None:
        filters.append("version = ?")
        params.append(version)
    if iteration is not None:
        filters.append("iteration = ?")
        params.append(iteration)

    sql = f"""
        SELECT DISTINCT url
        FROM pipeline_control
        WHERE {' AND '.join(filters)}
        ORDER BY updated_at DESC, created_at DESC
    """

    con = get_db_connection()
    try:
        df: pd.DataFrame = con.execute(sql, params).fetchdf()
        return df["url"].tolist()
    finally:
        con.close()


# URL pre-filtering utility
def get_urls_from_pipeline_control(
    *,
    status: Optional[
        Union[PipelineStatus, str, Sequence[Union[PipelineStatus, str]]]
    ] = None,
    stage: Optional[
        Union[PipelineStage, str, Sequence[Union[PipelineStage, str]]]
    ] = None,
    active_urls_only: bool = True,
    limit: Optional[int] = None,
    table: TableName = TableName.PIPELINE_CONTROL,
    con=None,
) -> List[str]:
    """
    Return DISTINCT URLs from `pipeline_control` filtered by status/stage.

    - `status` and `stage` can be a single value or a list (enum or string).
    - When `active_urls_only=True`, exclude only rows with
        process_status ∈ {'skipped','completed'}.
      (NULL, 'running', 'new', 'error', etc. are allowed.)
    - Ordered by updated_at DESC.
    """
    owns_con = con is None
    if owns_con:
        con = get_db_connection()

    try:
        where: List[str] = []
        params: List[Any] = []

        # Normalize filters
        status_vals = _to_values(status, PipelineStatus, lower=True)
        stage_vals = _to_values(
            stage, PipelineStage, lower=False
        )  # stage names are already canonical

        if status_vals:
            placeholders = ",".join(["?"] * len(status_vals))
            where.append(f"status IN ({placeholders})")
            params.extend(status_vals)

        if stage_vals:
            placeholders = ",".join(["?"] * len(stage_vals))
            where.append(f"stage IN ({placeholders})")
            params.extend(stage_vals)

        if active_urls_only:
            # Include NULL and everything except the terminal process states you want to exclude.
            where.append(
                "(process_status IS NULL OR process_status NOT IN ('skipped','completed'))"
            )

        where_sql = " AND ".join(where) if where else "1=1"

        sql = f"""
            SELECT DISTINCT url
            FROM {table.value}
            WHERE {where_sql}
            ORDER BY updated_at DESC
            { 'LIMIT ?' if limit is not None else '' }
        """
        if limit is not None:
            params.append(limit)

        rows = con.execute(sql, params).fetchall()
        return [r[0] for r in rows]
    finally:
        if owns_con:
            con.close()


# ✅ URL Lookup Utilities
def get_pipeline_state(url: str) -> pd.DataFrame:
    """
    Return the full pipeline_control row for a given URL.

    Args:
        url (str): The job posting URL.

    Returns:
        pd.DataFrame: A single-row DataFrame containing the pipeline state for the given URL.
    """
    con = get_db_connection()
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
    con = get_db_connection()
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
    con = get_db_connection()
    return con.execute(
        """
        SELECT url, stage, status, timestamp
        FROM pipeline_control
        ORDER BY timestamp DESC
        LIMIT ?
    """,
        (limit,),
    ).df()


# ✅ DB Schema Generation Utilities
def duckdb_type_from_annotation(annotation) -> str:
    """
    Maps a Pydantic annotation or standard Python type to a DuckDB-compatible SQL type.

    Args:
        annotation: The type annotation from a Pydantic field.

    Returns:
        str: A valid DuckDB SQL type (e.g., TEXT, INTEGER, DOUBLE).
    """
    base = get_origin(annotation) or annotation

    # ✅ Handle Optional[X] / Union[X, None]
    if base is Union:
        args = get_args(annotation)
        # remove NoneType from union
        base = next((arg for arg in args if arg is not type(None)), str)

    # ✅ Enum → TEXT
    if isinstance(base, type) and issubclass(base, Enum):
        return "TEXT"

    # ✅ Pydantic-specific / URL
    if base in [HttpUrl]:
        return "TEXT"

    # ✅ Date/time
    if base in [datetime]:
        return "TIMESTAMP"

    # ✅ Basic primitives
    if base in [str]:
        return "TEXT"
    if base in [int]:
        return "INTEGER"
    if base in [float]:
        return "DOUBLE"
    if base in [bool]:
        return "BOOLEAN"

    return "TEXT"  # ✅ Fallback for unknown or unsupported types


def generate_table_schema_from_model(
    model: type[BaseModel],
    table_name: str,
    primary_keys: list[str] | None = None,
) -> str:
    """
    Auto-generates a DuckDB CREATE TABLE DDL statement from a Pydantic model.

    Args:
        model (type[BaseModel]): A Pydantic model class (not instance).
        table_name (str): Desired DuckDB table name.
        primary_keys (list[str]): Column names to use as primary key.

    Returns:
        str: SQL CREATE TABLE statement.
    """
    lines = []

    for name, field in model.model_fields.items():
        annotation = field.annotation
        duckdb_type = duckdb_type_from_annotation(annotation)

        # Static default support (including Enums)
        default_clause = ""
        if field.default is not PydanticUndefined:
            val = field.default
            if val is None:
                default_clause = "DEFAULT NULL"
            elif isinstance(val, Enum):
                val = val.value
                default_clause = f"DEFAULT '{val}'"
            elif isinstance(val, str):
                default_clause = f"DEFAULT '{val}'"
            elif isinstance(val, bool):
                default_clause = f"DEFAULT {'TRUE' if val else 'FALSE'}"
            else:
                default_clause = f"DEFAULT {val}"

        lines.append(f"{name} {duckdb_type} {default_clause}".strip())

    pk_clause = f", PRIMARY KEY ({', '.join(primary_keys)})" if primary_keys else ""

    # 🔧 Use intermediate variable to avoid backslash in f-string
    field_block = ",\n    ".join(lines)
    return (
        f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
        f"    {field_block}"
        f"{pk_clause}\n);"
    )


# ---- Transition Helpers -------


def get_urls_ready_for_transition(
    stage: PipelineStage, limit: int | None = None
) -> list[str]:
    """
    Worklist picker: only rows with (decision_flag=1).
    """
    con = get_db_connection()
    try:
        sql = f"""
            SELECT url
            FROM {TableName.PIPELINE_CONTROL.value}
            WHERE stage = ? AND decision_flag = 1
        """
        params: list[object] = [stage.value]  # <-- allow str + int
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)  # keep as int, not str
        df = con.execute(sql, params).df()
        return df["url"].tolist()
    finally:
        con.close()


def _to_values(
    item: Optional[Union[str, E, Sequence[Union[str, E]]]],
    enum_cls: Optional[Type[E]] = None,
    *,
    lower: bool = True,
) -> List[str]:
    """
    Normalize an enum/string or a sequence of them into a list of string values.
    If enum_cls is provided, enum members become their .value; strings are optionally lowered.
    """
    if item is None:
        return []
    seq: Iterable[Union[str, E]]
    if isinstance(item, (str,)) or (enum_cls and isinstance(item, enum_cls)):
        seq = [item]  # single → list
    else:
        seq = item  # assume sequence

    out: List[str] = []
    for x in seq:
        if enum_cls and isinstance(x, enum_cls):
            out.append(getattr(x, "value"))
        else:
            s = str(x)
            out.append(s.lower() if lower else s)
    return out
