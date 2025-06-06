"""
pydantic_model_loaders_from_db.py

This module provides typed, Pydantic-based loaders for pulling structured content
from DuckDB tables. It supports both per-URL (FSM-aligned) and batch (dashboard/QA)
access patterns.

Each loader:
- Pulls a group of rows corresponding to a unique job posting (via url,
version, iteration)
- Rehydrates those rows into a validated Pydantic model
- Returns that model for use in FSM-driven pipeline stages or batch reporting

The rehydration logic ensures symmetry with the flattening process used
during ingestion.

-------------------------------------------------------------
 Loader Function                                        | Return Type
-------------------------------------------------------------
 load_job_postings_for_url_from_db(...)                 | JobSiteResponse
 load_extracted_requirements_for_url_from_db(...)       | RequirementsResponse
 load_flattened_requirements_for_url_from_db(...)       | Requirements
 load_flattened_responsibilities_for_url_from_db(...)   | Responsibilities
 load_edited_responsibilities_for_url_from_db(...)      | NestedResponsibilities
-------------------------------------------------------------
 load_all_job_postings_file_model_from_db()       | JobPostingsFile
 load_all_extracted_requirements_model_from_db()  | ExtractedRequirementsFile
 load_all_job_urls_from_db()                      | JobPostingUrlsFile
 get_urls_for_status_from_pipeline_control (...)  | List[str]
-------------------------------------------------------------

Key design goals:
- ✅ Per-URL isolation to match FSM processing granularity
- ✅ Centralized rehydration + Pydantic validation for data integrity
- ✅ Flexible enough for batch analytics and iterative development

>>> Example Usage:

    # * Load from requirements table
    urls = get_urls_by_status(stage=\"evaluation\", version=\"original\")
    for url in urls:
        requirements = load_flattened_requirements_for_url_from_db(url, version=\"original\")

    # * Force to load w/t disabled filter
    url = "https://example.com/job-posting"
    requirements_model = load_flattened_requirements_for_url_from_db(
    url=url,
    status=None,  # 👈 disables status filtering
    iteration=0,
)
"""

import logging
from typing import Union, Optional, cast, Dict, List
import pandas as pd
import duckdb
from pydantic import ValidationError, HttpUrl, TypeAdapter

# User defined
from models.resume_job_description_io_models import (
    JobPostingUrlsBatch,
    JobPostingsBatch,
    ExtractedRequirementsBatch,
    Requirements,
    Responsibilities,
    NestedResponsibilities,
    SimilarityMetrics,
)
from models.duckdb_table_models import PipelineState
from db_io.duckdb_adapter import get_duckdb_connection
from db_io.flatten_and_rehydrate import *
from db_io.db_schema_registry import TableName, PipelineStatus

logger = logging.getLogger(__name__)


# * db_loader_config
DB_LOADER_CONFIG = {
    TableName.JOB_URLS: {
        "rehydrate_fn": rehydrate_job_urls_from_table,
        "filterable": False,
        "returned_model": JobPostingUrlMetadata,
    },
    TableName.JOB_POSTINGS: {
        "rehydrate_fn": rehydrate_job_postings_from_table,
        "filterable": False,
        "returned_model": JobPostingsBatch,
    },
    TableName.EXTRACTED_REQUIREMENTS: {
        "rehydrate_fn": rehydrate_extracted_requirements_from_table,
        "filterable": False,
        "returned_model": ExtractedRequirementsBatch,
    },
    TableName.FLATTENED_REQUIREMENTS: {
        "rehydrate_fn": rehydrate_requirements_from_table,
        "filterable": False,
        "returned_model": Requirements,
    },
    TableName.FLATTENED_RESPONSIBILITIES: {
        "rehydrate_fn": rehydrate_responsibilities_from_table,
        "filterable": False,
        "returned_model": Responsibilities,
    },
    TableName.EDITED_RESPONSIBILITIES: {
        "rehydrate_fn": rehydrate_nested_responsibilities_from_table,
        "filterable": False,
        "returned_model": NestedResponsibilities,
    },
}


# URL pre-filtering utility
def get_urls_from_pipeline_control(
    status: PipelineStatus = PipelineStatus.IN_PROGRESS,
    url: Optional[str] = None,
    iteration: Optional[int] = None,
) -> List[str]:
    """
    Returns a list of distinct URLs from the pipeline_control table
    filtered by status and optional URL/iteration.

    Args:
        - status (PipelineStatus): Status to match (e.g., 'new', 'in_progress').
        - url (Optional[str]): Optional job URL filter.
        - iteration (Optional[int]): Optional iteration filter.

    Returns:
        List[str]: List of matching job posting URLs.
    """
    filters = ["status = ?"]
    params: List[str | int] = [status.value]

    if url:
        filters.append("url = ?")
        params.append(url)
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


# db_loader
def db_loader(
    table: TableName,
    url: Optional[str] = None,
    status: Optional[PipelineStatus] = None,
    iteration: Optional[int] = None,
) -> Any:
    """
    Loads and rehydrates data from DuckDB for a given table using
    a configured rehydration function.

    Supports optional filtering by URL, status, and iteration—only
    if the table is marked as `filterable` in `DB_LOADER_CONFIG`.

    Args:
        - table (TableName): Enum name of the DuckDB table
        (e.g., TableName.FLATTENED_REQUIREMENTS).
        - url (Optional[str]): Filter for a specific job posting URL.
        - status (Optional[PipelineStatus]): Filter for pipeline status
        (e.g., 'in_progress', 'new').
        - iteration (Optional[int]): Filter for iteration index (default is 0).

    Returns:
        Any: A validated Pydantic model corresponding to the table's schema.

    Example:
        >>> from db_io.pydantic_model_loaders_from_db import db_loader
        >>> from db_io.schema_definitions import TableName, PipelineStatus
        >>> model = db_loader(
        ...     table=TableName.FLATTENED_REQUIREMENTS,
        ...     url="https://example.com/job123",
        ...     status=PipelineStatus.IN_PROGRESS,
        ...     iteration=0
        ... )
        >>> print(model.requirements)  # access validated model fields
    """
    config = DB_LOADER_CONFIG.get(table)
    if not config:
        raise ValueError(f"No loader config found for table: {table}")

    rehydrate_fn = config["rehydrate_fn"]
    filterable = config.get("filterable", False)

    filters = []
    params = []

    if filterable:
        if url:
            filters.append("url = ?")
            params.append(url)
        if status:
            filters.append("status = ?")
            params.append(status.value)
        if iteration is not None:
            filters.append("iteration = ?")
            params.append(iteration)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = f"SELECT * FROM {table.value} {where_clause} ORDER BY timestamp DESC"

    con = get_duckdb_connection()
    df: pd.DataFrame = con.execute(sql, params).df()
    con.close()

    return rehydrate_fn(df)


# * FSM-compatible per-URL model loaders
def load_job_postings_for_url_from_db(
    url: str, status: Optional[PipelineStatus] = PipelineStatus.NEW, iteration: int = 0
) -> JobPostingsBatch:
    """
    Loads the job posting metadata for a specific URL.
    Returns a single JobSiteResponse (not the entire dict).

    >>> Example: How to disable filtering (search all statuses)
        url = "https://example.com/job-posting"
        job_posting_model = load_job_postings_for_url_from_db(
            url=url,
            status=None,  # 👈 disables status filtering (forces search across all statuses)
            iteration=0,
        )
    """
    # Fetch the entire table first and then select data for the url
    model = db_loader(
        TableName.JOB_POSTINGS,
        url=url,
        status=status,
        iteration=iteration,
    )
    try:
        return model.root[url]
    except KeyError:
        raise ValueError(
            f"No entry found for URL {url} in table {TableName.JOB_POSTINGS}"
        )


def load_extracted_requirements_for_url_from_db(
    url: str, status: PipelineStatus = PipelineStatus.IN_PROGRESS, iteration: int = 0
) -> RequirementsResponse:
    """
    Loads the extracted job requirements for a specific URL.
    Returns a single RequirementsResponse object.
    """
    # Fetch the entire table first and then select data for the url
    return db_loader(
        TableName.EXTRACTED_REQUIREMENTS, url=url, status=status, iteration=iteration
    )


def load_flattened_requirements_for_url_from_db(
    url: str, status: PipelineStatus = PipelineStatus.IN_PROGRESS, iteration: int = 0
) -> Requirements:
    """
    Loads the flattened requirements (key-value) for a specific job URL from
    the 'flattened_requirements' table.
    Typically used for matching against resume responsibilities.
    """
    return db_loader(
        TableName.FLATTENED_REQUIREMENTS,
        url=url,
        status=status,
        iteration=iteration,
    )


def load_flattened_responsibilities_for_url_from_db(
    url: str, status: PipelineStatus = PipelineStatus.IN_PROGRESS, iteration: int = 0
) -> Responsibilities:
    """
    Loads flattened resume responsibilities for a specific URL from
    the 'flattened_responsibilities' table.
    Typically used in evaluation or editing steps in the pipeline.
    """
    return db_loader(
        TableName.FLATTENED_RESPONSIBILITIES,
        url=url,
        status=status,
        iteration=iteration,
    )


def load_edited_responsibilities_for_url_from_db(
    url: str, status: PipelineStatus = PipelineStatus.IN_PROGRESS, iteration: int = 0
) -> NestedResponsibilities:
    """
    Loads LLM-edited responsibilities for a specific URL from
    the 'edited_responsibilities' table.
    Rehydrates nested responsibilities grouped by requirement.
    """
    return db_loader(
        TableName.EDITED_RESPONSIBILITIES,
        url=url,
        status=status,
        iteration=iteration,
    )


# * Batch loaders for dashboard / bulk usage
def load_all_job_postings_file_model_from_db() -> JobPostingsBatch:
    """
    Loads all job posting metadata records from the 'job_postings' table.
    Used for batch analysis, dashboards, or revalidation.
    """
    return db_loader(TableName.JOB_POSTINGS)


def load_all_extracted_requirements_model_from_db() -> ExtractedRequirementsBatch:
    """
    Loads all extracted job requirements from the 'extracted_requirements' table.
    Returns a dictionary of RequirementsResponse objects keyed by job URL.
    """
    return db_loader(TableName.EXTRACTED_REQUIREMENTS)


def load_all_job_urls_from_db() -> JobPostingUrlsBatch:
    """
    Loads the full set of job URLs and their metadata from the 'job_urls' table.
    Used as a central registry for all known jobs in the pipeline.
    """
    return db_loader(TableName.JOB_URLS)


def load_pipeline_state_for_url_from_db(
    url: str,
    version: str = "original",
    iteration: int = 0,
) -> PipelineState:
    """
    Loads the pipeline_control row for a specific job URL and version/iteration.
    Used to drive FSM transitions and state introspection.
    """

    sql = """
        SELECT * FROM pipeline_control
        WHERE url = ? AND version = ? AND iteration = ?
        ORDER BY last_updated DESC
        LIMIT 1
    """
    con = get_duckdb_connection()
    df = con.execute(sql, (url, version, iteration)).df()

    if df.empty:
        raise ValueError(f"No pipeline state found for URL: {url}")

    return PipelineState.model_validate(df.iloc[0].to_dict())
