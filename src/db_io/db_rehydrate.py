# db_io/db_rehydrate.py

from typing import Callable, TypeVar
import pandas as pd
from pydantic import BaseModel
from db_io.duckdb_adapter import get_duckdb_connection
from db_io.db_schema_registry import TableName, DUCKDB_SCHEMA_REGISTRY
from db_io.flatten_and_rehydrate import (
    rehydrate_job_urls_from_table,
    rehydrate_job_postings_from_table,
    rehydrate_extracted_requirements_from_table,
    rehydrate_requirements_from_table,
    rehydrate_responsibilities_from_table,
    rehydrate_nested_responsibilities_from_table,
)
from models.resume_job_description_io_models import (
    NestedResponsibilities,
    Requirements,
    Responsibilities,
    JobPostingsBatch,
    JobPostingUrlsBatch,
    ExtractedRequirementsBatch,
    SimilarityMetrics,
)
from models.llm_response_models import (
    JobSiteResponse,
    JobSiteData,
    RequirementsResponse,
    NestedRequirements,
)

# Dispatcher: maps table to appropriate rehydration function
T = TypeVar(
    "T", bound=BaseModel | JobPostingUrlsBatch | ExtractedRequirementsBatch
)  # TypeVar to declare type ()
RehydrateFunc = Callable[[pd.DataFrame], T]

REHYDRATE_DISPATCH: dict[TableName, RehydrateFunc] = {
    TableName.JOB_URLS: rehydrate_job_urls_from_table,
    TableName.JOB_POSTINGS: rehydrate_job_postings_from_table,
    TableName.EXTRACTED_REQUIREMENTS: rehydrate_extracted_requirements_from_table,
    TableName.FLATTENED_REQUIREMENTS: rehydrate_requirements_from_table,
    TableName.FLATTENED_RESPONSIBILITIES: rehydrate_responsibilities_from_table,
    TableName.PRUNED_RESPONSIBILITIES: rehydrate_nested_responsibilities_from_table,
    TableName.EDITED_RESPONSIBILITIES: rehydrate_nested_responsibilities_from_table,
}


def strip_ingestion_metadata(df: pd.DataFrame, table: TableName) -> pd.DataFrame:
    """
    Removes standardized ingestion metadata columns from a DataFrame
    before rehydrating into a Pydantic model.

    Args:
        - df (pd.DataFrame): The DataFrame loaded from DuckDB.
        - table (TableName): The DuckDB table being rehydrated.

    Returns:
        pd.DataFrame: Cleaned DataFrame with only schema-relevant fields.
    """
    schema_cols = DUCKDB_SCHEMA_REGISTRY[table].column_order

    # Define standard metadata fields (always candidates for removal)
    metadata_fields = {
        "source_file",
        "stage",
        "timestamp",
        "version",
        "llm_provider",
        "iteration",
    }

    # Keep only fields that are part of the model schema (not metadata)
    model_fields = [
        col for col in df.columns if col in schema_cols and col not in metadata_fields
    ]

    return df[model_fields].copy()


def rehydrate_model_from_duckdb(
    table: TableName,
    url: str,
    version: str | None = None,
    iteration: int | None = None,
) -> BaseModel:
    """
    Generic loader that reads from a DuckDB table and rehydrates
    a Pydantic model.

    Args:
        table (TableName): The DuckDB table to query.
        url (str): The job posting URL to match.
        version (str, optional): Optional version filter.
        iteration (int, optional): Optional iteration filter.

    Returns:
        BaseModel: The rehydrated Pydantic model.
    """
    con = get_duckdb_connection()

    # * Set single condition in the list
    # ("filter the data where the url column is equal to some value")
    filters = ["url = ?"]
    params = [url]

    # * Add additional filters (version, iteration)
    if version:
        filters.append("version = ?")
        params.append(version)
    if iteration is not None:
        filters.append("iteration = ?")
        params.append(iteration)  # pylint: disable=reportArgumentType

    sql = f"""
        SELECT * FROM {table.value}
        WHERE {' AND '.join(filters)}
        ORDER BY timestamp DESC
    """

    df = con.execute(sql, params).df()
    df = strip_ingestion_metadata(df, TableName.JOB_POSTINGS)

    if df.empty:
        raise ValueError(f"No records found in table '{table.value}' for url={url}")

    return REHYDRATE_DISPATCH[table](df)
