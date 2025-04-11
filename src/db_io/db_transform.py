"""
transform.py

This module defines a generic model-to-table transformation interface for DuckDB ingestion.

It provides:
- A dispatch dictionary (`FLATTEN_DISPATCH`) that maps logical DuckDB table names to
  their corresponding Pydantic model types and flattening functions.
- A single entrypoint function `flatten_model_to_df()` that validates model compatibility,
  flattens structured data into tabular format, and appends standardized metadata fields
  (`source_file`, `stage`, `timestamp`).

This module supports the ingestion of multiple structured content types, including:
- Job postings
- Extracted and flattened job requirements
- Responsibilities (nested and flattened)
- Similarity metrics
- Other table schemas defined in DUCKDB_SCHEMAS

Usage:
    df = flatten_model_to_df(model, table_name, source_file, stage)

Raises:
    ValueError: If an unsupported table name is provided.
    TypeError: If the model does not match the expected type for the table.

This utility is central to ensuring schema-aligned ingestion and downstream
consistency across the DuckDB database.
"""

from pathlib import Path
import logging
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Literal, Union, Callable, Tuple, Type, TypeVar
from db_io.schema_definitions import TableName, PipelineStage, DUCKDB_COLUMN_ORDER
from db_io.flatten_and_rehydrate import (
    flatten_job_postings_to_table,
    flatten_requirements_to_table,
    flatten_responsibilities_to_table,
    flatten_nested_responsibilities_to_table,
    flatten_extracted_requirements_to_table,
    flatten_job_urls_to_table,
)
from models.resume_job_description_io_models import (
    NestedResponsibilities,
    Requirements,
    Responsibilities,
    JobPostingsFile,
    JobPostingUrlsFile,
    ExtractedRequirementsFile,
    SimilarityMetrics,
)
from models.llm_response_models import (
    JobSiteResponse,
    JobSiteData,
    RequirementsResponse,
    NestedRequirements,
)

logger = logging.getLogger(__name__)

ModelType = Union[
    RequirementsResponse,
    Requirements,
    Responsibilities,
    NestedResponsibilities,
    JobPostingUrlsFile,
    JobPostingsFile,
    ExtractedRequirementsFile,
    SimilarityMetrics,
]

# Define type of flatten functions
T = TypeVar("T", bound=BaseModel)  # TypeVar to declare type ()

FlattenFuncTyped = Callable[[T], pd.DataFrame]  # input type is T & output is dataframe
FlattenDispatch = Dict[TableName, Tuple[Type[T], FlattenFuncTyped[T]]]

FLATTEN_DISPATCH: FlattenDispatch = {
    TableName.JOB_URLS: (JobPostingUrlsFile, flatten_job_urls_to_table),
    TableName.JOB_POSTINGS: (JobPostingsFile, flatten_job_postings_to_table),
    TableName.EXTRACTED_REQUIREMENTS: (
        ExtractedRequirementsFile,
        flatten_extracted_requirements_to_table,
    ),
    TableName.FLATTENED_REQUIREMENTS: (Requirements, flatten_requirements_to_table),
    TableName.FLATTENED_RESPONSIBILITIES: (
        Responsibilities,
        flatten_responsibilities_to_table,
    ),
    TableName.PRUNED_RESPONSIBILITIES: (
        NestedResponsibilities,
        flatten_nested_responsibilities_to_table,
    ),
    TableName.EDITED_RESPONSIBILITIES: (
        NestedResponsibilities,
        flatten_nested_responsibilities_to_table,
    ),
}


def add_ingestion_metadata(
    df: pd.DataFrame,
    file_path: Path,
    stage: PipelineStage,
    table: TableName,
    version: str | None = None,
    llm_provider: str | None = None,
) -> pd.DataFrame:
    """
    Conditionally adds standard metadata fields to a DataFrame before DuckDB insertion.

    Only adds fields that are expected by the DuckDB table schema.

    Args:
        df (pd.DataFrame): The target DataFrame
        file_path (Path): Source file path to populate 'source_file'
        stage (PipelineStage): Processing stage (e.g., PREPROCESSING)
        table (TableName): Enum name of the target table
        version (str, optional): Optional version tag (e.g., 'original', 'edited')
        llm_provider (str, optional): Optional LLM provider string (e.g., 'openai')

    Returns:
        pd.DataFrame: Modified DataFrame with added metadata columns if applicable.
    """
    schema_cols = DUCKDB_COLUMN_ORDER[table.value]

    if "source_file" in schema_cols:
        df["source_file"] = str(file_path)
    if "stage" in schema_cols:
        df["stage"] = stage.value
    if "timestamp" in schema_cols:
        df["timestamp"] = pd.Timestamp.now()
    if "version" in schema_cols and version is not None:
        df["version"] = version
    if "llm_provider" in schema_cols:
        df["llm_provider"] = llm_provider

    return df


def flatten_model_to_df(
    model: ModelType,
    table_name: TableName,  # ✅ pass the enum directly
    source_file: Path | str,
    stage: PipelineStage,  # ✅ Accept the Enum directly
) -> pd.DataFrame:
    """
    Generic model-to-DataFrame dispatcher using table-name config.

    Validates model compatibility and routes to the correct flattener.
    Adds source/stage/timestamp metadata.

    Args:
        - model (BaseModel): A validated Pydantic model to flatten.
        - table_name (TableName): Enum identifying the destination DuckDB table.
        - source_file (str | Path): File path the model was loaded from
        (used for traceability).
        - stage (PipelineStage): Enum identifying the processing stage
        (e.g., PREPROCESSING).

    Returns:
        pd.DataFrame: A schema-aligned, metadata-enriched DataFrame ready for
        DB insertion.

    >>> Example:
        df = flatten_model_to_df(
            model=model,
            table_name=table_name,
            source_file=file_path,
            stage=PipelineStage.STAGING,
        )
    """
    logger.info(f"🪄 Flattened model '{type(model).__name__}' for table '{table_name}'")

    if table_name not in FLATTEN_DISPATCH:
        raise ValueError(f"Unsupported table: {table_name}")

    expected_model_type, flatten_func = FLATTEN_DISPATCH[table_name]

    if not isinstance(model, expected_model_type):
        raise TypeError(
            f"Expected {expected_model_type.__name__} for table '{table_name}', "
            f"but got {type(model).__name__}"
        )

    df = flatten_func(model)

    # Ensure source file is a Path
    source_file = Path(source_file) if isinstance(source_file, str) else source_file

    # Add metadata columns
    df["source_file"] = str(source_file)
    df["stage"] = stage.value  # ✅ Convert Enum to string
    df["timestamp"] = pd.Timestamp.now()

    return df
