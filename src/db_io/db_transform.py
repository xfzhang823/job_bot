"""
transform.py

This module defines a generic model-to-table transformation interface for
inserting JSON objects into DuckDB.

It provides:
- A dispatch dictionary (`FLATTEN_DISPATCH`) that maps logical DuckDB table names
to their corresponding Pydantic model types and flattening functions.
- A single entrypoint function `flatten_model_to_df()` that validates model
compatibility, flattens structured data into tabular format, and appends
standardized metadata fields (`source_file`, `stage`, `timestamp`, etc.).

This module supports the ingestion of multiple structured content types, including:
- Job postings
- Extracted and flattened job requirements
- Responsibilities (nested and flattened)
- Similarity metrics
- Other table schemas defined in DUCKDB_SCHEMAS

Usage:
    df = flatten_model_to_df(model, table_name, stage=PipelineStage.STAGING)

Raises:
    ValueError: If an unsupported table name is provided.
    TypeError: If the model does not match the expected type for the table.

This utility is central to ensuring schema-aligned ingestion and downstream
consistency across the DuckDB database.
"""

from pathlib import Path
import logging
from typing import Dict, Optional, Union, Callable, Tuple, Type, TypeVar
from pydantic import BaseModel
import pandas as pd

# User defined
from db_io.db_schema_registry import DUCKDB_SCHEMA_REGISTRY
from db_io.pipeline_enums import (
    TableName,
    PipelineStage,
    LLMProvider,
    PipelineStatus,
    Version,
)
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
    JobPostingsBatch,
    JobPostingUrlsBatch,
    ExtractedRequirementsBatch,
    SimilarityMetrics,
)
from models.llm_response_models import RequirementsResponse

logger = logging.getLogger(__name__)

ModelType = Union[
    RequirementsResponse,
    Requirements,
    Responsibilities,
    NestedResponsibilities,
    JobPostingUrlsBatch,
    JobPostingsBatch,
    ExtractedRequirementsBatch,
    SimilarityMetrics,
]

T = TypeVar("T", bound=BaseModel)
FlattenFuncTyped = Callable[[T], pd.DataFrame]
FlattenDispatch = Dict[TableName, Tuple[Type[T], FlattenFuncTyped[T]]]

FLATTEN_DISPATCH: FlattenDispatch = {
    TableName.JOB_URLS: (JobPostingUrlsBatch, flatten_job_urls_to_table),
    TableName.JOB_POSTINGS: (JobPostingsBatch, flatten_job_postings_to_table),
    TableName.EXTRACTED_REQUIREMENTS: (
        ExtractedRequirementsBatch,
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


# Alternative and more flexible function to add metadata
def add_metadata(
    df: pd.DataFrame,
    file_path: Optional[Path | str],
    stage: PipelineStage,
    table: TableName,
    version: Optional[Version] = Version.ORIGINAL,
    llm_provider: Optional[LLMProvider] = LLMProvider.NONE,
    iteration: int = 0,
) -> pd.DataFrame:
    """
    Adds standard metadata columns to a DataFrame if expected by the schema.
    """
    version = version or Version.ORIGINAL
    llm_provider = llm_provider or LLMProvider.NONE

    schema_cols = DUCKDB_SCHEMA_REGISTRY[table].column_order

    if "source_file" in schema_cols:
        df["source_file"] = str(file_path) if file_path else None
    if "stage" in schema_cols:
        df["stage"] = stage.value
    if "timestamp" in schema_cols:
        df["timestamp"] = pd.Timestamp.now()
    if "version" in schema_cols:
        df["version"] = version.value
    if "llm_provider" in schema_cols:
        df["llm_provider"] = llm_provider.value
    if "iteration" in schema_cols:
        df["iteration"] = iteration

    return df


# * Function to add standard metadata to ALL TABLES
def add_all_metadata(
    df: pd.DataFrame,
    file_path: Optional[Path | str],
    stage: PipelineStage,
    table: TableName,
    version: Version | None = None,
    llm_provider: LLMProvider | None = None,
    iteration: int = 0,
) -> pd.DataFrame:
    """
    * Adds ALL STANDARD metadata columns, even if not in the target schema.
    Unused fields will be filled with None or defaults to ensure consistent
    downstream format.
    """
    version = version or Version.ORIGINAL
    llm_provider = llm_provider or LLMProvider.NONE

    df = df.copy()

    # todo: need to fix this
    # # Get metadata and full schema fields from registry
    # schema = DUCKDB_SCHEMA_REGISTRY[table]
    # metadata_fields = schema.metadata_fields
    # all_columns = schema.column_order

    df["source_file"] = str(file_path) if file_path else None
    df["stage"] = stage.value
    df["timestamp"] = pd.Timestamp.now()
    df["version"] = version.value
    df["llm_provider"] = llm_provider.value
    df["iteration"] = iteration

    # Add any missing columns (padding to align with table schema)
    schema_cols = DUCKDB_SCHEMA_REGISTRY[table].column_order
    for col in schema_cols:
        if col not in df.columns:
            df[col] = None

    return df[schema_cols]


def flatten_model_to_df(
    model: ModelType,
    table_name: TableName,
    stage: PipelineStage,
    source_file: Optional[Path | str] = None,
    version: Optional[Version] = None,
    llm_provider: Optional[LLMProvider] = None,
    iteration: int = 0,
) -> pd.DataFrame:
    """
    Flattens a validated Pydantic model into a tabular DataFrame for DuckDB insertion.
    """
    logger.info(
        f"🪄 Flattening model '{type(model).__name__}' for table '{table_name}'"
    )

    if table_name not in FLATTEN_DISPATCH:
        raise ValueError(f"Unsupported table: {table_name}")

    expected_model_type, flatten_func = FLATTEN_DISPATCH[table_name]
    if not isinstance(model, expected_model_type):
        raise TypeError(
            f"Expected model of type '{expected_model_type.__name__}' for table '{table_name}', "
            f"but got '{type(model).__name__}'"
        )

    df = flatten_func(model)

    path = (
        Path(source_file)
        if isinstance(source_file, str) and source_file
        else source_file
    )

    return add_all_metadata(
        df=df,
        file_path=path,
        stage=stage,
        table=table_name,
        version=version,
        llm_provider=llm_provider,
        iteration=iteration,
    )
