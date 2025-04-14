"""
transform.py

This module defines a generic model-to-table transformation interface for DuckDB ingestion.

It provides:
- A dispatch dictionary (`FLATTEN_DISPATCH`) that maps logical DuckDB table names to
  their corresponding Pydantic model types and flattening functions.
- A single entrypoint function `flatten_model_to_df()` that validates model compatibility,
  flattens structured data into tabular format, and appends standardized metadata fields
  (`source_file`, `stage`, `timestamp`, etc.).

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


def add_ingestion_metadata(
    df: pd.DataFrame,
    file_path: Optional[Path | str],
    stage: PipelineStage,
    table: TableName,
    version: Optional[str] = None,
    llm_provider: Optional[str] = None,
    iteration: int = 0,
) -> pd.DataFrame:
    """
    Adds standard metadata columns to a DataFrame if expected by the schema.

    Args:
        - df (pd.DataFrame): The DataFrame to modify.
        - file_path (Optional[Path | str]): File path used to populate 'source_file'
        column.
        - stage (PipelineStage): The pipeline stage this data came from.
        - table (TableName): Enum of the DuckDB table being written to.
        - version (Optional[str]): Optional version tag (e.g., 'original', 'edited').
        - llm_provider (Optional[str]): Optional LLM provider string (e.g., 'openai').
        - iteration (int): Number of times the job has been processed.

    Returns:
        pd.DataFrame: DataFrame with added metadata fields.
    """
    schema_cols = DUCKDB_COLUMN_ORDER[table.value]

    if "source_file" in schema_cols:
        df["source_file"] = str(file_path) if file_path else None
    if "stage" in schema_cols:
        df["stage"] = stage.value
    if "timestamp" in schema_cols:
        df["timestamp"] = pd.Timestamp.now()
    if "version" in schema_cols and version is not None:
        df["version"] = version
    if "llm_provider" in schema_cols:
        df["llm_provider"] = llm_provider
    if "iteration" in schema_cols:
        df["iteration"] = iteration

    return df


def flatten_model_to_df(
    model: ModelType,
    table_name: TableName,
    stage: PipelineStage,
    file_path: Optional[Path | str] = None,
    version: Optional[str] = None,
    llm_provider: Optional[str] = None,
    iteration: int = 0,
) -> pd.DataFrame:
    """
    Flattens a validated Pydantic model into a tabular DataFrame for DuckDB insertion.

    Args:
        model (ModelType): A validated Pydantic model to flatten.
        table_name (TableName): Enum identifying the DuckDB table to write to.
        stage (PipelineStage): The pipeline stage associated with this data.
        file_path (Optional[Path | str]): Optional source path (for traceability).
        version (Optional[str]): Optional version tag.
        llm_provider (Optional[str]): Optional LLM provider name.
        iteration (int): Processing iteration counter.

    Returns:
        pd.DataFrame: Flattened and metadata-enriched DataFrame.

        Example:
        >>> from db_io.db_transform import flatten_model_to_df
        >>> from models.resume_job_description_io_models import Responsibilities
        >>> from db_io.schema_definitions import TableName, PipelineStage
        >>> model = Responsibilities(url="https://job.com", responsibilities={"0": "Built APIs."})
        >>> df = flatten_model_to_df(
        ...     model=model,
        ...     table_name=TableName.FLATTENED_RESPONSIBILITIES,
        ...     stage=PipelineStage.STAGING,
        ...     file_path="input/responsibilities/job1.json",
        ...     version="original",
        ...     llm_provider="openai",
        ...     iteration=0,
        ... )
        >>> print(df.head())
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

    path = Path(file_path) if isinstance(file_path, str) and file_path else file_path
    return add_ingestion_metadata(
        df=df,
        file_path=path,
        stage=stage,
        table=table_name,
        version=version,
        llm_provider=llm_provider,
        iteration=iteration,
    )
