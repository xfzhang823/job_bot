"""
transform.py

This module contains tools such as add_metadata, helping transformating pyd models
to df, and vice versa.

This utility is central to ensuring schema-aligned ingestion and downstream
consistency across the DuckDB database.
"""

from pathlib import Path
import logging
import pandas as pd

# User defined
from job_bot.db_io.db_schema_registry import DUCKDB_SCHEMA_REGISTRY
from job_bot.db_io.pipeline_enums import (
    TableName,
    LLMProvider,
    Version,
)

logger = logging.getLogger(__name__)


# Alternative and more flexible function to add metadata
def add_metadata(
    df: pd.DataFrame,
    table: TableName,
    *,
    source_file: Path | str | None = None,
    iteration: int | None = None,
    version: Version | str | None = None,
    llm_provider: LLMProvider | str | None = None,
    model_id: str | None = None,
) -> pd.DataFrame:
    """
    Add table-aware metadata columns without overwriting existing values.

    Purpose
    -------
    Ensures each DataFrame has only the metadata fields owned by its target
    table, as defined in `DUCKDB_SCHEMA_REGISTRY[table].metadata_fields`.
    Prevents over-stamping (e.g., no global `stage`/`timestamp`) and avoids
    INSERT errors by aligning to schema intent.

    Behavior
    --------
    • Adds `source_file` if provided.
    • Adds `iteration` (defaults to 0 if table requires it).
    • Adds `version`, `llm_provider`, and `model_id` only if both provided
      *and* the table defines them.
    • Never overwrites existing DataFrame columns.
    • Ignores `stage`, `created_at`, `updated_at`, and metrics backends
      (set elsewhere).

    Usage
    -----
    Call this right before DB insertion, after flattening models:
      >>> add_metadata(df, TableName.FLATTENED_REQUIREMENTS, source_file="fsm")
      >>> add_metadata(df, TableName.EDITED_RESPONSIBILITIES,
      ...              source_file="fsm", llm_provider="openai", model_id="gpt-4.1-mini")

    Notes
    -----
    • `created_at` / `updated_at` come from DDL defaults or the write path.
    • Use FSM helpers (not this function) to stamp `stage` in `pipeline_control`.
    • Schema changes are safe: update the registry and this function adapts.
    """
    schema = DUCKDB_SCHEMA_REGISTRY[table]
    fields = set(getattr(schema, "metadata_fields", []))

    def set_if_needed(col: str, value):
        # only set if table wants it AND df doesn't already have it
        if col in fields and col not in df.columns:
            df[col] = value

    def as_value(x):
        return getattr(x, "value", x)

    # Optional stamps (table-aware)
    set_if_needed("source_file", str(source_file) if source_file is not None else None)

    # iteration: default to 0 if the table wants it and caller didn't provide one
    if "iteration" in fields and "iteration" not in df.columns:
        set_if_needed("iteration", 0 if iteration is None else iteration)

    # version + llm_provider only where present and provided
    if version is not None:
        set_if_needed("version", as_value(version))
    if llm_provider is not None:
        set_if_needed("llm_provider", as_value(llm_provider))

    # model_id (LLM artifacts)
    if model_id is not None:
        set_if_needed("model_id", model_id)

    # Explicitly DO NOT set:
    # - 'stage' (belongs only to pipeline_control and should be set by the FSM upsert)
    # - 'timestamp', 'created_at', 'updated_at' (handled by DDL/mixins/defaults)

    return df
