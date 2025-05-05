"""db_io/db_insert.py"""

import logging
import json
from typing import List, Optional, Literal
import pandas as pd
from pydantic import BaseModel, HttpUrl

# From project modules
from db_io.duckdb_adapter import get_duckdb_connection
from db_io.db_schema_registry import DUCKDB_SCHEMA_REGISTRY, TableName
from db_io.db_utils import align_df_with_schema

logger = logging.getLogger(__name__)


def insert_df_dedup(
    df: pd.DataFrame,
    table_name: str | TableName,
    key_cols: Optional[List[str]] = None,
    mode: Literal["append", "replace"] = "append",
) -> None:
    """
    Insert a DataFrame into a DuckDB table with automatic deduplication.

    This function always deduplicates based on `key_cols`, and supports two modes:
    - 'append': Remove existing duplicates and insert new data
    - 'replace': Same logic as 'append', but used to explicitly indicate replacement intent

    Args:
        df (pd.DataFrame): The data to insert.
        table_name (str | TableName): The DuckDB table to insert into.
        key_cols (Optional[List[str]]): Keys used to detect duplicates.
            Defaults to common keys.
        mode (Literal['append', 'replace']): Whether to append or replace matching records.

    Raises:
        ValueError: If the table name is invalid or schema is undefined.
        AssertionError: If the DataFrame column mismatch occurs before insert.
    """
    if df.empty:
        logger.info(f"⚠️ Skipped insert into '{table_name}' — DataFrame is empty")
        return

    # Ensure TableName enum type
    try:
        table_name = (
            next(t for t in TableName if t.value == table_name)
            if isinstance(table_name, str)
            else table_name
        )
    except StopIteration:
        raise ValueError(f"❌ Unknown table name: {table_name}")

    # Align columns to schema
    schema_columns = DUCKDB_SCHEMA_REGISTRY[table_name].column_order
    if not schema_columns:
        raise ValueError(f"❌ No schema defined for table '{table_name}'")

    df = align_df_with_schema(df, schema_columns, strict=True)

    # Connect + register DataFrame
    con = get_duckdb_connection()
    con.register("df", df)

    # Default deduplication keys
    if key_cols is None:
        key_cols = DUCKDB_SCHEMA_REGISTRY[table_name].primary_keys
        logger.info(f"Use default primary key(s) for {table_name.value}:\n{key_cols}")

    # Todo: debug; delete later
    dupes = df.duplicated(subset=key_cols, keep=False)
    if dupes.any():
        dup_df = df[dupes]
        logger.error(f"❌ Duplicate rows detected on key columns {key_cols}:\n{dup_df}")
        raise ValueError("Duplicate primary keys detected prior to insert.")

    # Always deduplicate (replace semantics optional)
    if key_cols:
        where_clause = " AND ".join([f"t.{col} = df.{col}" for col in key_cols])
        con.execute(
            f"""
            DELETE FROM {table_name} t
            USING df
            WHERE {where_clause}
            """
        )
        action = "🧹 Replaced" if mode == "replace" else "🧹 Deduplicated"
        logger.info(f"{action} on columns: {key_cols}")
    else:
        logger.warning(
            f"⚠️ No valid key columns found for deduplication — skipping delete"
        )

    # Final schema check
    assert df.shape[1] == len(schema_columns), (
        f"❌ Column mismatch before insert into '{table_name}' — "
        f"{df.shape[1]} columns in DataFrame vs {len(schema_columns)} expected.\n"
        f"Missing: {[col for col in schema_columns if col not in df.columns]}\n"
        f"Extra:   {[col for col in df.columns if col not in schema_columns]}"
    )

    con.execute(f"INSERT INTO {table_name} SELECT * FROM df")
    logger.info(f"✅ Inserted {len(df)} rows into '{table_name}' (mode='{mode}')")
