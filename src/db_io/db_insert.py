import logging
import json
from typing import List, Optional
import pandas as pd
from pydantic import BaseModel, HttpUrl
from db_io.duckdb_adapter import get_duckdb_connection
from db_io.db_schema_registry import DUCKDB_SCHEMA_REGISTRY, TableName


logger = logging.getLogger(__name__)


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

    Returns:
        pd.DataFrame: A cleaned and schema-aligned DataFrame.
    """
    df = df.copy()

    original_len = len(df)
    df.drop_duplicates(inplace=True)
    deduped_len = len(df)
    if deduped_len < original_len:
        logger.info(f"Dropped {original_len - deduped_len} duplicate rows")

    # Add missing columns
    for col in schema_columns:
        if col not in df.columns:
            df[col] = None
            logger.debug(f"Added missing column: {col}")

    # Drop unexpected columns (optional safety)
    missing_in_df = [col for col in schema_columns if col not in df.columns]
    extra_in_df = [col for col in df.columns if col not in schema_columns]

    if strict:
        if missing_in_df:
            raise ValueError(f"Missing columns in DataFrame: {missing_in_df}")
        if extra_in_df:
            raise ValueError(f"Unexpected extra columns in DataFrame: {extra_in_df}")

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


def insert_df_dedup(
    df: pd.DataFrame,
    table_name: str | TableName,
    key_cols: Optional[List[str]] = None,
    dedup: bool = True,
) -> None:
    """
    Inserts a DataFrame into a DuckDB table with optional deduplication.

    This function:
    - Aligns the DataFrame to match the expected schema from DUCKDB_SCHEMA_REGISTRY
    - Optionally deduplicates based on one or more key columns
    - Registers the DataFrame in DuckDB and performs deletion + insert

    Args:
        - df (pd.DataFrame): The DataFrame to insert.
            table_name: str | TableName: Name of the DuckDB table to insert into.
        - key_cols (Optional[List[str]]): Column names to match for duplicate deletion.
                                        If not provided, a default composite key is used.
        - dedup (bool): If True, deletes rows that match key columns before inserting.

    Raises:
        ValueError: If no schema is found for the given table.
    """
    if df.empty:
        logger.info(f"⚠️ Skipped insert into '{table_name}' — DataFrame is empty")
        return

    con = get_duckdb_connection()
    con.register("df", df)

    # * Ensure table_name is a TableName enum instance (convert from str if needed)
    try:
        table_name = (
            next(t for t in TableName if t.value == table_name)
            if isinstance(table_name, str)
            else table_name
        )
    except StopIteration:
        raise ValueError(f"❌ Unknown table name: {table_name}")

    schema_columns = DUCKDB_SCHEMA_REGISTRY[TableName[table_name]].column_order
    if schema_columns is None:
        raise ValueError(f"No schema defined for table '{table_name}'")

    # Final validation and alignment
    df = align_df_with_schema(df, schema_columns, strict=True)
    con.register("df", df)  # ✅ after alignment

    # Set default keys to intersection of preferred and available columns
    if key_cols is None:
        preferred_keys = ["url", "responsibility_key", "requirement_key"]
        key_cols = [col for col in preferred_keys if col in df.columns]

    if dedup and key_cols:
        where_clause = " AND ".join([f"t.{col} = df.{col}" for col in key_cols])
        con.execute(
            f"""
            DELETE FROM {table_name} t
            USING df
            WHERE {where_clause}
            """
        )
        logger.info(f"🧹 Deduplicated on columns: {key_cols}")

    # 🔍 Schema mismatch guard before insert
    assert df.shape[1] == len(schema_columns), (
        f"❌ Column mismatch before insert into '{table_name}' — "
        f"{df.shape[1]} columns in DataFrame vs {len(schema_columns)} expected.\n"
        f"Missing: {[col for col in schema_columns if col not in df.columns]}\n"
        f"Extra:   {[col for col in df.columns if col not in schema_columns]}"
    )

    con.execute(f"INSERT INTO {table_name} SELECT * FROM df")
    logger.info(f"✅ Inserted {len(df)} rows into '{table_name}' (dedup={dedup})")
