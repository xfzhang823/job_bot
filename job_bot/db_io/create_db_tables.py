"""
db_io/create_db.py

Creates all DuckDB tables as defined in DUCKDB_SCHEMAS.
This module should be run once at setup time or as part of a migration process.
"""

import logging
from db_io.duckdb_adapter import get_duckdb_connection
from db_io.db_schema_registry import DUCKDB_SCHEMA_REGISTRY
from db_io.pipeline_enums import TableName

logger = logging.getLogger(__name__)


def create_single_db_table(table_name: TableName | str) -> None:
    """
    Creates a single DuckDB table based on the schema defined in DUCKDB_SCHEMA_REGISTRY.

    This function accepts either a `TableName` enum instance or a string representation
    of a table name and attempts to create the corresponding DuckDB table. If the table
    already exists, it will not attempt to recreate it.

    Args:
        table_name (TableName | str): The enum value or string representation of the table.

    Example Usage:
        # Using a TableName enum instance:
        create_single_db_table(TableName.PIPELINE_CONTROL)

        # Using a string representation:
        create_single_db_table("pipeline_control")

    Raises:
        ValueError: If the provided table name is not a valid enum value.
        Exception: If the table creation query fails.
    """
    con = get_duckdb_connection()

    # If str, convert to enum
    try:
        table = TableName(table_name)
    except ValueError:
        logger.error(f"{table_name} is not a valid table name.")
        return

    schema = DUCKDB_SCHEMA_REGISTRY.get(table)

    if not schema:
        logger.error(f"⚠️ Table '{table.value}' not found in registry.")
        return

    try:
        con.execute(schema.ddl)
        logger.info(f"✅ Table '{table.value}' created successfully.")

    except Exception as e:
        logger.error(f"❌ Error creating table '{table.value}': {e}")


def create_all_db_tables():
    """
    Executes all `CREATE TABLE IF NOT EXISTS` statements from DUCKDB_SCHEMAS.

    This function connects to DuckDB and ensures that each table is created
    only if it doesn't already exist.

    Raises:
        Any exceptions encountered during DDL execution are caught and logged.
    """
    con = get_duckdb_connection()
    for table_name, table_schema in DUCKDB_SCHEMA_REGISTRY.items():
        try:
            con.execute(table_schema.ddl)
            logger.info(f"✅ Created or confirmed table '{table_name}' exists.")
        except Exception as e:
            logger.error(
                f"❌ Failed to create table '{table_name}': {e}", exc_info=True
            )
