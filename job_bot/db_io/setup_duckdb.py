"""
create_db.py

Creates all DuckDB tables as defined in DUCKDB_SCHEMAS.
This module should be run once at setup time or as part of a migration process.
"""

import logging
from db_io.duckdb_adapter import get_duckdb_connection
from db_io.db_schema_registry import DUCKDB_SCHEMA_REGISTRY
from db_io.pipeline_enums import TableName

logger = logging.getLogger(__name__)


def create_all_duckdb_tables():
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_all_duckdb_tables()
