"""
db_io/create_db_tables.py

Create DuckDB tables from DUCKDB_SCHEMA_REGISTRY using Pydantic models.
"""

from __future__ import annotations

import logging
from typing import Optional

from job_bot.db_io.get_db_connection import get_db_connection
from job_bot.db_io.db_schema_registry import DUCKDB_SCHEMA_REGISTRY
from job_bot.db_io.pipeline_enums import TableName
from job_bot.db_io.db_utils import generate_table_schema_from_model

logger = logging.getLogger(__name__)


def _to_tablename(name: TableName | str) -> Optional[TableName]:
    if isinstance(name, TableName):
        return name
    try:
        return TableName(name)
    except ValueError:
        for t in TableName:
            if t.value == name:
                return t
        return None


def create_single_db_table(table_name: TableName | str) -> bool:
    """
    Create a single DuckDB table defined in DUCKDB_SCHEMA_REGISTRY.
    Returns True if successful, False otherwise.
    """
    table = _to_tablename(table_name)
    if table is None:
        logger.error("❌ %r is not a valid TableName.", table_name)
        return False

    schema = DUCKDB_SCHEMA_REGISTRY.get(table)
    if not schema:
        logger.error("⚠️ Table '%s' not found in registry.", table.value)
        return False

    ddl = generate_table_schema_from_model(
        model=schema.model,
        table_name=schema.table_name,
        primary_keys=schema.primary_keys,
    )

    con = get_db_connection()
    try:
        con.execute(ddl)
        logger.info("✅ Table '%s' created or confirmed.", table.value)
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("❌ Error creating table '%s': %s", table.value, e, exc_info=True)
        return False
    finally:
        try:
            con.close()
        except Exception:
            pass


def create_all_db_tables() -> None:
    """
    Create all DuckDB tables listed in DUCKDB_SCHEMA_REGISTRY.
    """
    con = get_db_connection()
    try:
        for table, schema in DUCKDB_SCHEMA_REGISTRY.items():
            ddl = generate_table_schema_from_model(
                model=schema.model,
                table_name=schema.table_name,
                primary_keys=schema.primary_keys,
            )
            try:
                con.execute(ddl)
                logger.info("✅ Created or confirmed table '%s'.", table.value)
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "❌ Failed to create table '%s': %s", table.value, e, exc_info=True
                )
    finally:
        try:
            con.close()
        except Exception:
            pass
