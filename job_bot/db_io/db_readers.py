"""db_io/db_readers.py"""

from __future__ import annotations
import pandas as pd

from job_bot.db_io.get_db_connection import get_db_connection
from job_bot.db_io.pipeline_enums import TableName
from job_bot.db_io.db_schema_registry import DUCKDB_SCHEMA_REGISTRY

RESP, RESP_KEY = "responsibility", "responsibility_key"
REQ, REQ_KEY = "requirement", "requirement_key"
LLM_PROVIDER, MODEL_ID = "llm_provider", "model_id"


def fetch_flattened_requirements(url: str) -> pd.DataFrame:
    """
    Fetch all flattened job requirements for a given URL.

    Queries the `flattened_requirements` table using the schema registry,
    projecting only the canonical columns: `url`, `requirement_key`, `requirement`.
    Results are ordered by `requirement_key`.

    Args:
        url (str): Job posting URL to filter on.

    Returns:
        pd.DataFrame: DataFrame containing requirements for the specified URL with
        columns: [url, requirement_key, requirement].
    """
    schema = DUCKDB_SCHEMA_REGISTRY[TableName.FLATTENED_REQUIREMENTS]
    sql = schema.select_by_url_sql(["url", REQ_KEY, REQ], order_by=REQ_KEY)
    con = get_db_connection()
    try:
        return con.execute(sql, (url,)).df()
    finally:
        con.close()


def fetch_flattened_responsibilities(url: str) -> pd.DataFrame:
    """
    Fetch all flattened responsibilities for a given URL.

    Queries the `flattened_responsibilities` table using the schema registry,
    projecting only the canonical columns: `url`, `responsibility_key`, `responsibility`.
    Results are ordered by `responsibility_key`.

    Args:
        url (str): Job posting URL to filter on.

    Returns:
        pd.DataFrame: DataFrame containing responsibilities for the specified URL
            with columns: [url, responsibility_key, responsibility].
    """
    schema = DUCKDB_SCHEMA_REGISTRY[TableName.FLATTENED_RESPONSIBILITIES]
    sql = schema.select_by_url_sql(["url", RESP_KEY, RESP], order_by=RESP_KEY)
    con = get_db_connection()
    try:
        return con.execute(sql, (url,)).df()
    finally:
        con.close()


def fetch_edited_responsibilities(url: str) -> pd.DataFrame:
    """
    Fetch all LLM-edited responsibilities for a given URL.

    Queries the `edited_responsibilities` table using the schema registry,
    projecting only the canonical columns: `url`, `responsibility_key`,
    `responsibility`, 'llm_provider', 'model_id'.
    Results are ordered by `responsibility_key`.

    Args:
        url (str): Job posting URL to filter on.

    Returns:
        pd.DataFrame: DataFrame containing edited responsibilities for
            the specified URL with columns: [url, responsibility_key, responsibility].
    """
    schema = DUCKDB_SCHEMA_REGISTRY[TableName.EDITED_RESPONSIBILITIES]
    sql = schema.select_by_url_sql(
        ["url", RESP_KEY, RESP, LLM_PROVIDER, MODEL_ID], order_by=RESP_KEY
    )
    con = get_db_connection()
    try:
        return con.execute(sql, (url,)).df()
    finally:
        con.close()
