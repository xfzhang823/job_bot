"""
db_io/state_sync.py

"""

# Standard Imports
from typing import Optional
import logging
import pandas as pd

# Project level
from db_io.duckdb_adapter import get_duckdb_connection
from db_io.pipeline_enums import (
    TableName,
    Version,
    LLMProvider,
    PipelineStage,
    PipelineStatus,
)
from models.duckdb_table_models import PipelineState


logger = logging.getLogger(__name__)


def load_pipeline_state(
    url: str, table_name: TableName = TableName.PIPELINE_CONTROL
) -> Optional[PipelineState]:
    """
    Load a PipelineState from DuckDB by URL.
    """
    con = get_duckdb_connection()
    df = con.execute(f"SELECT * FROM {table_name.value} WHERE url = ?", (url,)).df()
    con.close()  # ✅ close after you're done reading from the DB

    if df.empty:
        return None
    return PipelineState(**df.iloc[0].to_dict())


def upsert_pipeline_state_to_duckdb(
    state: PipelineState,
    table_name: TableName = TableName.PIPELINE_CONTROL,
):
    """
    Upsert a `PipelineState` record into the DuckDB `pipeline_control` table.

    This function synchronizes the state of a specific job posting URL with the
    centralized `pipeline_control` table in DuckDB. It ensures that the record
    is either inserted as new or updated if an existing entry matches the
    primary key fields (url, iteration, version, llm_provider).

    Args:
        state (PipelineState): The pipeline state to persist.
        table_name (TableName): The DuckDB table to insert into.

    Key Operations:
    - Converts enums (`version`, `llm_provider`, `status`) to strings for compatibility
      with DuckDB.
    - Formats the `timestamp` field to the required `YYYY-MM-DD HH:MM:SS` format.
    - Reorders DataFrame columns to match the table schema in DuckDB, preventing
      field misalignment during insertion.
    - Executes a safe "upsert" operation: deletes the existing record with matching
      primary key fields and inserts the new record.

    Example Usage:
        >>> state = PipelineState(
                url="https://example.com/job/12345",
                iteration=0,
                version=Version.ORIGINAL,
                llm_provider=None,
                timestamp=datetime.now(),
                status=PipelineStatus.NEW,
            )
        >>> upsert_pipeline_state_to_duckdb(state)
    """
    logger.info(f"Upserting data to '{table_name.value}' table in DuckDB...")

    try:
        # Convert to a dictionary
        data = state.model_dump()

        # Remove 'stage' field completely
        data.pop("stage", None)

        # Handle llm_provider explicitly
        data["llm_provider"] = (
            data["llm_provider"].value if data["llm_provider"] else "none"
        )

        # Convert enums to strings
        data["version"] = (
            data["version"].value
            if isinstance(data["version"], Version)
            else data["version"]
        )
        data["status"] = (
            data["status"].value
            if isinstance(data["status"], PipelineStatus)
            else data["status"]
        )

        # Format timestamp
        if isinstance(data["timestamp"], pd.Timestamp):
            data["timestamp"] = data["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

        # Create DataFrame
        df = pd.DataFrame([data])

        # Debug output
        logger.debug(f"Processed data before insertion: {data}")
        logger.debug(f"DataFrame columns before reordering: {list(df.columns)}")

        # Connect to DuckDB
        con = get_duckdb_connection()

        # Fetch column order from DuckDB
        column_order = [
            row[1]
            for row in con.execute(
                f"PRAGMA table_info('{table_name.value}')"
            ).fetchall()
        ]
        logger.debug(f"Expected column order: {column_order}")

        # Ensure all expected columns exist in DataFrame
        for col in column_order:
            if col not in df.columns:
                df[col] = None  # Set missing columns to None

        # Reorder DataFrame columns
        df = df[column_order]

        # Debug: Verify DataFrame structure before insertion
        logger.debug(f"DataFrame columns after reordering: {list(df.columns)}")
        logger.debug(f"DataFrame head before insertion:\n{df.head()}")

        # Register DataFrame
        con.register("df", df)

        # Safe upsert: delete then insert
        con.execute(
            f"""
            DELETE FROM {table_name.value}
            WHERE url = ? AND iteration = ? AND version = ? AND llm_provider = ?
            """,
            (
                data["url"],
                data["iteration"],
                data["version"],
                data["llm_provider"],
            ),
        )

        # Insert the record
        con.execute(f"INSERT INTO {table_name.value} SELECT * FROM df")

        logger.info(f"✅ Saved state for {data['url']} to '{table_name.value}'")

    except Exception as e:
        logger.exception(f"❌ Failed to upsert pipeline state for {data['url']}: {e}")

    finally:
        if "con" in locals():
            con.close()
