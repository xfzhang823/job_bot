"""
db_io/state_sync.py

"""

# Standard Imports
from typing import Optional
import logging
import pandas as pd

# Project level
from db_io.duckdb_adapter import get_duckdb_connection
from db_io.pipeline_enums import TableName
from models.duckdb_table_models import PipelineState


logger = logging.getLogger(__name__)


def persist_pipeline_state_to_duckdb(
    state: PipelineState,
    table_name: TableName = TableName.PIPELINE_CONTROL,
):
    """
    Persists a single `PipelineState` object into the DuckDB control table.

    This function:
    - Creates the control table if it doesn't already exist (based on schema).
    - Deletes any existing record for the same URL, iteration, version, and provider
    (ensuring upsert behavior).
    - Inserts the new state row with updated metadata.
    - Closes the connection after completion.

    Args:
        - state (PipelineState): The pipeline state to persist.
        - table_name (TableName): The DuckDB table to insert into. Defaults to PIPELINE_CONTROL.

    Logs:
        - ✅ Success on insert/update
        - ❌ Any exceptions during DB operations

    Raises:
        - Logs exception if insert/update fails. Does not re-raise by default.
    """
    try:
        df = pd.DataFrame([state.model_dump()])
        con = get_duckdb_connection()

        logger.debug(f"\U0001f4e5 Preparing to persist state for URL: {state.url}")
        logger.debug(f"\U0001f9be Data: {df.to_dict(orient='records')[0]}")

        # Ensure table exists
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name.value} AS
            SELECT * FROM df LIMIT 0
            """
        )
        con.register("df", df)

        # Safe upsert: delete then insert
        con.execute(
            f"""
            DELETE FROM {table_name.value}
            WHERE url = ? AND iteration = ? AND version = ? AND llm_provider = ?
            """,
            (
                state.url,
                state.iteration,
                state.version.value,
                state.llm_provider.value,
            ),
        )

        con.execute(f"INSERT INTO {table_name.value} SELECT * FROM df")

        logger.info(f"✅ Saved state for {state.url} to '{table_name.value}'")

    except Exception as e:
        logger.exception(f"❌ Failed to persist pipeline state for {state.url}: {e}")
        # raise  # Uncomment if you want upstream error handling

    finally:
        if "con" in locals():
            con.close()


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
