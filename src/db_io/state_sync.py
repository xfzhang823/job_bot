# db_io/state_sync.py

from pydantic import BaseModel, HttpUrl
from typing import Optional
import pandas as pd
from datetime import datetime
from db_io.duckdb_adapter import get_duckdb_connection
from src.models.duckdb_table_models import PipelineState


def save_pipeline_state_to_duckdb(
    state: PipelineState, table_name: str = "pipeline_control"
):
    """
    Insert or update a single PipelineState into DuckDB.
    """
    df = pd.DataFrame([state.model_dump()])
    con = get_duckdb_connection()

    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} AS
        SELECT * FROM df LIMIT 0
    """
    )
    con.register("df", df)
    con.execute(f"DELETE FROM {table_name} WHERE url = '{state.url}'")
    con.execute(f"INSERT INTO {table_name} SELECT * FROM df")
    print(f"✅ Saved state for {state.url} to '{table_name}'")


def load_pipeline_state(
    url: str, table_name: str = "pipeline_control"
) -> Optional[PipelineState]:
    """
    Load a PipelineState from DuckDB by URL.
    """
    con = get_duckdb_connection()
    df = con.execute(f"SELECT * FROM {table_name} WHERE url = ?", (url,)).df()
    if df.empty:
        return None
    return PipelineState(**df.iloc[0].to_dict())
