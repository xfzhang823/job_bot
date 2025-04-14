"""
duckdb_adapter.py

* A DuckDB adapter is a focused module that wraps all interactions with DuckDB into
* a clean interface.

Abstract common read/write operations (e.g. insert_df, query_table)
Optionally handle schema management (e.g. creating tables, migrations)
Enforce consistent conventions (e.g. always store iteration, job_id, etc.)

📦 In practice, it does things like:
- Manage the connection lifecycle (connect(), close())
- Abstract common read/write operations (e.g. insert_df, query_table)
- Optionally handle schema management (e.g. creating tables, migrations)
- Enforce consistent conventions (e.g. always store iteration, job_id, etc.)



"""

# Internal or 3rd party dependencies
from pathlib import Path
import duckdb
import pandas as pd
from typing import Optional

# User defined
from utils.find_project_root import find_project_root


def get_duckdb_connection(
    db_path: Optional[Path | str] = None,
) -> duckdb.DuckDBPyConnection:
    """
    Create a connection to DuckDB.

    Args:
    db_path: Optional path to the DuckDB database. If not provided, a default path will be used.

    Returns:
    A DuckDBPyConnection object.
    """
    root_dir = find_project_root()
    if root_dir is None:
        raise FileNotFoundError("Could not locate the project root")

    if db_path is None:
        db_path = root_dir / "pipeline_data/db/pipeline_data.duckdb"
    elif isinstance(db_path, str):
        db_path = Path(db_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    return duckdb.connect(str(db_path))


def insert_df_to_table(df: pd.DataFrame, table: str):
    con = get_duckdb_connection()
    con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM df LIMIT 0")
    con.execute(f"INSERT INTO {table} SELECT * FROM df")


def query_df(sql: str) -> pd.DataFrame:
    con = get_duckdb_connection()
    return con.execute(sql).df()


def insert_json_to_table(json_glob_path: Path, table_name: str):
    con = get_duckdb_connection()

    # Let DuckDB load and infer schema from one or more files
    # read_json_auto() is a DuckDB func:
    # Reads JSON files from disk (including multiple via glob pattern),
    # Infers the schema automatically Supports nested fields and
    # flattens them when possible
    df = con.execute(
        f"""
        SELECT * FROM read_json_auto('{json_glob_path.as_posix()}')
    """
    ).df()

    # Create the table if it doesn't exist
    con.execute(
        f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df LIMIT 0"
    )  # LIMIT 0 makes sure no data is inserted during table creation—just the schema

    # Insert data
    con.register("df", df)
    con.execute(f"INSERT INTO {table_name} SELECT * FROM df")

    print(f"✅ Inserted {len(df)} rows into '{table_name}' from {json_glob_path}")
