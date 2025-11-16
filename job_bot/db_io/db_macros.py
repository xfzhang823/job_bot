"""
duckdb_macros.py
================
Centralized definitions + loader for DuckDB macros used across the job_bot pipeline.

This module ensures that all macros (TIMESTAMPTZ handling, ordering, date parsing, etc.)
are created in the current DuckDB connection automatically.
"""

from job_bot.db_io.get_db_connection import get_db_connection

# ---------------------------------------------------------------------
# 1. All macro definitions live here
# ---------------------------------------------------------------------
DUCKDB_MACROS_SQL = """
-- ===============================================================
--  TIMESTAMPTZ-safe timestamp helpers
-- ===============================================================

CREATE OR REPLACE MACRO ts_param(x) AS
  COALESCE(try_cast(x AS TIMESTAMPTZ), CURRENT_TIMESTAMP);

CREATE OR REPLACE MACRO ts_order(u, c) AS
  COALESCE(
    try_cast(u AS TIMESTAMPTZ),
    try_cast(c AS TIMESTAMPTZ),
    CURRENT_TIMESTAMP
  );

CREATE OR REPLACE MACRO ts_lt_now(x) AS
  (try_cast(x AS TIMESTAMPTZ) IS NULL
   OR try_cast(x AS TIMESTAMPTZ) < CURRENT_TIMESTAMP);

-- ===============================================================
--  Job-specific date parsing
-- ===============================================================

CREATE OR REPLACE MACRO parse_job_date(d) AS
  COALESCE(
    try_cast(d AS DATE),
    CAST(strptime(CAST(d AS VARCHAR), '%Y-%m-%d') AS DATE),
    CAST(strptime(CAST(d AS VARCHAR), '%b %d, %Y') AS DATE),
    CAST(strptime(CAST(d AS VARCHAR), '%B %d, %Y') AS DATE)
  );
"""

# ---------------------------------------------------------------------
# 2. Loader function: run this once at startup
# ---------------------------------------------------------------------
_MACROS_INSTALLED = False


def ensure_duckdb_macros() -> None:
    """
    Idempotently load all DuckDB macros into the active connection.

    Safe to call multiple times per process — macros will be replaced
    only if changed (DuckDB's CREATE OR REPLACE is idempotent).
    """
    global _MACROS_INSTALLED
    if _MACROS_INSTALLED:
        return

    con = get_db_connection()
    con.execute(DUCKDB_MACROS_SQL)

    _MACROS_INSTALLED = True
