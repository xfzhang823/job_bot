"""db_io/db_inserters.py"""

from __future__ import annotations

import datetime as dt
import logging
from functools import lru_cache
from typing import Dict, Literal, Optional, Any

import pandas as pd
import yaml

from job_bot.db_io.get_db_connection import get_db_connection
from job_bot.db_io.db_schema_registry import DUCKDB_SCHEMA_REGISTRY, TableName
from job_bot.db_io.db_utils import align_df_with_schema
from job_bot.config.project_config import DB_INSERTERS_YAML

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


# todo: need to remove url param as safety net - drive everything w/t dataframe
def insert_df_with_config(
    df: pd.DataFrame,
    table_name: str | TableName,
    *,
    url: Optional[str] = None,
    llm_provider: Optional[str] = None,
    model_id: Optional[str] = None,
    mode: Literal["append", "replace"] = "append",
    **stamps_kwargs: Any,  # NEW: extra stamped fields, e.g., resp_model_id=..., resp_llm_provider=...
) -> None:
    """
    Insert a DataFrame into DuckDB using YAML-driven insert policies.

    YAML semantics
    --------------
    • defaults.mode, defaults.dedup
    • defaults.stamp.* and tables.<name>.stamp.*
        - Values: now | default | param | none
        - Controls how each column is filled before insert
    • defaults.llm_defaults.{llm_provider, model_id}
        - Provides fallback values when stamp=default
    • defaults.inherit.iteration
        - Options: pipeline_control | param | constant | none
        - Controls how iteration is resolved
    • tables.<name>.inherit.iteration
        - Table-level override of iteration inheritance

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to insert.
    table_name : str | TableName
        Target table (enum or string).
    url : str, optional
        Canonical job URL, required if iteration comes from pipeline_control.
    llm_provider : str, optional
        Provider name (used if YAML stamp policy requests it).
    model_id : str, optional
        Model identifier (used if YAML stamp policy requests it).
    mode : {"append", "replace"}, default="append"
        Intent marker; actual dedup policy is determined by YAML.
    **stamps_kwargs : Any
        Extra per-row values for stamp:param fields (e.g., resp_llm_provider, resp_model_id).

    Raises
    ------
    ValueError
        If the table name is unknown or schema undefined.
    AssertionError
        If DataFrame columns mismatch schema.

    Notes
    -----
    • stamp:now → fill column with current UTC timestamp
    • stamp:default → fill from llm_defaults (llm_provider/model_id only today)
    • stamp:param → pull from kwargs (**stamps_kwargs) if provided
    • stamp:none → leave column unset
    • inherit.iteration:pipeline_control →
        SELECT MAX(iteration)
        FROM pipeline_control
        WHERE url=?
    • inherit.iteration:param → use explicit kwargs["iteration"]
    • inherit.iteration:constant → use defaults.constants.iteration
    • inherit.iteration:none → do not stamp iteration

    Behavior
    --------
    1. Normalize table name to a TableName enum.
    2. Fetch schema (columns + PKs) from DUCKDB_SCHEMA_REGISTRY.
    3. Load insert config from config/db_inserters.yaml.
    4. Stamp DataFrame columns according to YAML policies.
    5. Align columns strictly to schema order.
    6. Apply dedup policy:
       - pk_scoped → DELETE rows with matching PKs before insert
       - full_replace → DELETE all rows before insert
       - none → blind append
    7. Insert rows using explicit column lists, ensuring position-neutral
        behavior.

    Examples
    --------
    # Insert new job URLs (dedup by URL, no iteration or LLM fields stamped)
    df = pd.DataFrame([{"url": "https://example.com/job123"}])
    insert_df_with_config(df, TableName.JOB_URLS)

    # Insert extracted requirements (dedup by url+iteration, stamp iteration + LLM fields)
    reqs_df = pd.DataFrame([{"requirement_key": "req1", "requirement": "Python"}])
    insert_df_with_config(
        reqs_df,
        TableName.EXTRACTED_REQUIREMENTS,
        url="https://example.com/job123",
        llm_provider="OPENAI",
        model_id="gpt-4-1-nano",
    )

    # Insert similarity metrics (requires resp_* passed explicitly)
    metrics_df = pd.DataFrame([{"responsibility_key": "r1", "requirement_key": "q1"}])
    insert_df_with_config(
        metrics_df,
        TableName.SIMILARITY_METRICS,
        url="https://example.com/job123",
        resp_llm_provider="OPENAI",
        resp_model_id="gpt-4-1-nano",
    )
    """

    if df.empty:
        logger.info(f"⚠️ Skipped insert into '{table_name}' — empty DataFrame")
        return

    tbl = _normalize_table_name(table_name)
    schema = DUCKDB_SCHEMA_REGISTRY.get(tbl)
    if schema is None or not schema.column_order:
        raise ValueError(f"❌ No schema defined for table '{tbl.value}'")

    # Merge user-visible kwargs into a single map for stamping
    # Note: explicit params still win over llm_defaults
    provided_params: dict[str, Any] = {
        "url": url,
        "llm_provider": llm_provider,
        "model_id": model_id,
        **stamps_kwargs,  # resp_llm_provider, resp_model_id, etc.
    }

    # Stamp per YAML config (generic)
    df = _stamp_df_per_config(df, tbl, provided_params)

    # Align to schema (strict: reorders/keeps only known columns)
    df = align_df_with_schema(df, schema.column_order, strict=False)

    # Schema sanity (belt-and-suspenders)
    missing = [c for c in schema.column_order if c not in df.columns]
    extra = [c for c in df.columns if c not in schema.column_order]
    assert not missing and not extra, (
        f"❌ Column mismatch before insert into '{tbl.value}' —\n"
        f"Missing: {missing}\nExtra: {extra}"
    )

    # Dedup policy
    defaults, tcfg = _get_tbl_cfg(tbl.value)
    dedup_policy = (tcfg.get("dedup") or defaults.get("dedup") or "pk_scoped").lower()
    pk_cols = schema.primary_keys

    # Duplicate check in the incoming batch
    if pk_cols:
        dupes = df.duplicated(subset=pk_cols, keep=False)
        if dupes.any():
            logger.error(
                f"❌ Duplicate PKs in batch on {pk_cols}:\n{df.loc[dupes, pk_cols]}"
            )
            raise ValueError("Duplicate primary keys detected in incoming DataFrame.")

    con = get_db_connection()
    con.register("df", df)
    try:
        cols = list(df.columns)
        col_list = ", ".join(cols)
        sel_list = ", ".join([f"df.{c}" for c in cols])

        # Ensure table exists (no-op if already there)
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {tbl.value} AS "
            f"SELECT {col_list} FROM df WHERE 1=0"
        )

        # Dedup strategies
        if dedup_policy == "pk_scoped" and pk_cols:
            where = " AND ".join([f"t.{k} = df.{k}" for k in pk_cols])
            con.execute(f"DELETE FROM {tbl.value} t USING df WHERE {where}")
            logger.info(f"🧹 PK-scoped dedup on {pk_cols} for '{tbl.value}'")

        elif dedup_policy == "full_replace":
            con.execute(f"DELETE FROM {tbl.value}")
            logger.info(f"🧨 Full table replace for '{tbl.value}'")

        elif dedup_policy == "none":
            logger.info(f"➕ No dedup policy for '{tbl.value}' (blind append)")

        else:
            logger.warning(f"⚠️ Unknown dedup '{dedup_policy}', defaulting to pk_scoped")
            if pk_cols:
                where = " AND ".join([f"t.{k} = df.{k}" for k in pk_cols])
                con.execute(f"DELETE FROM {tbl.value} t USING df WHERE {where}")

        # Position-neutral insert
        con.execute(f"INSERT INTO {tbl.value} ({col_list}) SELECT {sel_list} FROM df")
        logger.info(
            f"✅ Inserted {len(df)} row(s) into '{tbl.value}' (mode='{mode}', dedup='{dedup_policy}')"
        )
    finally:
        try:
            con.unregister("df")
        except Exception:
            pass


# ----------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_db_inserters_config() -> Dict:
    cfg_path = DB_INSERTERS_YAML
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_tbl_cfg(tname: str) -> tuple[dict, dict]:
    cfg = _load_db_inserters_config()
    defaults = cfg.get("defaults", {}) or {}
    tables = cfg.get("tables", {}) or {}
    return defaults, (tables.get(tname, {}) or {})


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _ensure_schema_columns(df: pd.DataFrame, table: TableName) -> pd.DataFrame:
    schema_cols = DUCKDB_SCHEMA_REGISTRY[table].column_order
    out = df.copy()
    for col in schema_cols:
        if col not in out.columns:
            out[col] = pd.NA
    # 👇 drop any accidental extra cols here if you want
    return out[schema_cols] if set(out.columns) - set(schema_cols) else out


def _normalize_table_name(table_name: str | TableName) -> TableName:
    if isinstance(table_name, TableName):
        return table_name
    for t in TableName:
        if t.value == table_name:
            return t
    raise ValueError(f"❌ Unknown table name: {table_name!r}")


def _now_naive_utc() -> dt.datetime:
    return dt.datetime.now().replace(tzinfo=None)


def _merge(*dicts):
    out = {}
    for d in dicts:
        if d:
            out.update(d)
    return out


def _max_iteration_from_pipeline_control(url: Optional[str]) -> Optional[int]:
    if not url:
        return None
    con = get_db_connection()
    row = con.execute(
        "SELECT MAX(iteration) FROM pipeline_control WHERE url = ?", [url]
    ).fetchone()
    return None if not row or row[0] is None else int(row[0])


def _resolve_iteration_from_cfg(
    *,
    url: Optional[str],
    inherit_mode: Optional[str],
    explicit_param: Optional[int],
    constant_map: dict,
) -> Optional[int]:
    """
    iteration strategy:
      - 'pipeline_control'  -> use MAX(iteration) for this URL
      - 'param'             -> use explicit_param (if provided)
      - 'constant'          -> use constants['iteration'] (if provided)
      - 'none' or missing   -> do not stamp iteration
    """
    mode = (inherit_mode or "none").lower()
    if mode == "param":
        return int(explicit_param) if explicit_param is not None else 0
    if mode == "constant":
        return int(constant_map.get("iteration", 0))
    if mode == "pipeline_control":
        it = _max_iteration_from_pipeline_control(url)
        return 0 if it is None else it
    # none / unknown
    return 0


def _stamp_df_per_config(
    df: pd.DataFrame,
    table: TableName,
    provided_params: dict[str, Any],
) -> pd.DataFrame:
    """
    Generic stamping per YAML:
      • timestamps: now
      • iteration: inherit strategy
      • any other field: param | default | none
      • LLM defaults come from defaults.llm_defaults
    """
    defaults, tcfg = _get_tbl_cfg(table.value)
    stamp_cfg = _merge(defaults.get("stamp"), tcfg.get("stamp"))
    inherit_cfg = _merge(defaults.get("inherit"), tcfg.get("inherit"))
    constants = _merge(defaults.get("constants"), tcfg.get("constants"))
    llm_defaults = defaults.get("llm_defaults", {}) or {}
    url = provided_params.get("url")

    # Ensure schema cols exist (should use pd.NA inside)
    out_df = _ensure_schema_columns(df, table)

    # (optional strictness) Only stamp schema columns
    schema_cols = set(DUCKDB_SCHEMA_REGISTRY[table].column_order)

    # URL convenience (if part of PK)
    pk = DUCKDB_SCHEMA_REGISTRY[table].primary_keys
    if "url" in pk and ("url" in out_df.columns) and (url is not None):
        # only fill nulls to avoid overwriting existing values
        out_df["url"] = out_df["url"].fillna(url)

    # Iteration inheritance
    inherit_iter_mode = inherit_cfg.get("iteration")
    stamp_iter_mode = (stamp_cfg or {}).get("iteration")
    effective_iter_mode = (
        inherit_iter_mode if stamp_iter_mode is None else stamp_iter_mode
    )
    if ("iteration" in out_df.columns) and (effective_iter_mode not in (None, "none")):
        out_df["iteration"] = _resolve_iteration_from_cfg(
            url=url,
            inherit_mode=effective_iter_mode,
            explicit_param=provided_params.get("iteration"),
            constant_map=constants,
        )

    # One timestamp per batch
    now_ts = _now_naive_utc()

    for col, mode in (stamp_cfg or {}).items():
        if col not in schema_cols:
            continue  # optional: ignore non-schema stamp targets

        if mode == "now":
            # create if missing (shouldn't happen post _ensure_schema_columns)
            if col not in out_df.columns:
                out_df[col] = now_ts
            else:
                out_df[col] = out_df[col].fillna(now_ts)

        elif mode == "default":
            if col in ("llm_provider", "model_id"):
                val = provided_params.get(col) or llm_defaults.get(col)
                if val is not None:
                    out_df[col] = out_df[col].fillna(val)

        elif mode == "param":
            val = provided_params.get(col)
            if val is not None:
                out_df[col] = out_df[col].fillna(val)

        elif mode == "none":
            pass
        else:
            pass

    return out_df
