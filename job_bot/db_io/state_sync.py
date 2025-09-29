"""
db_io/state_sync.py

Control-plane helpers that delegate to the unified loaders/inserters.

- load_pipeline_state(url): returns the latest PipelineState for a URL
  using db_loaders.load_table with table-level order_by config.

- update_and_persist_pipeline_state(state): YAML-driven insert via
  db_inserters.insert_df_with_config (dedup + stamps handled centrally).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from job_bot.db_io.pipeline_enums import TableName
from job_bot.models.db_table_models import PipelineState
from job_bot.db_io.db_loaders import load_table
from job_bot.db_io.db_inserters import insert_df_with_config

logger = logging.getLogger(__name__)


def _enum_val(x: Any) -> Any:
    """Return Enum.value for Enum instances; identity otherwise."""
    try:
        from enum import Enum

        if isinstance(x, Enum):
            return x.value
    except Exception:
        pass
    return x


def load_pipeline_state(
    url: str,
    table_name: TableName = TableName.PIPELINE_CONTROL,
) -> Optional[PipelineState]:
    """
    Load the current (latest) PipelineState for a given URL.

    This leverages `db_loaders.load_table`, which applies the loader YAML:
      • filters:   (url, iteration, etc.)
      • order_by:  table/default order (e.g., updated_at DESC, created_at DESC)
      • rehydrate: returns a PipelineState model when configured

    Args:
        url: The job URL to fetch.
        table_name: Target table (default: PIPELINE_CONTROL).

    Returns:
        PipelineState if found; otherwise None.
    """
    try:
        model = load_table(table_name, url=url)
        # When a single URL is requested and a rehydrator exists, load_table
        # returns the typed model (not a DataFrame).
        if isinstance(model, PipelineState):
            return model
        # If your loader for pipeline_control isn’t modeled yet, `model` could be a DF.
        if isinstance(model, pd.DataFrame) and not model.empty:
            # Take first row (loader should already order newest first)
            return PipelineState(**model.iloc[0].to_dict())
        return None
    except Exception as e:
        logger.error(
            "❌ Failed to load pipeline state for %s: %s", url, e, exc_info=True
        )
        return None


def update_and_persist_pipeline_state(
    state_model: PipelineState,
    table_name: TableName = TableName.PIPELINE_CONTROL,
) -> None:
    """
    Update and persist a PipelineState by delegating to `insert_df_with_config`.

    FSM semantics
    -------------
    • Called whenever a PipelineState is initialized, advanced, retried, or marked.
    • Guarantees a single authoritative row per (url, iteration) in `pipeline_control`.

    DB semantics
    ------------
    • Implemented as an upsert (update-or-insert) via `insert_df_with_config`.
    • Central inserter applies:
        1. Deduplication (e.g., pk_scoped → delete existing PK rows before insert).
        2. Stamping (created_at/updated_at, iteration, etc., per YAML config).
        3. Schema alignment (column order, defaults).

    Notes
    -----
    • The state is converted to a one-row DataFrame; enums are flattened to `.value`.
    • If you need to carry an existing `created_at`, include it explicitly;
        the inserter’s stamping rules will set it on first insert.

    Args
    ----
    state_model : PipelineState
        The validated PipelineState to persist.
    table_name : TableName, default = PIPELINE_CONTROL
        Target DuckDB table.
    """

    try:
        row = {
            "url": state_model.url,
            "iteration": state_model.iteration,
            "stage": _enum_val(state_model.stage),
            "status": _enum_val(state_model.status),
            "version": _enum_val(getattr(state_model, "version", None)),
            "source_file": getattr(state_model, "source_file", None),
            "notes": getattr(state_model, "notes", None),
            # If the schema includes created_at/updated_at and you want to
            # explicitly carry them through, you can add them here.
            # Otherwise, let YAML config handle them:
            # "created_at": getattr(state_model, "created_at", None),
            # "updated_at": getattr(state_model, "updated_at", None),
        }
        df = pd.DataFrame([row])

        insert_df_with_config(
            df,
            table_name,
            # Provide url so inserter can resolve iteration inheritance
            url=str(state_model.url),
            # Any stamp:param fields can be supplied via kwargs here if needed:
            # iteration=state_model.iteration, resp_llm_provider=..., resp_model_id=...
        )
        logger.info("✅ Upserted pipeline state for %s", state_model.url)
    except Exception as e:
        logger.error(
            "❌ Failed to upsert pipeline state for %s: %s",
            getattr(state_model, "url", "<unknown>"),
            e,
            exc_info=True,
        )
        raise
