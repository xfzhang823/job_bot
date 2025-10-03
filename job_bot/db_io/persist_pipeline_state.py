# job_bot/db_io/persist_pipeline_state.py
from __future__ import annotations
import logging
from typing import Optional, cast

import pandas as pd

from job_bot.db_io.pipeline_enums import (
    TableName,
    PipelineStatus,
    PipelineProcessStatus,
    PipelineStage,
)
from job_bot.models.db_table_models import PipelineState
from job_bot.db_io.db_inserters import insert_df_with_config
from job_bot.db_io.get_db_connection import get_db_connection


logger = logging.getLogger(__name__)

URL_ONLY_STAGES: set[PipelineStage] = {PipelineStage.JOB_URLS}  # extend if needed


def _normalize_process_status_for_stage(
    state_model: PipelineState,
) -> PipelineProcessStatus:
    """
    Enforce: if stage is URL-only -> NEW; else -> RUNNING,
    unless already terminal (COMPLETED or SKIPPED).
    """
    # Safely typed reads
    status: Optional[PipelineStatus] = cast(
        Optional[PipelineStatus], getattr(state_model, "status", None)
    )
    pstat: Optional[PipelineProcessStatus] = cast(
        Optional[PipelineProcessStatus], getattr(state_model, "process_status", None)
    )
    stage: Optional[PipelineStage] = cast(
        Optional[PipelineStage], getattr(state_model, "stage", None)
    )

    # Terminal sets (no 'CANCELED' in your project)
    TERMINAL_PS: set[PipelineProcessStatus] = {PipelineProcessStatus.COMPLETED}
    if hasattr(PipelineProcessStatus, "SKIPPED"):
        TERMINAL_PS.add(
            getattr(PipelineProcessStatus, "SKIPPED")
        )  # still a PipelineProcessStatus

    # 1) If already terminal, keep it.
    if pstat in TERMINAL_PS:
        return pstat  # type: ignore[return-value]  # safe: membership guarantees non-None, correct enum

    # 2) Map terminal *status* to process_status terminal
    if status in {PipelineStatus.COMPLETED} | (
        {getattr(PipelineStatus, "SKIPPED")}
        if hasattr(PipelineStatus, "SKIPPED")
        else set()
    ):
        if hasattr(PipelineProcessStatus, "SKIPPED") and status == getattr(
            PipelineStatus, "SKIPPED", None
        ):
            return getattr(PipelineProcessStatus, "SKIPPED")  # exact enum
        return PipelineProcessStatus.COMPLETED

    # 3) Non-terminal: derive from stage
    if stage in URL_ONLY_STAGES:
        return PipelineProcessStatus.NEW
    return PipelineProcessStatus.RUNNING


def _recompute_decision_flag_row(
    url: str, table_name: TableName = TableName.PIPELINE_CONTROL
) -> int:
    """
    Status-only rule to avoid importing orchestrator:
      decision_flag = 0 if status IN ('completed','skipped') else 1
      transition_flag = 0
    """
    con = get_db_connection()
    res = con.execute(
        f"""
        UPDATE {table_name.value}
        SET decision_flag = CASE WHEN LOWER(status) IN ('completed','skipped') THEN 0 ELSE 1 END,
            transition_flag = 0,
            updated_at = now()
        WHERE url = ?
        """,
        (url,),
    )
    return getattr(res, "rowcount", 0)


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
    • Used sync_decision_flag_for_control_row to update go/no go (1/0) decision flag.

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
        # Ensure correct project status
        state_model.process_status = _normalize_process_status_for_stage(state_model)

        # 2) Dump the ENTIRE model in JSON mode so Enums → values, HttpUrl → str
        #    exclude_none=False ensures optional columns (e.g., decision_flag) are included
        payload = state_model.model_dump(mode="json", exclude_none=False)

        # 3) (belt-and-suspenders) Ensure url is a plain string
        if "url" in payload:
            payload["url"] = str(payload["url"])

        df = pd.DataFrame([payload])

        # 4) Insert with validating table schemas
        insert_df_with_config(
            df=df,
            table_name=table_name,
            url=str(
                state_model.url
            ),  # Provide url - let inserter resolve iteration inheritance
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
