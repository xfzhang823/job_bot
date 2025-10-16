"""
pipelines_with_fsm/state_orchestration_pipeline.py

Registry-driven control-plane sync for FSM pipelines.

This module keeps the `pipeline_control` table aligned with stage tables
(job_urls, job_postings, requirements, responsibilities, metrics). It does not
execute heavy pipeline work (scraping, LLM parsing, editing, evaluation); it
only manages orchestration state so stage pipelines know what to process next.

Core responsibilities:
- Initialize FSM state for new URLs (one row per url/iteration).
- Incrementally sync metadata (e.g., source_file, version) into
    pipeline_control.
- Preserve existing stage/status unless explicitly overridden.
- Run optional FSM integrity checks.

Typical usage:
    from job_bot.pipelines_with_fsm.state_orchestration_pipeline import (
        sync_all_tables_to_pipeline_control
    )
    sync_all_tables_to_pipeline_control(full=True)

Integration:
- Uses `DUCKDB_SCHEMA_REGISTRY` for schema/PK alignment.
- Interfaces with stage pipelines, FSM manager, and DuckDB for persistence.

This design ensures transparent, idempotent synchronization of control-plane
state, forming the backbone of FSM-driven orchestration.
"""

from __future__ import annotations
import logging
from typing import Optional, Literal, Iterable

from job_bot.db_io.get_db_connection import get_db_connection
from job_bot.db_io.create_db_tables import create_single_db_table
from job_bot.db_io.db_schema_registry import DUCKDB_SCHEMA_REGISTRY
from job_bot.db_io.pipeline_enums import (
    PipelineStage,
    PipelineStatus,
    PipelineProcessStatus,
    TableName,
)
from job_bot.fsm.check_pipeline_fsm import check_fsm_integrity
from job_bot.db_io.state_sync import (
    retry_error_one,
    advance_completed_one,
)
from db_io.db_utils import get_urls_from_pipeline_control


logger = logging.getLogger(__name__)


# -------------------------------
# Small helpers
# -------------------------------


def sql_str_lit(val: str) -> str:
    return "'" + val.replace("'", "''") + "'"


def coalesce_col(alias: str, fallback_sql: str, src_cols: set[str]) -> str:
    """Return COALESCE(j.alias, fallback_sql) if alias exists in source."""
    return f"COALESCE(j.{alias}, {fallback_sql})" if alias in src_cols else fallback_sql


def list_intersection(a: Iterable[str], b: Iterable[str]) -> list[str]:
    sb = set(b)
    return [x for x in a if x in sb]


# -------------------------------
# Sync to advance stage (ensure that existing pipeline control is not "stuck")
# -------------------------------


def fsm_retry_errors_all(
    *,
    dry_run: bool = False,
    table: TableName = TableName.PIPELINE_CONTROL,
) -> int:
    """
    Find URLs whose *status* == ERROR and call retry_error_one(url).
    Only touches `status`, never `process_status`.
    """
    urls = get_urls_from_pipeline_control(
        status=PipelineStatus.ERROR, active_urls_only=True
    )
    if dry_run:
        for u in urls:
            logger.info("DRY-RUN retry_error_one → %s", u)
        return len(urls)

    changed = 0
    for u in urls:
        try:
            if retry_error_one(u, table=table):
                changed += 1
        except Exception:
            logger.exception("retry_error_one failed for %s", u)
    logger.info("♻️ Auto-retry: %s / %s URLs reset to NEW", changed, len(urls))
    return changed


def fsm_auto_advance_completed_all(
    *,
    dry_run: bool = False,
    table: TableName = TableName.PIPELINE_CONTROL,
) -> int:
    """
    Find URLs whose *status* ==
    COMPLETED and advance one stage (status -> NEW)
    via advance_completed_one(url).

    Only touches `status`, never `process_status`.
    """
    urls = get_urls_from_pipeline_control(
        status=PipelineStatus.COMPLETED, active_urls_only=True
    )

    logger.debug(f"urls selected to update: {urls}")  # todo: debug; delete later
    if dry_run:
        for u in urls:
            logger.info("DRY-RUN advance_completed_one → %s", u)
        return len(urls)

    moved = 0
    for u in urls:
        try:
            if advance_completed_one(u, table=table):
                moved += 1
        except Exception:
            logger.exception("advance_completed_one failed for %s", u)
    logger.info(
        "➡️ Auto-advance: %s / %s URLs moved to next stage (status=NEW)",
        moved,
        len(urls),
    )
    return moved


# -------------------------------
# Generic control-plane sync
# -------------------------------


def sync_table_to_pipeline_control(
    *,
    source_table: TableName,
    stage: PipelineStage,
    insert_only: bool = True,
    refresh_meta_fields: tuple[str, ...] = ("source_file",),
    preserve_stage_on_update: bool = True,
    preserve_status_on_update: bool = True,
    set_status_on_insert: PipelineStatus = PipelineStatus.NEW,
    set_process_status_on_insert: PipelineProcessStatus = PipelineProcessStatus.NEW,
    mode: Literal["append", "replace"] = "append",
    notes: Optional[str] = None,
) -> None:
    """
    Incremental sync of stage table rows into `pipeline_control`.

    This function keeps the control-plane table aligned with data in a
    given source stage table.

    By default it operates in incremental mode:
        only inserting missing rows and refreshing selected metadata fields,
        without disturbing existing FSM stage or status values.

    This sync sets process_status=NEW for newly inserted control rows.
    Existing rows’ lifecycle is not modified;
    lifecycle is owned by the FSM/orchestrators.

    Behavior
    --------
    - Reads distinct rows from the given `source_table` based on its schema.
    - Inserts rows into `pipeline_control` when no matching control entry exists
      (primary key: url, iteration).
    - Optionally refreshes non-orchestration metadata fields
      (e.g., source_file, version).
    - Preserves `stage` and `status` on existing rows unless explicitly told
      to overwrite.
    - If `mode="replace"`, deletes matching control rows before re-inserting.
      This recreates the legacy “full resync” behavior.

    Parameters
    ----------
    source_table : TableName
        Stage/source table to sync from (e.g., JOB_URLS, JOB_POSTINGS).

    stage : PipelineStage
        Pipeline stage to assign on INSERT for these rows.

    insert_only : bool, default=True
        If True, insert only missing rows.

    mode : Literal["append", "replace"], default="append"
        If "append", INSERT only the rows that don't exist in `pipeline_control`.
        If "replace", DELETE matching control rows first and then re-INSERT all rows
        from the source table (used for bootstrap/recovery scenarios).

    refresh_meta_fields : tuple[str, ...], default=("source_file",)
        Metadata columns to refresh on existing rows. This never touches orchestration
        columns: `stage`, `status`, `decision_flag`, `transition_flag`, `created_at`,
        or `updated_at`.

    preserve_stage_on_update : bool, default=True
        If True, never overwrite an existing `stage` value.

    preserve_status_on_update : bool, default=True
        If True, never overwrite an existing `status` value.

    set_status_on_insert : PipelineStatus, default=PipelineStatus.NEW
        Status to assign on newly inserted rows.

    set_process_status_on_insert :
        PipelineProcessStatus, default=PipelineProcessStatus.NEW
        Process status to indicate whether a URL/job should be processed
            (vs skipped, completed, etc.).

    notes : str, optional
        Free-text notes to attach to inserted control rows.

    Notes
    -----
    - Use the default incremental settings for routine syncs (safe, idempotent).
    - Use `mode="replace"` and disable the preserve_* flags only for bootstrap
      or recovery scenarios where a full control-plane reset is required.
    """

    con = get_db_connection()

    # Source schema (used to know which columns are available for coalescing)
    source_schema = DUCKDB_SCHEMA_REGISTRY.get(source_table)
    if not source_schema:
        logger.error(f"Schema not found for {source_table}")
        return
    src_cols = set(source_schema.model.model_fields.keys())

    # Control-plane schema (authoritative list of target columns)
    control_schema = DUCKDB_SCHEMA_REGISTRY.get(TableName.PIPELINE_CONTROL)
    if not control_schema:
        logger.error("Schema not found for pipeline_control")
        return
    control_cols: list[str] = list(control_schema.model.model_fields.keys())

    # Composite PK used for de-dup / joining (fallback to url-only when missing in source)
    pk = control_schema.primary_keys
    join_keys = list_intersection(pk, src_cols) or ["url"]
    on_clause = " AND ".join([f"p.{c} = j.{c}" for c in join_keys])

    # Build a SELECT list for inserting into pipeline_control.
    # For each target control column, derive an expression from source or literals.
    select_exprs: list[str] = []
    for col in control_cols:
        if col == "url":
            select_exprs.append("j.url AS url")
        elif col == "iteration":
            select_exprs.append(
                f"{coalesce_col('iteration', '0', src_cols)} AS iteration"
            )
        elif col == "stage":
            select_exprs.append(f"{sql_str_lit(stage.value)} AS stage")
        elif col == "source_file":
            select_exprs.append(
                f"{coalesce_col('source_file', 'NULL', src_cols)} AS source_file"
            )
        elif col == "version":
            select_exprs.append(
                f"{coalesce_col('version', 'NULL', src_cols)} AS version"
            )
        elif col == "status":
            select_exprs.append(f"{sql_str_lit(set_status_on_insert.value)} AS status")
        elif col == "process_status":
            select_exprs.append(
                f"{sql_str_lit(set_process_status_on_insert.value)} AS process_status"
            )
        elif col == "decision_flag":  # <-- NEW
            select_exprs.append("0 AS decision_flag")
        elif col == "transition_flag":  # <-- NEW
            select_exprs.append("0 AS transition_flag")
        elif col == "notes":
            select_exprs.append(
                ("NULL" if notes is None else sql_str_lit(notes)) + " AS notes"
            )
        elif col == "created_at":
            select_exprs.append(
                f"{coalesce_col('created_at', 'now()', src_cols)} AS created_at"
            )
        elif col == "updated_at":
            select_exprs.append("now() AS updated_at")
        else:
            select_exprs.append(
                (f"j.{col} AS {col}") if col in src_cols else f"NULL AS {col}"
            )

        insert_cols_sql = ", ".join(control_cols)
        select_cols_sql = ",\n        ".join(select_exprs)

    # 1) INSERT new rows (append mode)
    if insert_only or mode == "append":
        con.execute(
            f"""
            INSERT INTO {TableName.PIPELINE_CONTROL.value} ({insert_cols_sql})
            SELECT
                {select_cols_sql}
            FROM {source_table.value} j
            LEFT JOIN {TableName.PIPELINE_CONTROL.value} p
              ON {on_clause}
            WHERE p.url IS NULL
            """
        )
        logger.info(f"✅ Control-plane: inserted NEW from {source_table.value}.")

    # 2) REPLACE: delete matching rows then re-insert (rare)
    if mode == "replace":
        con.execute(
            f"""
            DELETE FROM {TableName.PIPELINE_CONTROL.value} p
            USING {source_table.value} j
            WHERE {on_clause}
            """
        )
        con.execute(
            f"""
            INSERT INTO {TableName.PIPELINE_CONTROL.value} ({insert_cols_sql})
            SELECT
                {select_cols_sql}
            FROM {source_table.value} j
            """
        )
        logger.info(f"♻️  Control-plane: replaced rows from {source_table.value}.")

    # 3) Refresh selected metadata fields when they exist in source
    if refresh_meta_fields:
        # columns we are allowed to refresh (excluding orchestration fields)
        blocked = {
            "stage",
            "status",
            "created_at",
            "updated_at",
            "decision_flag",
            "transition_flag",
        }  # start with user-provided list ∩ (present-in-source) ∩ (present-in-control)
        refresh_set = {
            col
            for col in refresh_meta_fields
            if col not in blocked and col in src_cols and col in control_cols
        }

        if refresh_set:
            sets = [
                f"{col} = COALESCE(j.{col}, p.{col})" for col in sorted(refresh_set)
            ]
            # null-safe, type-safe change detection
            changed_predicates = [
                f"p.{col} IS DISTINCT FROM j.{col}" for col in sorted(refresh_set)
            ]

            # always bump updated_at when any refreshed column changes
            sets.append("updated_at = now()")

            set_clause = ", ".join(sets)
            change_clause = " OR ".join(changed_predicates)

            con.execute(
                f"""
                UPDATE {TableName.PIPELINE_CONTROL.value} p
                SET {set_clause}
                FROM {source_table.value} j
                WHERE {on_clause}
                AND ({change_clause})
                """
            )
            logger.info("🔄 Control-plane: refreshed metadata where changed.")

    # 4) Optional stage/status overrides (usually off)
    if not preserve_stage_on_update and "stage" in control_cols:
        con.execute(
            f"""
            UPDATE {TableName.PIPELINE_CONTROL.value} p
            SET stage = {sql_str_lit(stage.value)}, updated_at = now()
            FROM {source_table.value} j
            WHERE {on_clause}
              AND p.stage <> {sql_str_lit(stage.value)}
            """
        )
    if not preserve_status_on_update and "status" in control_cols:
        con.execute(
            f"""
            UPDATE {TableName.PIPELINE_CONTROL.value} p
            SET status = {sql_str_lit(set_status_on_insert.value)}, updated_at = now()
            FROM {source_table.value} j
            WHERE {on_clause}
              AND p.status <> {sql_str_lit(set_status_on_insert.value)}
            """
        )


# -------------------------------
# Data-driven sync plan
# -------------------------------

SYNC_PLAN: list[dict] = [
    # Order matters for bootstraps; tweak status as you prefer
    dict(
        source=TableName.JOB_URLS,
        stage=PipelineStage.JOB_URLS,
        status=PipelineStatus.NEW,
    ),
    dict(
        source=TableName.JOB_POSTINGS,
        stage=PipelineStage.JOB_POSTINGS,
        status=PipelineStatus.IN_PROGRESS,
    ),
    dict(
        source=TableName.EXTRACTED_REQUIREMENTS,
        stage=PipelineStage.EXTRACTED_REQUIREMENTS,
        status=PipelineStatus.IN_PROGRESS,
    ),
    dict(
        source=TableName.FLATTENED_REQUIREMENTS,
        stage=PipelineStage.FLATTENED_REQUIREMENTS,
        status=PipelineStatus.IN_PROGRESS,
    ),
    dict(
        source=TableName.FLATTENED_RESPONSIBILITIES,
        stage=PipelineStage.FLATTENED_RESPONSIBILITIES,
        status=PipelineStatus.IN_PROGRESS,
    ),
    dict(
        source=TableName.EDITED_RESPONSIBILITIES,
        stage=PipelineStage.EDITED_RESPONSIBILITIES,
        status=PipelineStatus.IN_PROGRESS,
    ),
    # Similarity metrics are often a gating signal; keep them last
    dict(
        source=TableName.SIMILARITY_METRICS,
        stage=PipelineStage.SIM_METRICS_EVAL,
        status=PipelineStatus.IN_PROGRESS,
        refresh_meta_fields=(),
    ),
]


# -------------------------------
# * Helper functions for each table sync with enum-aligned stage/version values
# * Optional to use: not directly need in the main pipeline!
# -------------------------------


def sync_job_urls_to_pipeline_control():
    logger.info("✅ Syncing job_urls with pipeline_control...")
    sync_table_to_pipeline_control(
        source_table=TableName.JOB_URLS,
        stage=PipelineStage.JOB_URLS,
        insert_only=True,
        refresh_meta_fields=("source_file",),  # single-element tuple
        preserve_stage_on_update=True,
        preserve_status_on_update=True,
        set_status_on_insert=PipelineStatus.NEW,
        set_process_status_on_insert=PipelineProcessStatus.NEW,
    )


def sync_job_postings_to_pipeline_control():
    logger.info("✅ Syncing job_postings with pipeline_control...")
    sync_table_to_pipeline_control(
        source_table=TableName.JOB_POSTINGS,
        stage=PipelineStage.JOB_POSTINGS,
        insert_only=True,  # fill gaps only
        refresh_meta_fields=("source_file",),
        preserve_stage_on_update=True,
        preserve_status_on_update=True,
        set_status_on_insert=PipelineStatus.IN_PROGRESS,
        set_process_status_on_insert=PipelineProcessStatus.NEW,
    )


def sync_extracted_requirements_to_pipeline_control():
    logger.info("✅ Syncing extracted_requirements with pipeline_control...")
    sync_table_to_pipeline_control(
        source_table=TableName.EXTRACTED_REQUIREMENTS,
        stage=PipelineStage.EXTRACTED_REQUIREMENTS,
        insert_only=True,
        refresh_meta_fields=("source_file",),
        preserve_stage_on_update=True,
        preserve_status_on_update=True,
        set_status_on_insert=PipelineStatus.IN_PROGRESS,
        set_process_status_on_insert=PipelineProcessStatus.NEW,
    )


def sync_flattened_requirements_to_pipeline_control():
    logger.info("✅ Syncing flattened_requirements with pipeline_control...")
    sync_table_to_pipeline_control(
        source_table=TableName.FLATTENED_REQUIREMENTS,
        stage=PipelineStage.FLATTENED_REQUIREMENTS,
        insert_only=True,
        refresh_meta_fields=("source_file",),
        preserve_stage_on_update=True,
        preserve_status_on_update=True,
        set_status_on_insert=PipelineStatus.IN_PROGRESS,
        set_process_status_on_insert=PipelineProcessStatus.NEW,
    )


def sync_flattened_responsibilities_to_pipeline_control():
    logger.info("✅ Syncing flattened_responsibilities with pipeline_control...")
    sync_table_to_pipeline_control(
        source_table=TableName.FLATTENED_RESPONSIBILITIES,
        stage=PipelineStage.FLATTENED_RESPONSIBILITIES,
        insert_only=True,
        refresh_meta_fields=("source_file",),
        preserve_stage_on_update=True,
        preserve_status_on_update=True,
        set_status_on_insert=PipelineStatus.IN_PROGRESS,
        set_process_status_on_insert=PipelineProcessStatus.NEW,
    )


def sync_edited_responsibilities_to_pipeline_control():
    # Edited responsibilities carry a version marker (original vs edited)

    logger.info("✅ Syncing edited_responsibilities with pipeline_control...")

    sync_table_to_pipeline_control(
        source_table=TableName.EDITED_RESPONSIBILITIES,
        stage=PipelineStage.EDITED_RESPONSIBILITIES,
        insert_only=True,
        refresh_meta_fields=("source_file", "version"),
        preserve_stage_on_update=True,
        preserve_status_on_update=True,
        set_status_on_insert=PipelineStatus.IN_PROGRESS,
        set_process_status_on_insert=PipelineProcessStatus.NEW,
    )


def sync_similarity_metrics_to_pipeline_control():

    logger.info("✅ Syncing similarity_metrics with pipeline_control...")

    sync_table_to_pipeline_control(
        source_table=TableName.SIMILARITY_METRICS,
        stage=PipelineStage.SIM_METRICS_EVAL,
        insert_only=True,
        refresh_meta_fields=("source_file",),
        preserve_stage_on_update=True,
        preserve_status_on_update=True,
        set_status_on_insert=PipelineStatus.IN_PROGRESS,
        set_process_status_on_insert=PipelineProcessStatus.NEW,
    )


# todo: commented out; no longer use; delete later
# def sync_all_tables_to_pipeline_control(
#     *,
#     full: bool = True,
#     create_table: bool = True,
#     integrity_check: bool = True,
# ) -> None:
#     """
#     Orchestrate syncing stage/source tables into `pipeline_control`
#     (FSM control plane).

#     When to run
#     -----------
#     - Bootstrap: first-time setup to seed control rows from existing
#         stage tables.
#     - Recovery: after failures or manual edits, to realign control rows.
#     - Periodic refresh: keep `pipeline_control` consistent with stage tables.

#     What it does
#     ------------
#     - Optionally ensures the `pipeline_control` table exists.
#     - If `full=True`, runs all stage syncs in order
#       (job_urls → edited_responsibilities).
#     - Always refreshes similarity metrics (often your gating signal).
#     - Optionally runs a quick FSM integrity check.

#     Notes
#     -----
#     - Each sync is isolated with its own try/except so one failure does not
#       halt the rest.
#     - This function does not execute the stage pipelines;
#       it only manages control state.
#     """

#     logger.info("🧭 Starting pipeline state orchestration...")

#     if create_table:
#         try:
#             create_single_db_table(table_name=TableName.PIPELINE_CONTROL)
#         except Exception as e:
#             logger.warning(
#                 f"⚠️ Could not ensure pipeline_control table exists: {e}", exc_info=True
#             )

#         # NEW: normalize any historical NULLs → 0
#         try:
#             con = get_db_connection()
#             try:
#                 con.execute(
#                     """
#                     SET decision_flag = CASE
#                         WHEN LOWER(status) IN ('completed','skipped') THEN 0
#                         ELSE COALESCE(decision_flag, 0) -- initialize but never downgrade 1
#                     END,
#                     transition_flag = CASE
#                         WHEN LOWER(status) IN ('completed','skipped') THEN 0
#                     ELSE COALESCE(transition_flag, 0)
#                     END
#                 """
#                 )
#             finally:
#                 con.close()
#         except Exception as e:
#             logger.warning("⚠️ Could not backfill NULL flags to 0: %s", e, exc_info=True)

#     if full:
#         for item in SYNC_PLAN:
#             try:
#                 sync_table_to_pipeline_control(
#                     source_table=item["source"],
#                     stage=item["stage"],
#                     set_status_on_insert=item["status"],
#                     insert_only=True,
#                     refresh_meta_fields=item.get(
#                         "refresh_meta_fields",
#                         ("source_file", "version"),
#                     ),
#                     preserve_stage_on_update=True,
#                     preserve_status_on_update=True,
#                     set_process_status_on_insert=PipelineProcessStatus.NEW,
#                     mode="append",
#                 )
#             except Exception as e:
#                 logger.error(
#                     f"❌ Failed syncing {item['source'].value}: {e}", exc_info=True
#                 )

#     if integrity_check:
#         try:
#             check_fsm_integrity()
#         except Exception as e:
#             logger.warning(
#                 f"⚠️ FSM integrity check encountered an issue: {e}", exc_info=True
#             )

#     logger.info("✅ Pipeline control table sync complete.")


# -------------------------------
# Sync to set decision flat to 1 (if stuck on 0)
# -------------------------------


# def sync_decision_flags_all(
#     *,
#     table_name: TableName = TableName.PIPELINE_CONTROL,
#     con=None,
# ) -> int:
#     """
#     Recompute decision_flag for ALL rows in pipeline_control
#     and reset transition_flag to 0.

#     Rule:
#       decision_flag = 0 if status or process_status IN {COMPLETED, SKIPPED}
#                       else keep existing (COALESCE(decision_flag, 0))
#     """
#     owns_con = con is None
#     if owns_con:
#         con = get_db_connection()
#     try:
#         st_completed = PipelineStatus.COMPLETED.value
#         st_skipped = getattr(
#             PipelineStatus, PipelineStatus.SKIPPED, PipelineStatus.COMPLETED
#         ).value
#         ps_completed = PipelineProcessStatus.COMPLETED.value
#         ps_skipped = getattr(
#             PipelineProcessStatus, "SKIPPED", PipelineProcessStatus.COMPLETED
#         ).value

#         res = con.execute(
#             f"""
#             UPDATE {table_name.value}
#             SET decision_flag = CASE
#                                    WHEN status IN (?, ?) OR process_status IN (?, ?)
#                                      THEN 0
#                                    ELSE COALESCE(decision_flag, 0)
#                                 END,
#                 transition_flag = COALESCE(transition_flag, 0),
#                 updated_at = now()
#             """,
#             (st_completed, st_skipped, ps_completed, ps_skipped),
#         )
#         updated = getattr(res, "rowcount", 0)
#         logger.info("🔄 decision_flag full-table sync updated %s rows.", updated)
#         return updated
#     finally:
#         if owns_con:
#             con.close()
