"""
pipelines_with_fsm/resume_editing_pipeline_async_fsm.py

DB-native Resume Editing Pipeline (FSM-driven; async; lease-aware claimables;
no filesystem I/O)

Worklist (from pipeline_control; human-gated + lease-aware):
  • EDIT: stage = EDITED_RESPONSIBILITIES,
    status ∈ {NEW[, ERROR if retry_errors=True]}

Inputs (per URL):
  • flattened_requirements        (requirement_key, requirement, url, …)
  • flattened_responsibilities    (responsibility_key, responsibility, url, …)

Output (per URL):
  • edited_responsibilities
      Columns (typical): url, responsibility_key, requirement_key,
        responsibility, iteration, version, llm_provider, stage, status,
        created_at, updated_at

Orchestration (claimables model):
  1) Build claimable worklist for EDITED_RESPONSIBILITIES with status {NEW[, ERROR]}.
  2) Generate a worker_id for this run.
  3) For each (url, iteration):
       a) try_claim_one(url, iteration, worker_id) → acquire lease
        or skip if already claimed.
       b) edit_and_persist_responsibilities_for_url(…) → pure compute/insert
        (no lease/FSM mutation).
       c) finalize_one_row_in_pipeline_control(url, iteration, worker_id, ok, notes)
          - Atomically sets final status (COMPLETED/ERROR)
          + clears lease iff ownership matches.
          - Returns True/False indicating whether finalize occurred
            (i.e., we still owned the lease).
       d) If ok and finalized → fsm.step() to advance to SIM_METRICS_REVAL
        (marks NEW there).

Notes:
  • Strictly DB-native: no JSON/CSV I/O.
  • Reuses Pydantic-validated loaders for rehydration & validation.
  • Mirrors the similarity-metrics FSM pipeline structure
    (bounded concurrency; gather).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence
import pandas as pd

# Pipeline enums / metadata
from job_bot.db_io.pipeline_enums import (
    PipelineStage,
    PipelineStatus,
    TableName,
)

# FSM
from job_bot.fsm.pipeline_fsm_manager import PipelineFSMManager

# Worklist + IO
from job_bot.db_io.db_inserters import insert_df_with_config
from job_bot.db_io.db_loaders import load_table

# Worklist + IO (lease-aware claimables)
from job_bot.db_io.db_utils import (
    get_claimable_worklist,
    try_claim_one,
    finalize_one_row_in_pipeline_control,
    generate_worker_id,
)

# enum fields
from job_bot.db_io.pipeline_enums import Version

# async editor to rewrite responsibilities against requirements
from job_bot.evaluation_optimization.resumes_editing_async import (
    modify_multi_resps_based_on_reqs_async,
)

# Pyd models (for types)
from job_bot.models.resume_job_description_io_models import (
    ResponsibilityMatches,
    Responsibilities,
    Requirements,
)

# Pipeline tools
from job_bot.pipelines_with_fsm.resume_editing.pair_filtering import (
    pick_pairs_to_edit_with_config,
    group_pairs_by_responsibility,
    PairFilterConfig,
)
from job_bot.pipelines_with_fsm.resume_editing.fill_with_original import (
    materialize_edited_responsibilities_rows,
)

# Configs
from job_bot.config.project_config import OPENAI, GPT_4_1_NANO, ANTHROPIC, CLAUDE_HAIKU

logger = logging.getLogger(__name__)


def get_resps_dict_strict(obj) -> Dict[str, str]:
    if isinstance(obj, Responsibilities):
        return dict(obj.responsibilities)
    raise TypeError(f"Expected Responsibilities, got {type(obj).__name__}")


def get_reqs_dict_strict(obj) -> Dict[str, str]:
    if isinstance(obj, Requirements):
        return dict(obj.requirements)
    raise TypeError(f"Expected Requirements, got {type(obj).__name__}")


# =============================================================================
# Per-URL worker
# =============================================================================
async def edit_and_persist_responsibilities_for_url(
    url: str,
    *,
    iteration: int,
    semaphore: asyncio.Semaphore,
    llm_provider: str,
    model_id: str,
    no_of_concurrent_workers_for_llm: int = 3,
) -> bool:
    """
    Edit (rewrite) resume responsibilities for a single URL using a
    similarity-metrics–guided, drift-safe workflow, and persist the
    results to DuckDB.

    High-level behavior
    -------------------
    This function performs **pair-gated responsibility editing**:

    - Only responsibility–requirement pairs deemed *eligible* by the
      ORIGINAL similarity_metrics pass are allowed to influence edits.
    - Responsibilities and requirements outside the eligible set are never
      shown to the LLM.
    - Missing or skipped eligible pairs are filled with the ORIGINAL
      responsibility text to preserve table completeness.

    The function is **DB-native**, **FSM-compatible**, and **idempotent per
    (url, iteration)**.

    Detailed steps
    --------------
    1) Load inputs from DuckDB:
       - flattened_responsibilities
       - flattened_requirements

    2) Load similarity_metrics (version='original') and select eligible
       (responsibility_key, requirement_key) pairs using score-based,
       per-responsibility filtering.

    3) Build filtered dictionaries:
       - responsibilities: only those with ≥1 eligible requirement
       - requirements: only those appearing in eligible pairs

    4) Run the async LLM editor **once** on the filtered dictionaries
       (no changes to the editor’s interface).

    5) Post-filter the editor output back to the *exact* eligible pairs.

    6) Materialize the final row set:
       - Use edited text where available
       - Fall back to ORIGINAL responsibility text for missing eligible pairs

    7) Insert rows into `edited_responsibilities` via `insert_df_with_config`,
       which stamps standard metadata (url, iteration, llm_provider, model_id)
       and handles deduplication according to table config.

    Concurrency
    -----------
    - URL-level concurrency is bounded by the provided semaphore.
    - LLM-level concurrency inside the editor is controlled by
      `no_of_concurrent_workers_for_llm`.

    Error handling and control flow
    -------------------------------
    - Any failure during load, filtering, editing, or insert:
        → logs the error
        → returns False
        → does NOT advance the FSM
    - If no eligible pairs exist:
        → no LLM call is made
        → no rows are inserted
        → returns True (treated as a successful no-op)

    Parameters
    ----------
    url:
        Canonical job posting URL.
    iteration:
        Iteration stamp for auditing and controlled reruns.
    semaphore:
        Async semaphore bounding concurrent URL processing.
    llm_provider:
        LLM provider label (e.g., "openai", "anthropic").
    model_id:
        LLM model identifier used for responsibility editing.
    no_of_concurrent_workers_for_llm:
        Internal concurrency limit used by the editor per URL.

    Returns
    -------
    bool
        True if the operation succeeded or was a safe no-op;
        False if a failure occurred and the URL should be marked ERROR.
    """
    async with semaphore:
        # -------------------------
        # 1) Load inputs (resps/reqs)
        # -------------------------
        try:
            resps_model = load_table(TableName.FLATTENED_RESPONSIBILITIES, url=url)
            reqs_model = load_table(TableName.FLATTENED_REQUIREMENTS, url=url)

            resps_dict = get_resps_dict_strict(resps_model)
            reqs_dict = get_reqs_dict_strict(reqs_model)

            if not resps_dict or not reqs_dict:
                raise ValueError(
                    "Empty responsibilities or requirements after rehydration"
                )

            # todo: debug - delete later
            logger.debug(
                "Loaded inputs for %s | responsibilities=%d | requirements=%d",
                url,
                len(resps_dict),
                len(reqs_dict),
            )

        except Exception:
            logger.exception("❌ Failed to load/rehydrate inputs for %s", url)
            return False

        # ----------------------------------------
        # 2) Load metrics + pick eligible pairs
        # ----------------------------------------
        try:
            df_metrics = load_table(
                TableName.SIMILARITY_METRICS,
                url=url,
                version=Version.ORIGINAL.value,
            )
            if not isinstance(df_metrics, pd.DataFrame) or df_metrics.empty:
                raise ValueError(
                    "No similarity_metrics found for url (version='original')."
                )

            # todo: debug; delete later
            logger.debug(
                "Loaded similarity_metrics for %s | rows=%d | version=%s",
                url,
                len(df_metrics),
                Version.ORIGINAL.value,
            )

            pair_cfg = PairFilterConfig(
                min_score=0.45,
                top_k=2,
                min_keep_per_resp=0,  # recommend while debugging
                min_keep_score_floor=None,  # disable floor logic if min_keep=0 anyway
            )

            # todo: debug; delete later
            logger.info(
                "Pair filter params for %s | min_score=%.2f top_k=%d min_keep=%d floor=%s",
                url,
                pair_cfg.min_score,
                pair_cfg.top_k,
                pair_cfg.min_keep_per_resp,
                str(pair_cfg.min_keep_score_floor),
            )

            eligible_pairs = pick_pairs_to_edit_with_config(df_metrics, pair_cfg)

        except Exception:
            logger.exception("❌ Failed to load/filter similarity_metrics for %s", url)
            return False

        if not eligible_pairs:
            # No safe edits to make -> return True (nothing to do) or False?
            # I strongly recommend True so pipeline can progress without ERROR.
            logger.info("🟨 No eligible pairs to edit for %s (skipping insert).", url)
            return True

        eligible_map = group_pairs_by_responsibility(eligible_pairs)
        eligible_set = {(str(r), str(q)) for (r, q) in eligible_pairs}

        # todo: debug; delete later
        logger.info(
            "Pair filtering for %s | eligible_pairs=%d | responsibilities_affected=%d",
            url,
            len(eligible_pairs),
            len(eligible_map),
        )

        if logger.isEnabledFor(logging.DEBUG):
            # show a small sample to confirm sanity
            sample = list(eligible_pairs)[:5]
            logger.debug("Eligible pair sample for %s: %s", url, sample)
        # todo: delete later

        # ----------------------------------------
        # 3) Build filtered dicts for editor call
        # ----------------------------------------
        resps_sub_dict = {
            rk: resps_dict[rk] for rk in eligible_map.keys() if rk in resps_dict
        }

        eligible_req_keys = {qk for _, qk in eligible_pairs}
        reqs_sub_dict = {
            qk: reqs_dict[qk] for qk in eligible_req_keys if qk in reqs_dict
        }

        if not resps_sub_dict or not reqs_sub_dict:
            logger.warning(
                "🟨 Eligible pairs exist but filtered dicts are empty for %s "
                "(resps_sub=%d reqs_sub=%d). Skipping.",
                url,
                len(resps_sub_dict),
                len(reqs_sub_dict),
            )
            return True

        # -------------------------
        # 4) Run editor (unchanged)
        # -------------------------
        # todo: debug; delete later
        logger.debug(
            "Editor input for %s | resps_sub=%d | reqs_sub=%d",
            url,
            len(resps_sub_dict),
            len(reqs_sub_dict),
        )

        try:
            matches: ResponsibilityMatches = (
                await modify_multi_resps_based_on_reqs_async(
                    responsibilities=resps_sub_dict,
                    requirements=reqs_sub_dict,
                    llm_provider=llm_provider,
                    model_id=model_id,
                    no_of_concurrent_workers=no_of_concurrent_workers_for_llm,
                    eligible_map=eligible_map,
                )
            )
        except Exception:
            logger.exception("❌ Editor failed for %s", url)
            return False

        # todo: debug; delete later
        total_emitted = sum(
            len(by_req.optimized_by_requirements)
            for by_req in matches.responsibilities.values()
        )

        logger.info(
            "Editor output for %s | responsibilities=%d | emitted_pairs=%d",
            url,
            len(matches.responsibilities),
            total_emitted,
        )

        # ----------------------------------------------------------
        # 5) Post-filter output to exact eligible pairs + fill missing
        # ----------------------------------------------------------
        edited_map: dict[tuple[str, str], str] = {}

        # Flatten ONLY eligible pairs emitted by the model
        for resp_key, by_req in matches.responsibilities.items():
            rk = str(resp_key)
            for req_key, optimized_text in by_req.optimized_by_requirements.items():
                qk = str(req_key)
                if (rk, qk) not in eligible_set:
                    continue

                # Create the lookup table
                # output -> edited_map: {(resp_key, req_key) -> edited_text}
                edited_map[(rk, qk)] = str(optimized_text.optimized_text)

        # ----------------------------------------------------------
        # 6) Materialize FULL matrix (all resps × all reqs)
        #    - Eligible pairs: use edited (fallback to original if missing)
        #    - Non-eligible pairs: always original
        # ----------------------------------------------------------
        final_rows: list[dict[str, Any]] = []

        # Canonical requirement universe for "structure maintains"
        all_req_keys = list(reqs_dict.keys())  # preserves dict insertion order (Py3.7+)

        # Matrialize entire table
        for rk, orig_text in resps_dict.items():
            for qk in all_req_keys:
                if (rk, qk) in eligible_set:
                    text = edited_map.get(
                        (rk, qk), orig_text
                    )  # edited if present, else original
                else:
                    text = orig_text

                final_rows.append(
                    {
                        "url": url,
                        "responsibility_key": rk,
                        "requirement_key": qk,
                        "responsibility": text,
                    }
                )

        if not final_rows:
            logger.info("🟨 No rows to insert after post-filter/fill for %s.", url)
            return True

        # todo: debug; delete later
        total_pairs = len(final_rows)
        eligible_total = len(eligible_set)
        edited_emitted = len(edited_map)

        non_eligible_original = total_pairs - eligible_total
        eligible_fallback_original = eligible_total - edited_emitted

        logger.info(
            "Materialized edited_responsibilities for %s | total_pairs=%d | eligible_pairs=%d | "
            "edited_emitted=%d | non_eligible_original=%d | eligible_fallback_original=%d",
            url,
            total_pairs,
            eligible_total,
            edited_emitted,
            non_eligible_original,
            eligible_fallback_original,
        )

        # -------------------------
        # 7) Insert
        # -------------------------
        # todo: debug; delete later
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Insert preview for %s: %s",
                url,
                final_rows[:3],
            )

        try:
            edited_df = pd.DataFrame(final_rows)
            logger.debug(
                "Edited sample: %s", edited_df.head(2).to_dict(orient="records")
            )

            insert_df_with_config(
                edited_df,
                TableName.EDITED_RESPONSIBILITIES,
                url=url,
                llm_provider=llm_provider,
                model_id=model_id,
                iteration=iteration,
            )
            return True

        except Exception:
            logger.exception("❌ Persist failed for %s", url)
            return False


# =============================================================================
# Batch runner (bounded concurrency)
# =============================================================================
async def process_resume_editing_batch_async_fsm(
    url_iter_pairs: List[tuple[str, int]],
    *,
    worker_id: str,
    llm_provider: str,
    model_id: str,
    max_concurrent_urls: int = 4,
    no_of_concurrent_workers_for_llm: int = 3,
) -> list[asyncio.Task]:
    """
    Claim → run editing → finalize/step for a batch of (url, iteration) pairs.

    - Attempts to claim each (url, iteration) with `worker_id`.
    - If claimed, performs pure editing compute/insert.
    - Finalizes row with ok status and steps FSM on success (if we still own lease).
    """
    semaphore = asyncio.Semaphore(max_concurrent_urls)
    fsm_manager = PipelineFSMManager()

    async def _run_one(url: str, iteration: int) -> None:
        # Acquire lease or skip
        if not try_claim_one(url=url, iteration=iteration, worker_id=worker_id):
            logger.info("⏭️ Skipping %s@%s — already claimed.", url, iteration)
            return

        try:
            ok = await edit_and_persist_responsibilities_for_url(
                url,
                iteration=iteration,
                semaphore=semaphore,
                llm_provider=llm_provider,
                model_id=model_id,
                no_of_concurrent_workers_for_llm=no_of_concurrent_workers_for_llm,
            )

            finalized = finalize_one_row_in_pipeline_control(
                url=url,
                iteration=iteration,
                worker_id=worker_id,
                ok=ok,
                notes="Edited responsibilities saved to DB" if ok else "Editing failed",
            )

            if not finalized:
                logger.warning(
                    "[finalize] Lost lease for %s@%s; not stepping.", url, iteration
                )
                return

            if ok:
                try:
                    # EDITED_RESPONSIBILITIES → SIM_METRICS_REVAL
                    expected_source_stage = PipelineStage.EDITED_RESPONSIBILITIES

                    fsm = fsm_manager.get_fsm(url)
                    if fsm.get_current_stage() == expected_source_stage.value:
                        fsm.step()
                except Exception:
                    logger.exception("FSM step() failed for %s@%s", url, iteration)

        except Exception as e:
            logger.exception("❌ Failure in _run_one for %s@%s: %s", url, iteration, e)
            # Best-effort error finalize (still lease-validated)
            finalized = finalize_one_row_in_pipeline_control(
                url=url,
                iteration=iteration,
                worker_id=worker_id,
                ok=False,
                notes=f"editing failed: {e}",
            )
            if not finalized:
                logger.warning(
                    "[finalize] Could not mark ERROR for %s@%s (lease mismatch).",
                    url,
                    iteration,
                )

    return [asyncio.create_task(_run_one(u, it)) for (u, it) in url_iter_pairs]


# =============================================================================
# Entrypoint (stage worklist → tasks → await)
# =============================================================================
async def run_resume_editing_pipeline_async_fsm(
    *,
    llm_provider: str = OPENAI,
    model_id: str = GPT_4_1_NANO,
    max_concurrent_urls: int = 4,
    no_of_concurrent_workers_for_llm: int = 3,
    filter_urls: Optional[Sequence[str]] = None,
    limit_urls: Optional[int] = None,
    retry_errors: bool = False,
) -> None:
    """
    FSM-aware entrypoint for **editing** flattened responsibilities and persisting
    them into DuckDB.

    Workflow (lease-aware, human-gated, claimables pattern)
    -------------------------------------------------------
    1) Build worklist (DB): call
         get_claimable_worklist(stage=EDITED_RESPONSIBILITIES, status={NEW[, ERROR]})
       • Enforces human gate (task_state='READY') and lease rules
        (unclaimed or lease expired).
       • Returns a list of (url, iteration) pairs.
    2) Optional filter: if `filter_urls` provided, restrict worklist to those URLs;
        if `limit_urls` is provided, truncate the list.
    3) Worker identity: generate a `worker_id` via generate_worker_id("resume_editing").
    4) Process batch (bounded concurrency):
         For each (url, iteration) in the worklist:
           a) try_claim_one(url, iteration, worker_id) — acquire a lease or skip
            if already claimed.
           b) edit_and_persist_responsibilities_for_url(...) — pure compute & insert,
            no lease/FSM mutation.
           c) finalize_one_row_in_pipeline_control(url, iteration, worker_id, ok, notes)
            — atomically
              writes final status and clears lease iff we still own it;
                returns bool (finalized or not).
           d) If ok and finalized → fsm.step() to advance to SIM_METRICS_REVAL.
    5) Await all tasks and log completion.

    Parameters
    ----------
        llm_provider : str
            Provider label used for stamping (e.g., "openai" or "anthropic").
        model_id : str
            Model ID used by the editing function.
        max_concurrent_urls : int
            Maximum number of concurrent URL tasks (outer-level semaphore).
        no_of_concurrent_workers_for_llm : int
            Internal concurrency inside the editor per URL.
        filter_urls : Optional[Sequence[str]]
            Optional subset of URLs to process.
        limit_urls : Optional[int]
            Optional cap on how many URLs to pull from the worklist
                (after filtering).
        retry_errors : bool
            If True, include ERROR rows in the claimable worklist in addition to NEW.

    Returns
    -------
    None

    Side effects:
        include writing rows to DuckDB and updating the `pipeline_control`
        FSM state.

    Notes
    -----
    • Idempotent per (url, iteration): inserter config should deduplicate
        on your chosen keys.
    • Keep concurrency modest to avoid long lease holds during LLM calls.

    Concurrency
    -----------
    - Uses a stage-level semaphore. The editor can also run multiple
        internal workers per-URL (`no_of_concurrent_workers_for_llm`).

    Error Handling
    --------------
    - If no URLs match, the function logs and returns early.
    - For individual URLs:
        * Any load/LLM/insert/FSM error logs, marks status = `ERROR`,
          does not advance, and continues with other URLs.
    - The pipeline completes even if some URLs fail.
    """
    statuses = (
        (PipelineStatus.NEW, PipelineStatus.ERROR, PipelineStatus.IN_PROGRESS)
        if retry_errors
        else (PipelineStatus.NEW,)
    )
    worklist: List[tuple[str, int]] = get_claimable_worklist(
        stage=PipelineStage.EDITED_RESPONSIBILITIES,
        status=statuses,
        max_rows=max(1000, max_concurrent_urls * 4),
    )

    if filter_urls:
        filt = set(filter_urls)
        worklist = [(u, it) for (u, it) in worklist if u in filt]

    if not worklist:
        logger.info(
            f"📭 No claimable rows at {PipelineStage.EDITED_RESPONSIBILITIES.value}."
        )
        return

    if limit_urls:
        worklist = worklist[:limit_urls]

    worker_id = generate_worker_id("resume_editing")
    logger.info(
        "✏️ Starting resume editing | %d item(s) | worker_id=%s | provider=%s model=%s",
        len(worklist),
        worker_id,
        llm_provider,
        model_id,
    )

    tasks = await process_resume_editing_batch_async_fsm(
        url_iter_pairs=worklist,
        worker_id=worker_id,
        llm_provider=llm_provider,
        model_id=model_id,
        max_concurrent_urls=max_concurrent_urls,
        no_of_concurrent_workers_for_llm=no_of_concurrent_workers_for_llm,
    )

    await asyncio.gather(*tasks)

    logger.info("✅ Finished resume editing FSM pipeline.")
