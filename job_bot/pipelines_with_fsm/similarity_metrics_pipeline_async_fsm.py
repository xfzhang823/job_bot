"""
pipelines_with_fsm/similarity_pipeline_async_fsm.py

DB-native similarity evaluation pipeline (FSM-driven; lease-aware claimables).

Worklists (from pipeline_control, human-gated + lease-aware):
   • EVAL:    SIM_METRICS_EVAL / NEW
    (optionally include ERROR if retry_errors=True)
   • RE-EVAL: SIM_METRICS_REVAL / NEW
    (optionally include ERROR if retry_errors=True)

DB inputs (per URL):
  • flattened_requirements    (requirement_key, requirement)
  • responsibilities:
      - EVAL   → flattened_responsibilities
        (responsibility_key, responsibility)
      - RE-EVAL→ edited_responsibilities
        (responsibility_key, responsibility, requirement_key)

Output:
  • similarity_metrics (rows stamped with stage via inserter config)
    - version: Version.ORIGINAL (EVAL) or Version.EDITED (RE-EVAL)
    - resp_llm_provider / resp_model_id stamped via inserter for RE-EVAL

Orchestration (claimables model):
  1) Build claimable worklist
    (stage=SIM_METRICS_EVAL|SIM_METRICS_REVAL, status IN NEW[, ERROR])
    respecting human gate and expired leases.
  2) Generate a worker_id for this run.
  3) For each (url, iteration):
       a) try_claim_one(url, iteration, worker_id) → acquire lease or skip.
       b) evaluate_similarity_for_url_async(...) → pure compute + DB insert
        (no FSM mutation).
       c) finalize_one_row_in_pipeline_control(url, iteration, worker_id, ok=...)
          - Sets final status & clears lease if worker still owns it.
       d) If ok: fsm.step() to advance the control-plane stage.

Notes:
  • Vectorized compute on DataFrames for speed.
  • Keys canonicalized as *_key (requirement_key, responsibility_key).
  • URL is the unit of work; idempotent and safe for concurrency.
"""

from __future__ import annotations

# --- Standard & 3rd party imports
import asyncio
import logging
from typing import Any, Dict, Optional
from time import perf_counter
from collections.abc import Hashable
import numpy as np
from pydantic import TypeAdapter
from time import perf_counter
import pandas as pd

# --- Project imports ---
from job_bot.fsm.pipeline_fsm_manager import PipelineFSMManager
from job_bot.db_io.db_inserters import insert_df_with_config
from job_bot.db_io.pipeline_enums import (
    TableName,
    Version,
    PipelineStage,
    PipelineStatus,
)
from job_bot.db_io.db_schema_registry import DUCKDB_SCHEMA_REGISTRY
from job_bot.db_io.db_readers import (
    fetch_edited_responsibilities,
    fetch_flattened_responsibilities,
    fetch_flattened_requirements,
)
from job_bot.db_io.state_sync import load_pipeline_state
from job_bot.db_io.db_utils import (
    get_claimable_worklist,
    try_claim_one,
    finalize_one_row_in_pipeline_control,
    generate_worker_id,
)

# Reuse compute from your evaluation code (computation only; no file I/O)
from job_bot.evaluation_optimization.metrics_calculator import (
    categorize_scores_for_df,
    calculate_text_similarity_metrics,
)
from job_bot.evaluation_optimization.evaluation_optimization_utils import (
    add_multivariate_indices,
)

# Pyd models for row-level schema validation of similarity metrics
from job_bot.models.resume_job_description_io_models import SimilarityMetrics


logger = logging.getLogger(__name__)

# Toggle to enforce row-level Pydantic validation before DB insert.
VALIDATE_WITH_PYDANTIC: bool = False  # Set to False for faster speed


# =============================================================================
# Per-URL worker (pure compute + insert; does not mutate leases or FSM)
# =============================================================================


async def evaluate_similarity_for_url_async(
    url: str,
    *,
    iteration: int,
    metrics_stage: PipelineStage,  # SIM_METRICS_EVAL or SIM_METRICS_REVAL
) -> bool:
    """
    Evaluate and persist similarity metrics for a single (url, iteration).

    Caller is responsible for claim/finalize and FSM step. This function:
      1) Loads responsibilities + flattened requirements from DuckDB (per stage).
      2) Builds (responsibility, requirement) pairs.
      3) Scores pairs concurrently, adds categories and multivariate indices.
      4) Normalizes/stamps required columns and inserts rows into similarity_metrics.

    Returns:
      True on success (rows persisted), False on any failure or empty result.
    """
    # Load inputs
    reqs = await asyncio.to_thread(fetch_flattened_requirements, url)
    if metrics_stage == PipelineStage.SIM_METRICS_EVAL:
        resps = fetch_flattened_responsibilities(url)
        version = Version.ORIGINAL
        resp_prov: tuple[str | None, str | None] = (None, None)
    elif metrics_stage == PipelineStage.SIM_METRICS_REVAL:
        resps = fetch_edited_responsibilities(url)
        version = Version.EDITED
        # Infer a single (llm_provider, model_id) pair for provenance
        pairs = list(
            resps[["llm_provider", "model_id"]]
            .dropna()
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if not pairs:
            logger.error(
                "No (llm_provider, model_id) for edited responsibilities: %s", url
            )
            return False
        if len(pairs) > 1:
            logger.error(
                "Multiple edited sets for %s; provide editor_llm_provider/model_id. Found: %s",
                url,
                pairs,
            )
            return False
        resp_prov = pairs[0]
    else:
        logger.error("Unsupported metrics_stage: %s", metrics_stage)
        return False

    if reqs.empty or resps.empty:
        logger.warning(
            "%s: missing reqs/resps (reqs=%d, resps=%d)", url, len(reqs), len(resps)
        )
        return False

    try:
        # Build pairs for BOTH stages
        pairs_df = build_pairs_df(resps, reqs, stage=metrics_stage)
        if pairs_df.empty:
            logger.error(
                "%s: empty pairs_df after build_pairs_df (%s)", url, metrics_stage
            )
            return False

        # Score
        df = await _compute_similarity_df_async(pairs_df)
        if df.empty:
            return False

        # Normalize keys, ensure URL, and validate if enabled
        df = _normalize_similarity_keys(df)
        if "url" not in df.columns:
            df.insert(0, "url", url)

        if VALIDATE_WITH_PYDANTIC:
            df = _validate_similarity_rows(df)

        # Required columns
        if not {"url", "responsibility_key", "requirement_key"}.issubset(df.columns):
            logger.error("Missing required columns after normalization/validation")
            return False

        # ---- Add categories and multivariate indices BEFORE stamping/inserting ----
        try:
            df = categorize_scores_for_df(df)
        except Exception as e:
            logger.warning("Categorization failed; continuing without *_cat: %s", e)

        try:
            df = add_multivariate_indices(df)
        except Exception as e:
            logger.warning(
                "Index computation failed; will continue without indices: %s", e
            )

        # Stage-owned fields
        df["version"] = version.value

        # If your table owns backend columns and they’re missing, set defaults
        schema = DUCKDB_SCHEMA_REGISTRY[TableName.SIMILARITY_METRICS]
        owned = set(getattr(schema, "metadata_fields", [])) | set(schema.column_order)
        for col in ("similarity_backend", "nli_backend"):
            if col in owned and col not in df.columns:
                df[col] = None

        # Rename optimized_text / requirement_text back to responsibility / requirement if needed
        if "optimized_text" in df.columns and "responsibility" not in df.columns:
            df["responsibility"] = df["optimized_text"]
        if "requirement_text" in df.columns and "requirement" not in df.columns:
            df["requirement"] = df["requirement_text"]

        # Insert (REVAL stamps editor provenance via YAML stamp:param)
        if version == Version.EDITED:
            if not VALIDATE_WITH_PYDANTIC:
                for col in ("resp_llm_provider", "resp_model_id"):
                    if col not in df.columns:
                        df[col] = None

            insert_df_with_config(
                df,
                TableName.SIMILARITY_METRICS,
                url=url,
                resp_llm_provider=resp_prov[0],  # YAML stamp:param
                resp_model_id=resp_prov[1],  # YAML stamp:param
                iteration=iteration,  # ← pass iteration
            )
        else:
            insert_df_with_config(
                df,
                TableName.SIMILARITY_METRICS,
                url=url,
                resp_llm_provider="N/A",
                resp_model_id="N/A",
                iteration=iteration,  # ← pass iteration
            )

        logger.info(
            "✅ Similarity rows inserted: %s (stage=%s)", url, metrics_stage.value
        )
        return True

    except Exception:
        logger.exception("❌ Compute/insert failed for %s", url)
        return False


# =============================================================================
# Batch runner (claim → run → finalize → step)
# =============================================================================


async def process_similarity_batch_async_fsm(
    url_iter_pairs: list[tuple[str, int]],
    *,
    worker_id: str,
    metrics_stage: PipelineStage,
    no_of_concurrent_workers: int = 5,
) -> list[asyncio.Task]:
    """
    Claim → run pure op → finalize/step. Mirrors the updated patterns used elsewhere.

    - The worker attempts to claim each (url, iteration).
    - If claimed, it performs similarity compute/insert.
    - Finalizes row with ok status and steps FSM on success.
    """
    semaphore = asyncio.Semaphore(no_of_concurrent_workers)
    fsm_manager = PipelineFSMManager()

    async def run_one(url: str, iteration: int) -> None:
        # Optional: pre-flight existence check
        if load_pipeline_state(url) is None:
            logger.warning("⚠️ No pipeline_state row for %s", url)

        # Lease: claim or skip
        if not try_claim_one(url=url, iteration=iteration, worker_id=worker_id):
            logger.info("⏭️ Skipping %s@%s — already claimed.", url, iteration)
            return

        try:
            ok = await evaluate_similarity_for_url_async(
                url,
                iteration=iteration,
                semaphore=semaphore,
                metrics_stage=metrics_stage,
            )

            finalized = finalize_one_row_in_pipeline_control(
                url=url,
                iteration=iteration,
                worker_id=worker_id,
                ok=ok,
                notes="Similarity saved to DB" if ok else "Similarity compute failed",
            )
            if not finalized:
                logger.warning(
                    "[finalize] Lost lease for %s@%s; not stepping.", url, iteration
                )
                return

            if ok:
                try:
                    expected_source_stage = metrics_stage
                    fsm = fsm_manager.get_fsm(url)
                    if fsm.get_current_stage() == expected_source_stage.value:
                        fsm.step()
                    else:
                        logger.info(
                            "Not stepping %s@%s: stage moved from %s → %s elsewhere.",
                            url,
                            iteration,
                            expected_source_stage.value,
                            fsm.state,
                        )
                except Exception:
                    logger.exception("FSM step() failed for %s@%s", url, iteration)
        except Exception as e:
            logger.exception("❌ Failure in run_one for %s@%s: %s", url, iteration, e)
            # Best-effort error finalize (still lease-validated)
            finalized = finalize_one_row_in_pipeline_control(
                url=url,
                iteration=iteration,
                worker_id=worker_id,
                ok=False,
                notes=f"similarity failed: {e}",
            )
            if not finalized:
                logger.warning(
                    "[finalize] Could not mark ERROR for %s@%s (lease mismatch).",
                    url,
                    iteration,
                )

    return [asyncio.create_task(run_one(u, it)) for (u, it) in url_iter_pairs]


# =============================================================================
# Entrypoints
# =============================================================================


async def run_similarity_metrics_eval_pipeline_async_fsm(
    *,
    max_concurrent_tasks: int = 5,
    filter_keys: Optional[list[str]] = None,
    retry_errors: bool = False,
) -> None:
    """
    Run the similarity **EVAL** pipeline (flattened_responsibilities × flattened_requirements).

    Workflow (lease-aware, human-gated, claimables pattern)
    -------------------------------------------------------
    1) Build worklist (DB): call `get_claimable_worklist(stage=SIM_METRICS_EVAL, status=...)`
       • Status set = {NEW} by default, or {NEW, ERROR} if `retry_errors=True`
       • Enforces human gate (`task_state='READY'`) and lease rules (not claimed or lease expired)
       • Returns a list of (url, iteration) pairs
    2) Optional filter: if `filter_keys` provided, restrict worklist to those URLs.
    3) Worker identity: generate a `worker_id` (stable per run) via `generate_worker_id("sim_metrics_eval")`.
    4) Process batch (bounded concurrency):
       For each (url, iteration) in the worklist:
         a) `try_claim_one(url, iteration, worker_id)` — acquire a lease or skip if already claimed
         b) `evaluate_similarity_for_url_async(...)` — pure compute & insert (no lease/FSM mutation):
              - Load inputs: flattened responsibilities + flattened requirements
              - Build pairs: full m×n cartesian product (per URL)
              - Score: concurrent CPU-bound similarity metrics (thread offloaded)
              - Categorize + multivariate indices; stamp version=ORIGINAL
              - Insert rows into `similarity_metrics` (via `insert_df_with_config`)
         c) Finalize:
              `finalize_one_row_in_pipeline_control(url, iteration, worker_id, ok, notes)`
              - Atomically sets final status (COMPLETED/ERROR) and clears lease **iff** `worker_id` still owns it
              - Returns True/False (finalized or not)
         d) Step on success:
              If `ok` and `finalized` → `fsm.step()` to advance the control-plane stage
    5) Await all tasks and log completion.

    Parameters
    ----------
    max_concurrent_tasks : int
        Maximum number of (url, iteration) workers running simultaneously.
    filter_keys : list[str] | None
        Optional restrict-to-URLs filter (bypasses other URLs present in the worklist).
    retry_errors : bool
        If True, include rows with status=ERROR in the claimable worklist in addition to NEW.

    Returns
    -------
    None

    Notes
    -----
    • This entrypoint does not mutate leases directly; claim/finalize/step is handled per item.
    • The EVAL path stamps Version.ORIGINAL and uses `resp_llm_provider="N/A"`, `resp_model_id="N/A"`.
    • Idempotent per URL/iteration: repeated runs safely dedupe at insert according to your table config.
    """
    statuses = (
        (PipelineStatus.NEW, PipelineStatus.ERROR)
        if retry_errors
        else (PipelineStatus.NEW,)
    )
    worklist: list[tuple[str, int]] = get_claimable_worklist(
        stage=PipelineStage.SIM_METRICS_EVAL,
        status=statuses,
        max_rows=max(1000, max_concurrent_tasks * 4),
    )

    if filter_keys:
        filt = set(filter_keys)
        worklist = [(u, it) for (u, it) in worklist if u in filt]

    if not worklist:
        logger.info("📭 No claimable rows at SIM_METRICS_EVAL.")
        return

    worker_id = generate_worker_id("sim_metrics_eval")
    logger.info(
        "🚀 Starting SIM_METRICS_EVAL | %d items | worker_id=%s",
        len(worklist),
        worker_id,
    )

    tasks = await process_similarity_batch_async_fsm(
        url_iter_pairs=worklist,
        worker_id=worker_id,
        metrics_stage=PipelineStage.SIM_METRICS_EVAL,
        no_of_concurrent_workers=max_concurrent_tasks,
    )
    await asyncio.gather(*tasks)
    logger.info("✅ Finished similarity FSM run (SIM_METRICS_EVAL).")


async def run_similarity_metrics_reval_pipeline_async_fsm(
    *,
    max_concurrent_tasks: int = 5,
    filter_keys: Optional[list[str]] = None,
    retry_errors: bool = False,
) -> None:
    """
    Run the similarity **RE-EVAL** pipeline (edited_responsibilities aligned
        to requirements).

    Workflow (lease-aware, human-gated, claimables pattern)
    -------------------------------------------------------
    1) Build worklist (DB):
        call `get_claimable_worklist(stage=SIM_METRICS_REVAL, status=...)`
       • Status set = {NEW} by default, or {NEW, ERROR} if `retry_errors=True`
       • Enforces human gate (`task_state='READY'`) and lease rules (not claimed or
        lease expired)
       • Returns a list of (url, iteration) pairs
    2) Optional filter: if `filter_keys` provided, restrict worklist to those URLs.
    3) Worker identity:
        generate a `worker_id` via `generate_worker_id("sim_metrics_reval")`.
    4) Process batch (bounded concurrency):
       For each (url, iteration) in the worklist:
         a) `try_claim_one(url, iteration, worker_id)` — acquire a lease or skip
            if already claimed
         b) `evaluate_similarity_for_url_async(...)` — pure compute & insert
            (no lease/FSM mutation):
              - Load inputs: edited responsibilities (pre-aligned to `requirement_key`)
              + flattened requirements
              - Build pairs: join on `requirement_key` (no cartesian product)
              - Score: concurrent CPU-bound similarity metrics (thread offloaded)
              - Categorize + multivariate indices; stamp version=EDITED
              - Insert rows into `similarity_metrics`, stamping editor provenance:
                    `resp_llm_provider`,
                    `resp_model_id` (from edited rows; validated single pair)
         c) Finalize:
              `finalize_one_row_in_pipeline_control(url, iteration, worker_id, ok, notes)`
              - Atomically sets final status (COMPLETED/ERROR)
                and clears lease **iff** `worker_id` still owns it
              - Returns True/False (finalized or not)
         d) Step on success:
              If `ok` and `finalized` → `fsm.step()` to advance the control-plane stage
    5) Await all tasks and log completion.

    Parameters
    ----------
    max_concurrent_tasks : int
        Maximum number of (url, iteration) workers running simultaneously.
    filter_keys : list[str] | None
        Optional restrict-to-URLs filter (bypasses other URLs present in the worklist).
    retry_errors : bool
        If True, include rows with status=ERROR in the claimable worklist in addition to NEW.

    Returns
    -------
    None

    Notes
    -----
    • The RE-EVAL path requires a single (llm_provider, model_id) pair across edited rows;
        if multiple or none
      are detected, the item fails and is finalized as ERROR.
    • Idempotent per URL/iteration: repeated runs safely dedupe at insert
        according to your table config.
    • Keep concurrency modest if edited sets are large to avoid long leases.
    """
    statuses = (
        (PipelineStatus.NEW, PipelineStatus.ERROR)
        if retry_errors
        else (PipelineStatus.NEW,)
    )
    worklist: list[tuple[str, int]] = get_claimable_worklist(
        stage=PipelineStage.SIM_METRICS_REVAL,
        status=statuses,
        max_rows=max(1000, max_concurrent_tasks * 4),
    )

    if filter_keys:
        filt = set(filter_keys)
        worklist = [(u, it) for (u, it) in worklist if u in filt]

    if not worklist:
        logger.info("📭 No claimable rows at SIM_METRICS_REVAL.")
        return

    worker_id = generate_worker_id("sim_metrics_reval")
    logger.info(
        "🚀 Starting SIM_METRICS_REVAL | %d items | worker_id=%s",
        len(worklist),
        worker_id,
    )

    tasks = await process_similarity_batch_async_fsm(
        url_iter_pairs=worklist,
        worker_id=worker_id,
        metrics_stage=PipelineStage.SIM_METRICS_REVAL,
        no_of_concurrent_workers=max_concurrent_tasks,
    )
    await asyncio.gather(*tasks)
    logger.info("✅ Finished similarity FSM run (SIM_METRICS_REVAL).")


# =============================================================================
# Pairing & compute helpers
# =============================================================================


def build_pairs_df(
    resps: pd.DataFrame,
    reqs: pd.DataFrame,
    *,
    stage: PipelineStage,  # SIM_METRICS_EVAL or SIM_METRICS_REVAL
) -> pd.DataFrame:
    """
    Build pairs DataFrame with columns:
        ["url", "responsibility_key", "requirement_key", "optimized_text", "requirement_text"]

    - EVAL  → cross-join flattened responsibilities × flattened requirements
    - REVAL → merge edited_responsibilities on requirement_key

    Guardrails:
    - REVAL requires 'requirement_key' in `resps` (edited rows are aligned).
    - EVAL does NOT require 'requirement_key' in `resps`.
    """
    if stage == PipelineStage.SIM_METRICS_REVAL:
        needed = {"url", "responsibility_key", "requirement_key", "responsibility"}
        missing = needed.difference(resps.columns)
        if missing:
            raise ValueError(f"REVAL: edited_responsibilities missing {missing}")

        pairs = (
            resps[["url", "responsibility_key", "requirement_key", "responsibility"]]
            .rename(columns={"responsibility": "optimized_text"})
            .merge(
                reqs.rename(columns={"requirement": "requirement_text"}),
                on=["url", "requirement_key"],
                how="inner",
                validate="many_to_one",
            )
            .drop_duplicates(subset=["url", "responsibility_key", "requirement_key"])
            .reset_index(drop=True)
        )

    elif stage == PipelineStage.SIM_METRICS_EVAL:
        needed = {"url", "responsibility_key", "responsibility"}
        missing = needed.difference(resps.columns)
        if missing:
            raise ValueError(f"EVAL: flattened_responsibilities missing {missing}")

        resps = resps.rename(columns={"responsibility": "optimized_text"})
        reqs = reqs.rename(columns={"requirement": "requirement_text"})

        urls = sorted(set(resps["url"]).intersection(set(reqs["url"])))
        chunks = []
        for u in urls:
            r_u = resps.loc[resps["url"] == u].assign(_k=1)
            q_u = reqs.loc[reqs["url"] == u].assign(_k=1)
            chunk = (
                r_u.merge(q_u, on="_k", how="inner")
                .drop(columns="_k")
                .reset_index(drop=True)
            )
            chunks.append(chunk)

        pairs = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    else:
        raise ValueError(f"Unsupported stage for build_pairs_df: {stage}")

    # Drop blanks and dedupe
    if not pairs.empty:
        pairs = (
            pairs[
                pairs["optimized_text"].astype(str).str.strip().ne("")
                & pairs["requirement_text"].astype(str).str.strip().ne("")
            ]
            .drop_duplicates(
                subset=["url", "responsibility_key", "requirement_key"], keep="last"
            )
            .reset_index(drop=True)
        )
    return pairs


def _normalize_similarity_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure canonical *_key columns exist in the similarity DataFrame,
    regardless of calculator output names.
    """
    ALT_RESP_ID_COLS = ("responsibility_key", "resp_id", "responsibility_id")
    ALT_REQ_ID_COLS = ("requirement_key", "req_id", "requirement_id")

    if "responsibility_key" not in df.columns:
        for c in ALT_RESP_ID_COLS:
            if c in df.columns:
                df = df.rename(columns={c: "responsibility_key"})
                break

    if "requirement_key" not in df.columns:
        for c in ALT_REQ_ID_COLS:
            if c in df.columns:
                df = df.rename(columns={c: "requirement_key"})
                break

    return df


def _validate_similarity_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate DataFrame rows against the Pydantic `SimilarityMetrics` model and return
    a normalized DataFrame ready for insertion.
    """
    adapter = TypeAdapter(list[SimilarityMetrics])
    models = adapter.validate_python(df.to_dict(orient="records"))
    return pd.DataFrame([m.model_dump() for m in models])


async def _compute_similarity_df_async(
    pairs_df: pd.DataFrame,
    *,
    max_concurrency: int = 6,
) -> pd.DataFrame:
    """
    Compute pairwise similarity for prepared (responsibility_key, requirement_key, texts) pairs.
    Uses asyncio + bounded semaphore; CPU-bound scoring dispatched to threads.
    """
    t0 = perf_counter()

    required_cols = {
        "url",
        "responsibility_key",
        "requirement_key",
        "optimized_text",
        "requirement_text",
    }
    if not required_cols.issubset(pairs_df.columns):
        missing = required_cols.difference(pairs_df.columns)
        raise ValueError(f"Pairs missing required columns: {missing}")

    if pairs_df.empty:
        return pairs_df.copy()

    # Clean & dedupe
    cleaned = (
        pairs_df[
            pairs_df["optimized_text"].astype(str).str.strip().ne("")
            & pairs_df["requirement_text"].astype(str).str.strip().ne("")
        ]
        .drop_duplicates(
            subset=["url", "responsibility_key", "requirement_key"],
            keep="last",
        )
        .reset_index(drop=True)
    )
    if cleaned.empty:
        return cleaned

    sem = asyncio.Semaphore(max_concurrency)

    async def _score_one(idx: Hashable, row: pd.Series) -> Dict[str, Any]:
        async with sem:
            try:
                return await asyncio.to_thread(
                    calculate_text_similarity_metrics,
                    row["optimized_text"],
                    row["requirement_text"],
                )
            except Exception:
                logger.exception(
                    "Scoring failed for index=%s (resp_key=%s, req_key=%s).",
                    idx,
                    row.get("responsibility_key"),
                    row.get("requirement_key"),
                )
                return {
                    "bert_score_precision": np.nan,
                    "soft_similarity": np.nan,
                    "word_movers_distance": np.nan,
                    "deberta_entailment_score": np.nan,
                    "roberta_entailment_score": np.nan,
                }

    tasks = [asyncio.create_task(_score_one(i, r)) for i, r in cleaned.iterrows()]
    results = await asyncio.gather(*tasks)

    metrics_df = pd.DataFrame(results)
    out = (
        pd.concat([cleaned, metrics_df], axis=1, copy=False)
        .sort_values(["url", "responsibility_key", "requirement_key"])
        .reset_index(drop=True)
    )

    logger.info(
        "Scored %d pairs in %.2fs (max_concurrency=%d).",
        len(out),
        perf_counter() - t0,
        max_concurrency,
    )
    return out
