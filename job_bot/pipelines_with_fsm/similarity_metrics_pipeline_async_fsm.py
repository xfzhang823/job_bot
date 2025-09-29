"""
pipelines_with_fsm/similarity_pipeline_async_fsm.py

DB-native similarity evaluation pipeline (FSM-driven; no fallback).

Worklists:
  • EVAL:   FLATTENED_RESPONSIBILITIES / NEW   ← URLs are pulled from this stage
  • RE-EVAL: EDITED_RESPONSIBILITIES / NEW     ← URLs are pulled from this stage

DB inputs (per URL):
  • flattened_requirements    (requirement_key, requirement)
  • responsibilities:
      - EVAL   → flattened_responsibilities   (responsibility_key, responsibility)
      - RE-EVAL→ edited_responsibilities      (responsibility_key, responsibility)

Output:
  • similarity_metrics / NEW
    (rows stamped with stage = SIM_METRICS_EVAL or SIM_METRICS_REVAL)

Per-URL flow (no filesystem I/O, no LLM):
  1) FSM: mark IN_PROGRESS on the current responsibilities stage (flattened or edited).
  2) Read responsibilities (per stage) and flattened requirements from DuckDB.
  3) Compute pairwise similarity (many-to-many), categorize scores,
    add multivariate indices.
  4) (Optional) Validate rows against the SimilarityMetrics Pydantic model.
  5) Stamp metadata, align to registry schema, insert into `similarity_metrics`
    de-dup.
  6) FSM: mark COMPLETED → step(SIM_METRICS_EVAL or SIM_METRICS_REVAL) → mark NEW.

Notes:
  • No fallback path; the stage determines which responsibility table is read.
  • Vectorized compute on DataFrames for speed (no Pydantic round-trips in
    the hot path).
  • Keys are canonicalized as *_key (requirement_key, responsibility_key)
    to match the registry.
  • URL is the unit of work; the pipeline is idempotent and concurrency-safe.
"""

# --- Standard & 3rd party imports
from __future__ import annotations
import asyncio
import logging
from typing import Optional
import pandas as pd
from pydantic import TypeAdapter

# --- Project imports (adjust the package prefix to match your repo layout) ---
from job_bot.fsm.pipeline_fsm_manager import PipelineFSMManager
from job_bot.db_io.db_utils import get_urls_by_stage_and_status
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

# Reuse compute from the file ETL (computation only; no file I/O)
from job_bot.evaluation_optimization.metrics_calculator import (
    calculate_many_to_many_similarity_metrices,  # note: name as defined in your codebase
    categorize_scores_for_df,
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
# Per-URL worker
# =============================================================================


async def evaluate_similarity_for_url(
    url: str,
    *,
    fsm_manager: PipelineFSMManager,
    semaphore: asyncio.Semaphore,
    metrics_stage: PipelineStage,  # SIM_METRICS_EVAL or SIM_METRICS_REVAL
    editor_llm_provider: str | None = None,  # only if multiple edited sets exist
    editor_model_id: str | None = None,  # only if multiple edited sets exist
) -> bool:
    """
    Evaluate and persist similarity metrics for a single job posting URL.

    This stage compares responsibilities (original or edited) against
    requirements, computes pairwise similarity/entailment scores,
    validates rows, and inserts them into the `similarity_metrics` table.

    Flow
    ----
    1. Guard on FSM stage and mark IN_PROGRESS.
       • SIM_METRICS_EVAL → requires FLATTENED_RESPONSIBILITIES
       • SIM_METRICS_REVAL → requires EDITED_RESPONSIBILITIES
    2. Load responsibilities + requirements from DuckDB.
    3. Compute similarity scores with `_compute_full_similarity_df`.
    4. Normalize keys and validate rows (Pydantic optional).
    5. Stamp business metadata:
       • version = ORIGINAL (EVAL) or EDITED (REVAL)
       • resp_llm_provider/model_id (REVAL only, provenance of editor)
       • similarity_backend / nli_backend if schema owns them
       • iteration inherited via YAML (from pipeline_control)
    6. Insert into SIMILARITY_METRICS (dedup per YAML config).
    7. Mark FSM COMPLETED, step() to the metrics stage, and mark NEW.

    Args
    ----
    url : str
        Canonical job posting URL (unit of work).
    fsm_manager : PipelineFSMManager
        FSM manager for controlling per-URL pipeline state.
    semaphore : asyncio.Semaphore
        Concurrency limiter for this worker.
    metrics_stage : PipelineStage
        Target metrics stage (SIM_METRICS_EVAL or SIM_METRICS_REVAL).
    editor_llm_provider : str, optional
        Disambiguator when multiple edited responsibility sets exist
        (REVAL only).
    editor_model_id : str, optional
        Disambiguator when multiple edited responsibility sets exist
        (REVAL only).

    Returns
    -------
    bool
        True if similarity metrics were computed, inserted, and FSM advanced;
        False if skipped or failed.

    Notes
    -----
    • EVAL runs on flattened_responsibilities (version=ORIGINAL),
      no editor provenance stamped.
    • REVAL runs on edited_responsibilities (version=EDITED),
      requires a unique (llm_provider, model_id) per URL/iteration.
      If multiple sets exist, `editor_llm_provider` and
      `editor_model_id` must be passed explicitly.
    • resp_llm_provider / resp_model_id are stamped only for REVAL,
      so metrics rows can be tied back to the correct edited set.
    • similarity_backend / nli_backend values come from the compute
      function and are stamped only if the schema owns them.
    • Key columns must include url, responsibility_key,
      and requirement_key.
    """

    async with semaphore:
        fsm = fsm_manager.get_fsm(url)

        # Guard source stage based on target metrics stage
        required_src_stage = (
            PipelineStage.FLATTENED_RESPONSIBILITIES
            if metrics_stage == PipelineStage.SIM_METRICS_EVAL
            else PipelineStage.EDITED_RESPONSIBILITIES
        )
        if fsm.get_current_stage() != required_src_stage.value:
            logger.info(
                "⏩ Skipping %s; need %s, current %s",
                url,
                required_src_stage,
                fsm.state,
            )
            return False

        fsm.mark_status(PipelineStatus.IN_PROGRESS, notes="Evaluating similarity…")

        # Load inputs
        reqs = fetch_flattened_requirements(url)
        if metrics_stage == PipelineStage.SIM_METRICS_EVAL:
            resps = fetch_flattened_responsibilities(url)
            version = Version.ORIGINAL
            resp_prov: tuple[str | None, str | None] = (None, None)
        elif metrics_stage == PipelineStage.SIM_METRICS_REVAL:
            resps = fetch_edited_responsibilities(url)
            version = Version.EDITED

            # Optional disambiguation if multiple edited sets exist
            if editor_llm_provider and editor_model_id:
                resps = resps[
                    (resps["llm_provider"] == editor_llm_provider)
                    & (resps["model_id"] == editor_model_id)
                ]

            # Infer single (llm_provider, model_id) pair
            pairs = list(
                resps[["llm_provider", "model_id"]]
                .dropna()
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            if not pairs:
                fsm.mark_status(
                    PipelineStatus.ERROR, notes="No edited responsibilities provenance"
                )
                logger.error(
                    "No (llm_provider, model_id) for edited responsibilities: %s", url
                )
                return False
            if len(pairs) > 1:
                fsm.mark_status(
                    PipelineStatus.ERROR,
                    notes="Ambiguous edited sets; specify provider/model",
                )
                logger.error(
                    "Multiple edited sets for %s; provide editor_llm_provider/model_id. Found: %s",
                    url,
                    pairs,
                )
                return False
            resp_prov = pairs[0]
        else:
            logger.error("❌ Unsupported metrics_stage: %s", metrics_stage)
            return False

        if reqs.empty or resps.empty:
            logger.warning(
                "%s: missing reqs/resps (reqs=%d, resps=%d)", url, len(reqs), len(resps)
            )
            fsm.mark_status(PipelineStatus.ERROR, notes="Missing reqs or resps")
            return False

        try:
            # Compute similarity → (df[, meta])
            result = await _compute_full_similarity_df(resps, reqs)
            if isinstance(result, tuple):
                df, meta = result
                similarity_backend = meta.get("similarity_backend")
                nli_backend = meta.get("nli_backend")
            else:
                df = result
                similarity_backend = None
                nli_backend = None

            if df.empty:
                fsm.mark_status(PipelineStatus.ERROR, notes="No similarity rows")
                return False

            # Normalize keys, ensure URL, validate to schema model (adds optional columns with defaults)
            df = _normalize_similarity_keys(df)
            if "url" not in df.columns:
                df.insert(0, "url", url)

            if VALIDATE_WITH_PYDANTIC:
                df = _validate_similarity_rows(df)

            # Ensure required columns present post-validation
            assert {"url", "responsibility_key", "requirement_key"}.issubset(
                df.columns
            ), "Missing required columns after normalization/validation"

            # Business fields owned by this stage
            df["version"] = version

            # Stamp compute backends if the table owns them
            schema = DUCKDB_SCHEMA_REGISTRY[TableName.SIMILARITY_METRICS]
            owned = set(getattr(schema, "metadata_fields", [])) | set(
                schema.column_order
            )
            if "similarity_backend" in owned and "similarity_backend" not in df.columns:
                df["similarity_backend"] = similarity_backend
            if "nli_backend" in owned and "nli_backend" not in df.columns:
                df["nli_backend"] = nli_backend

            # One insert:
            # - iteration is inherited via YAML (pipeline_control) using url=
            # - for REVAL, pass editor provenance via stamp:param per YAML
            if version == Version.EDITED:
                # Ensure schema columns exist for REVAL provenance if not validating with Pydantic
                if not VALIDATE_WITH_PYDANTIC:
                    for col in ("resp_llm_provider", "resp_model_id"):
                        if col not in df.columns:
                            df[col] = None

                insert_df_with_config(
                    df,
                    TableName.SIMILARITY_METRICS,
                    url=url,
                    resp_llm_provider=resp_prov[0],  # YAML: stamp:param
                    resp_model_id=resp_prov[1],  # YAML: stamp:param
                )
            else:
                insert_df_with_config(
                    df,
                    TableName.SIMILARITY_METRICS,
                    url=url,
                )

        except Exception:
            logger.exception("❌ Compute/insert failed for %s", url)
            fsm.mark_status(PipelineStatus.ERROR, notes="Compute or DB insert failed")
            return False

        # Advance FSM
        try:
            fsm.mark_status(PipelineStatus.COMPLETED, notes="Similarity saved to DB")
            fsm.step(metrics_stage)
            fsm.mark_status(PipelineStatus.NEW, notes="Metrics ready")
            logger.info("✅ Similarity complete: %s", url)
            return True
        except Exception:
            logger.exception("❌ FSM transition failed for %s", url)
            fsm.mark_status(PipelineStatus.ERROR, notes="FSM transition failed")
            return False


# =============================================================================
# Entrypoints: run eval pipe; run re-eval pipe
# =============================================================================
# Public entrypoints (no fallback; strict two-stage similarity)


async def run_similarity_metrics_eval_async_fsm(
    *,
    max_concurrent: int = 4,
    filter_urls: Optional[list[str]] = None,
) -> None:
    """
    Run the similarity EVAL pipeline.

    - Worklist: FLATTENED_RESPONSIBILITIES / NEW.
    - Reads flattened responsibilities + flattened requirements.
    - Writes to `similarity_metrics` with stage = SIM_METRICS_EVAL.
    - Advances FSM to SIM_METRICS_EVAL and marks NEW for downstream stages.
    """
    await _run_similarity_metrics(
        max_concurrent=max_concurrent,
        metrics_stage=PipelineStage.SIM_METRICS_EVAL,
        filter_urls=filter_urls,
    )


async def run_similarity_metrics_reval_async_fsm(
    *,
    max_concurrent: int = 4,
    filter_urls: Optional[list[str]] = None,
) -> None:
    """
    Run the similarity RE-EVAL pipeline.

    - Worklist: EDITED_RESPONSIBILITIES / NEW (or pass `urls=` explicitly).
    - Reads edited responsibilities + flattened requirements.
    - Writes to `similarity_metrics` with stage = SIM_METRICS_REVAL.
    - Advances FSM to SIM_METRICS_REVAL and marks NEW for downstream stages.
    """
    await _run_similarity_metrics(
        max_concurrent=max_concurrent,
        metrics_stage=PipelineStage.SIM_METRICS_REVAL,
        filter_urls=filter_urls,
    )


# =============================================================================
# Internal helpers/utils
# =============================================================================
def _normalize_similarity_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure canonical *_key columns exist in the similarity DataFrame,
    regardless of calculator output names.

    - Maps resp_id / responsibility_id → responsibility_key
    - Maps req_id / requirement_id → requirement_key

    Args:
        df (pd.DataFrame): DataFrame with similarity results.

    Returns:
        pd.DataFrame: DataFrame with guaranteed `responsibility_key` and
        `requirement_key` columns.
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

    This enforces schema alignment and types based on your Pyd models.
    """
    adapter = TypeAdapter(list[SimilarityMetrics])
    models = adapter.validate_python(df.to_dict(orient="records"))
    return pd.DataFrame([m.model_dump() for m in models])


# =============================================================================
# Core internal compute (score → categorize → multivariate indices)
# =============================================================================


async def _compute_full_similarity_df(
    resps: pd.DataFrame,
    reqs: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """
    Compute the full similarity result and return both the dataframe and metadata.

    Steps:
      - Many-to-many scoring
      - Categorization
      - Multivariate indices (composite / PCA)

    Input:
        resps: DataFrame with ['responsibility_key', 'responsibility']
        reqs:  DataFrame with ['requirement_key', 'requirement']

    Returns:
        tuple:
            • DataFrame: pairwise rows with score columns, ready for validation/insert
            • dict: metadata with compute provenance, e.g.:
                {
                    "similarity_backend": "<embedding model name>",
                    "nli_backend": "<NLI model name>",
                }
    """
    # Build {id: text} dicts expected by your calculators
    resp_map = dict(zip(resps["responsibility_key"], resps["responsibility"]))
    req_map = dict(zip(reqs["requirement_key"], reqs["requirement"]))

    # CPU-bound work moved to a thread to keep the event loop snappy
    base_scores = await asyncio.to_thread(
        calculate_many_to_many_similarity_metrices,
        responsibilities=resp_map,
        requirements=req_map,
    )
    with_categories = await asyncio.to_thread(categorize_scores_for_df, base_scores)
    with_composites = await asyncio.to_thread(add_multivariate_indices, with_categories)

    # Metadata: adapt to whatever your calculators expose or constants you configure
    # * They are placeholders for now!
    meta = {
        "similarity_backend": getattr(
            calculate_many_to_many_similarity_metrices, "BACKEND", None
        ),
        "nli_backend": getattr(categorize_scores_for_df, "BACKEND", None),
    }

    return with_composites, meta


# =============================================================================
# Internal orchestrator
# =============================================================================
async def _run_similarity_metrics(
    *,
    max_concurrent: int,
    metrics_stage: PipelineStage,
    filter_urls: list[str] | None = None,
) -> None:
    """
    Orchestrate the similarity run (eval or re-eval) over a list of URLs.

    - If `urls` is not provided, pulls the worklist from pipeline_control at:
        • FLATTENED_RESPONSIBILITIES / NEW for EVAL
        • EDITED_RESPONSIBILITIES    / NEW for RE-EVAL
    - Spawns bounded-concurrency workers with the specified `metrics_stage`.

    Args:
        max_concurrent: Max concurrent URL workers.
        metrics_stage: SIM_METRICS_EVAL or SIM_METRICS_REVAL.
        urls: Optional explicit list of URLs to process (bypasses worklist query).
    """
    # Pick btw eval and reval stage
    if metrics_stage == PipelineStage.SIM_METRICS_EVAL:
        src_stage = PipelineStage.FLATTENED_RESPONSIBILITIES
    else:
        src_stage = PipelineStage.EDITED_RESPONSIBILITIES

    urls = get_urls_by_stage_and_status(stage=src_stage, status=PipelineStatus.NEW)

    if filter_urls:
        urls = [u for u in urls if u in filter_urls]

    if not urls:
        logger.info("📭 No URLs at worklist stage for %s.", metrics_stage)
        return

    sem = asyncio.Semaphore(max_concurrent)
    fsm_manager = PipelineFSMManager()
    tasks = [
        asyncio.create_task(
            evaluate_similarity_for_url(
                u,
                fsm_manager=fsm_manager,
                semaphore=sem,
                metrics_stage=metrics_stage,
            )
        )
        for u in urls
    ]
    await asyncio.gather(*tasks)
    logger.info("✅ Finished similarity FSM run (%s).", metrics_stage)


# async def evaluate_similarity_for_url(
#     url: str,
#     *,
#     fsm_manager: PipelineFSMManager,
#     semaphore: asyncio.Semaphore,
#     metrics_stage: PipelineStage,  # SIM_METRICS_EVAL or SIM_METRICS_REVAL
#     iteration: int | None = None,
#     # Optional disambiguators used ONLY if multiple edited sets exist:
#     # Do not expect to use these most of the time!
#     editor_llm_provider: str | None = None,
#     editor_model_id: str | None = None,
# ) -> bool:
#     """

#     """

#     async with semaphore:
#         fsm = fsm_manager.get_fsm(url)

#         # Guard: ensure we're on the right source stage
#         required_src_stage = (
#             PipelineStage.FLATTENED_RESPONSIBILITIES
#             if metrics_stage == PipelineStage.SIM_METRICS_EVAL
#             else PipelineStage.EDITED_RESPONSIBILITIES
#         )
#         if fsm.get_current_stage() != required_src_stage:
#             logger.info(
#                 "⏩ Skipping %s; need %s, current %s",
#                 url,
#                 required_src_stage,
#                 fsm.state,
#             )
#             return False

#         fsm.mark_status(PipelineStatus.IN_PROGRESS, notes="Evaluating similarity…")

#         # Load inputs
#         reqs = fetch_flattened_requirements(url)
#         if metrics_stage == PipelineStage.SIM_METRICS_EVAL:
#             resps = fetch_flattened_responsibilities(url)
#             version = Version.ORIGINAL
#             resp_prov = (None, None)  # (llm_provider, model_id)
#         elif metrics_stage == PipelineStage.SIM_METRICS_REVAL:
#             resps = fetch_edited_responsibilities(url)
#             version = Version.EDITED

#             # Optional disambiguation if multiple edited sets exist
#             if editor_llm_provider and editor_model_id:
#                 resps = resps[
#                     (resps["llm_provider"] == editor_llm_provider)
#                     & (resps["model_id"] == editor_model_id)
#                 ]

#             # Infer a single (llm_provider, model_id) pair from what's left
#             pairs = (
#                 resps[["llm_provider", "model_id"]]
#                 .dropna()
#                 .drop_duplicates()
#                 .itertuples(index=False, name=None)
#             )
#             pairs = list(pairs)
#             if not pairs:
#                 fsm.mark_status(
#                     PipelineStatus.ERROR, notes="No edited responsibilities provenance"
#                 )
#                 logger.error(
#                     "No (llm_provider, model_id) for edited responsibilities: %s", url
#                 )
#                 return False
#             if len(pairs) > 1:
#                 fsm.mark_status(
#                     PipelineStatus.ERROR,
#                     notes="Ambiguous edited sets; specify provider/model",
#                 )
#                 logger.error(
#                     "Multiple edited sets for %s; provide editor_llm_provider/model_id. Found: %s",
#                     url,
#                     pairs,
#                 )
#                 return False
#             resp_prov = pairs[0]
#         else:
#             logger.error("❌ Unsupported metrics_stage: %s", metrics_stage)
#             return False

#         if reqs.empty or resps.empty:
#             logger.warning(
#                 "%s: missing reqs/resps (reqs=%d, resps=%d)", url, len(reqs), len(resps)
#             )
#             fsm.mark_status(PipelineStatus.ERROR, notes="Missing reqs or resps")
#             return False

#         try:
#             # Compute returns (df, meta) where meta has backends
#             result = await _compute_full_similarity_df(resps, reqs)
#             if isinstance(result, tuple):
#                 df, meta = result
#                 similarity_backend = meta.get("similarity_backend")
#                 nli_backend = meta.get("nli_backend")
#             else:
#                 df = result
#                 similarity_backend = None
#                 nli_backend = None

#             if df.empty:
#                 fsm.mark_status(PipelineStatus.ERROR, notes="No similarity rows")
#                 return False

#             df = _normalize_similarity_keys(df)
#             if "url" not in df.columns:
#                 df.insert(0, "url", url)
#             assert {"responsibility_key", "requirement_key"}.issubset(
#                 df.columns
#             ), "Missing *_key columns after normalization"

#             if VALIDATE_WITH_PYDANTIC:
#                 df = _validate_similarity_rows(df)

#             insert_df_with_config(
#                 df,
#                 TableName.SIMILARITY_METRICS,
#                 url=url,
#                 # for REVAL only, either set columns in df or pass as params:
#                 # resp_llm_provider=resp_prov[0],
#                 # resp_model_id=resp_prov[1],
#             )

#             # Option A: stamp editor provenance ONLY for edited runs
#             if version == Version.EDITED:
#                 df["resp_llm_provider"], df["resp_model_id"] = resp_prov
#             else:
#                 # Keep columns present/NULL if your schema includes them
#                 if "resp_llm_provider" not in df.columns:
#                     df["resp_llm_provider"] = None
#                 if "resp_model_id" not in df.columns:
#                     df["resp_model_id"] = None

#             # Stamp compute backends if the schema owns them
#             schema = DUCKDB_SCHEMA_REGISTRY[TableName.SIMILARITY_METRICS]
#             owned = set(getattr(schema, "metadata_fields", [])) | set(
#                 schema.column_order
#             )
#             if "similarity_backend" in owned and "similarity_backend" not in df.columns:
#                 df["similarity_backend"] = similarity_backend
#             if "nli_backend" in owned and "nli_backend" not in df.columns:
#                 df["nli_backend"] = nli_backend

#             insert_df_with_config(df, TableName.SIMILARITY_METRICS)

#         except Exception:
#             logger.exception("❌ Compute/insert failed for %s", url)
#             fsm.mark_status(PipelineStatus.ERROR, notes="Compute or DB insert failed")
#             return False

#         try:
#             fsm.mark_status(PipelineStatus.COMPLETED, notes="Similarity saved to DB")
#             fsm.step(metrics_stage)
#             fsm.mark_status(PipelineStatus.NEW, notes="Metrics ready")
#             logger.info("✅ Similarity complete: %s", url)
#             return True
#         except Exception:
#             logger.exception("❌ FSM transition failed for %s", url)
#             fsm.mark_status(PipelineStatus.ERROR, notes="FSM transition failed")
#             return False
