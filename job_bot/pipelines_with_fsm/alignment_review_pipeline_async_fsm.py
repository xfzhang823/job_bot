"""Alignment Review pipeline (FSM-driven).

This stage exports the classic Excel crosstab for a URL, using records in
`similarity_metrics` (prefer version="edited"). It follows the same URL-only,
stage-scoped claim/finalize pattern as other FSM pipelines, and intentionally
does **not** modify process-level flags beyond the per-stage status.

Key behaviors
-------------
- Worklist is sourced via the lease-aware helpers (`get_claimable_worklist` →
    `try_claim_one`).
- Claim sets: NEW → IN_PROGRESS and stamps lease fields
    (`is_claimed`, `worker_id`, `lease_until`).
- This stage does not set or clear `decision_flag`; FSM stepping logic
    remains external.
- Export builds an XLSX "Alignment Review" sheet with the legacy layout:
  * Column 0: "Resp Key / Req Key"
  * Optional column 1: reference text ("Original Responsibility" OR
    "Edited Responsibility (best match)")
  * Then pairs of columns for each requirement:
      "{requirement text}", "{requirement text} (Score)"
  * First data row repeats `requirement_key` under each pair.
- Finalize sets status to COMPLETED (or ERROR on exceptions).

Notes
-----
- `PipelineStage` / `PipelineStatus` values are used in lowercase form.
- `task_state` is orchestrator-managed; this module only updates the
  current stage’s status for the URL(s).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict
import pandas as pd

from job_bot.config.project_config import EXCEL_DIR
from job_bot.db_io.db_loaders import load_table
from job_bot.db_io.db_readers import (
    fetch_flattened_responsibilities,
    fetch_edited_responsibilities,
)  # readers read into df instead of models

# from job_bot.db_io.db_utils import get_urls_ready_for_transition
from job_bot.db_io.db_utils import (
    generate_worker_id,
    get_claimable_worklist,
    try_claim_one,
    release_one,
    finalize_one_row_in_pipeline_control,
)
from job_bot.db_io.get_db_connection import get_db_connection
from job_bot.db_io.pipeline_enums import (
    PipelineStage,
    PipelineStatus,
    TableName,
    Version,
)
from job_bot.utils.file_name_utils import _slug, _short_hash, _extract_job_id


logger = logging.getLogger(__name__)


# ----------------------------
# Small utils
# ----------------------------
def make_alignment_review_filename(
    url: str, *, ext: str = "xlsx", maxlen: int = 80
) -> str:
    """Build a short, human-friendly export filename for the given job URL.

    Format: ``{company}-{job_title}-{jobid_or_hash}.{ext}``

    - `company`, `job_title` are fetched from the `job_urls` table (falls back to
      domain / empty if not found).
    - `jobid_or_hash` prefers a numeric job_id parsed from URL (query param or
      any 6+ digit run); otherwise uses a 6-char md5 hash of the URL.
    - No stage/iteration metadata is included.
    - Total length is capped (default 80 chars) by trimming human parts first.

    Args:
        url: Job posting URL.
        ext: File extension to append (default "xlsx").
        maxlen: Maximum filename length including the extension.

    Returns:
        A filesystem-safe filename string (no directories), including `ext`.
    """
    # fetch company & job_title from DB
    company, title = None, None
    con = get_db_connection()
    try:
        row = con.execute(
            "SELECT company, job_title FROM job_urls WHERE url = ? LIMIT 1", (url,)
        ).fetchone()
        if row:
            company, title = row[0], row[1]
    finally:
        con.close()

    # fallbacks if DB lacks values
    host = (urlparse(url).hostname or "").split(".")
    domain = host[-2] if len(host) >= 2 else (host[0] if host else "job")
    company_slug = _slug(company or domain, 24)
    title_slug = _slug(title or "", 40)  # optional; may be empty

    jobid = _extract_job_id(url)
    suffix = jobid if jobid else _short_hash(url, 6)

    # assemble base name (without extension)
    parts = [p for p in [company_slug, title_slug, suffix] if p]
    base = "_".join(parts)

    # enforce overall length (incl. dot + ext)
    max_base_len = maxlen - (len(ext) + 1)
    if len(base) > max_base_len:
        # trim title first, then company if still too long
        # try to keep suffix intact
        pieces = [company_slug, title_slug, suffix]

        # recompute until it fits
        def fit(pieces):
            return "-".join([p for p in pieces if p])

        while len(fit(pieces)) > max_base_len:
            if pieces[1] and len(pieces[1]) > 10:
                pieces[1] = pieces[1][:-1]
            elif pieces[0] and len(pieces[0]) > 8:
                pieces[0] = pieces[0][:-1]
            else:
                break
        base = fit(pieces)

    return f"{base}.{ext}"


# ----------------------------
# Crosstab builders (legacy layout)
# ----------------------------
def _disambiguate_requirement_headers(names: pd.Series, keys: pd.Series) -> list[str]:
    """Return unique requirement column headers.

    If a requirement text appears multiple times, append its `[requirement_key]`
    to disambiguate the column header.

    Args:
        names: Series of requirement texts.
        keys: Series of requirement keys aligned to `names`.

    Returns:
        A list of display names, each unique.
    """
    counts = names.value_counts(dropna=False)
    out: list[str] = []
    for name, key in zip(names, keys):
        out.append(f"{name} [{key}]" if counts.get(name, 0) > 1 else name)
    return out


def create_resp_req_crosstab_df(
    df_metrics: pd.DataFrame, score_threshold: float = 0.0
) -> pd.DataFrame:
    """Construct the legacy responsibilities × requirements crosstab.

    Input is a `similarity_metrics`-like DataFrame (prefer version="edited")
    with at least:
      `responsibility_key`, `responsibility`, `requirement_key`,
      `requirement`, and `composite_score`.

    Output columns:
      - "Resp Key / Req Key"
      - For each requirement (order by first appearance), two columns:
          "{requirement text}", "{requirement text} (Score)"
      - The first data row under each pair contains the `requirement_key`.

    Row semantics:
      - One row per `responsibility_key` (order of first appearance).
      - Each cell holds the responsibility text **only if**
        `composite_score >= score_threshold`; otherwise empty.
      - Each adjacent "(Score)" cell holds the numeric score.

    Args:
        df_metrics: Similarity metrics DataFrame for a single URL.
        score_threshold: Minimum score to show responsibility text.

    Returns:
        Crosstab DataFrame ready for Excel export.

    Raises:
        KeyError: If required columns are missing.
    """
    required_cols = {
        "responsibility_key",
        "requirement_key",
        "responsibility",
        "requirement",
        "composite_score",
    }
    missing = required_cols - set(df_metrics.columns)
    if missing:
        raise KeyError(
            f"similarity_metrics missing required columns: {sorted(missing)}"
        )

    reqs = (
        df_metrics[["requirement_key", "requirement"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # (resp_key, req_key) -> (text, score)
    pairs: dict[tuple[str, str], tuple[str, float]] = {}
    for _, r in df_metrics.iterrows():
        k = (str(r["responsibility_key"]), str(r["requirement_key"]))
        pairs[k] = (
            str(r.get("responsibility", "")),
            float(r.get("composite_score", 0.0)),
        )

    # headers
    req_col_names = _disambiguate_requirement_headers(
        reqs["requirement"], reqs["requirement_key"]
    )
    result_cols = ["Resp Key / Req Key"]
    for rc in req_col_names:
        result_cols.extend([rc, f"{rc} (Score)"])

    # header row of requirement_keys
    header_row: list[Any] = [""]
    for rk in reqs["requirement_key"]:
        header_row.extend([rk, rk])

    rows: list[list[Any]] = [header_row]

    # responsibility rows
    resp_order: list[str] = (
        df_metrics[["responsibility_key"]]
        .drop_duplicates()["responsibility_key"]
        .astype(str)
        .tolist()
    )
    for resp_key in resp_order:
        row: list[Any] = [resp_key]
        for req_key in reqs["requirement_key"].astype(str).tolist():
            resp_txt, score = pairs.get((resp_key, req_key), ("", 0.0))
            row.extend([resp_txt if score >= score_threshold else "", score])
        rows.append(row)

    return pd.DataFrame(rows, columns=result_cols)


def insert_reference_column(
    df: pd.DataFrame, reference_map: dict[str, str] | None, *, column_name: str
) -> pd.DataFrame:
    """Insert a reference text column after the first column.

    The reference column’s header is `column_name`, its first row is an empty
    string, and subsequent rows map `responsibility_key` → reference text using
    `reference_map`. If `reference_map` is None, the DataFrame is returned
    unchanged.

    Args:
        df: Crosstab produced by `create_resp_req_crosstab_df`.
        reference_map: Mapping from responsibility_key (str) to reference text.
        column_name: Display name for the reference column.

    Returns:
        A new DataFrame with an inserted reference column, or the original `df`.
    """
    if reference_map is None:
        return df
    df = df.copy()
    ref_vals: list[str] = [""]
    for i in range(1, len(df)):
        rk = str(df.iloc[i, 0])  # responsibility_key in col 0
        ref_vals.append(reference_map.get(rk, ""))
    df.insert(1, column_name, ref_vals)
    return df


def create_resp_req_crosstab_from_df(
    df_metrics: pd.DataFrame,
    *,
    score_threshold: float = 0.0,
    reference_map: dict[str, str] | None = None,
    reference_column_name: str = "Original Responsibility",
) -> pd.DataFrame:
    """
    Create the alignment-review crosstab with an optional reference column.

    This is a thin wrapper around `create_resp_req_crosstab_df` that inserts a
    human-readable reference column (original or edited responsibility text).

    Args:
        df_metrics: Similarity metrics DataFrame for a single URL.
        score_threshold: Minimum score for displaying responsibility text.
        reference_map: Optional mapping {responsibility_key → reference text}.
        reference_column_name: Header for the inserted reference column.

    Returns:
        Crosstab DataFrame including the reference column (if provided).
    """
    core = create_resp_req_crosstab_df(df_metrics, score_threshold)
    return insert_reference_column(
        core, reference_map, column_name=reference_column_name
    )


def export_alignment_review_excel(xtab: pd.DataFrame, out_path: str | Path) -> Path:
    """Write a crosstab DataFrame to an Excel file (legacy layout).

    - Sheet name: "Alignment Review"
    - Freezes header row and (optionally) the reference column.
    - Auto-sizes columns up to a reasonable width.

    Args:
        xtab: Crosstab DataFrame.
        out_path: Destination path for the .xlsx file.

    Returns:
        The resolved `Path` to the written file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xw:
        xtab.to_excel(xw, index=False, sheet_name="Alignment Review")
        ws = xw.sheets["Alignment Review"]
        has_ref_col = any(
            c.startswith(("Original Responsibility", "Edited Responsibility"))
            for c in xtab.columns
        )
        ws.freeze_panes(1, 2 if has_ref_col else 1)
        for i, col in enumerate(xtab.columns):
            try:
                width = min(60, max(12, int(xtab[col].astype(str).str.len().max()) + 2))
            except Exception:
                width = 20
            ws.set_column(i, i, width)

    return out_path


# ----------------------------
# Reference map helpers
# ----------------------------
def _edited_resps_best_match_map(
    df_metrics: pd.DataFrame, df_edited: pd.DataFrame
) -> Dict[str, str]:
    """
    Build a mapping of {responsibility_key → edited responsibility text (best match)}.

    For each responsibility_key in the similarity metrics:
      1. Identify the requirement_key with the highest composite_score.
      2. Join that (responsibility_key, requirement_key) pair to the edited
         responsibilities table to retrieve the corresponding edited text.
      3. Return a dictionary mapping each responsibility_key to its best-match
         edited responsibility text.

    Both DataFrames must include compatible key columns:
      - df_metrics: responsibility_key, requirement_key, composite_score
      - df_edited: responsibility_key, requirement_key, responsibility

    Args:
        df_metrics: Similarity metrics for a single URL (version='edited').
        df_edited: Edited responsibilities for the same URL.

    Returns:
        A dictionary mapping responsibility_key → best-match edited responsibility text.

    Raises:
        ValueError: If either input DataFrame is missing required columns.
    """
    needed_m = {"responsibility_key", "requirement_key", "composite_score"}
    missing_m = needed_m - set(df_metrics.columns)
    if missing_m:
        raise ValueError(f"similarity_metrics missing columns: {sorted(missing_m)}")

    needed_e = {"responsibility_key", "requirement_key", "responsibility"}
    missing_e = needed_e - set(df_edited.columns)
    if missing_e:
        raise ValueError(
            f"edited_responsibilities missing columns: {sorted(missing_e)}"
        )

    # Normalize join keys to strings to avoid dtype mismatches
    dfm = df_metrics.copy()
    dfe = df_edited.copy()
    dfm["responsibility_key"] = dfm["responsibility_key"].astype(str)
    dfm["requirement_key"] = dfm["requirement_key"].astype(str)
    dfe["responsibility_key"] = dfe["responsibility_key"].astype(str)
    dfe["requirement_key"] = dfe["requirement_key"].astype(str)

    # Best requirement per responsibility by highest composite_score
    best = (
        dfm.sort_values(
            ["responsibility_key", "composite_score"], ascending=[True, False]
        )
        .groupby("responsibility_key", as_index=False)
        .first()[["responsibility_key", "requirement_key"]]
    )

    ref = (
        best.merge(
            dfe[["responsibility_key", "requirement_key", "responsibility"]],
            on=["responsibility_key", "requirement_key"],
            how="left",
        )
        .set_index("responsibility_key")["responsibility"]
        .fillna("")
    )
    return ref.to_dict()


# ----------------------------
# Per-URL job (sync)
# ----------------------------
def _export_one_url_alignment_review(
    url: str,
    *,
    score_threshold: float = 0.0,
    out_dir: Path | str,
    reference_column_enum: Version = Version.ORIGINAL,  # Version.ORIGINAL or Version.EDITED
) -> Path:
    """
    Create and write the alignment-review workbook for one URL.

    Steps:
      1) Load `similarity_metrics` rows for version='edited'.
      2) Build an optional reference column:
         - ORIGINAL → flattened_responsibilities (by responsibility_key)
         - EDITED   → best-match edited text per responsibility (highest composite_score)
      3) Construct the crosstab and export to Excel.
      4) Return the output path.
    """
    # 1) Load edited metrics as a DataFrame
    df_metrics_any = load_table(TableName.SIMILARITY_METRICS, url=url, version="edited")
    if not isinstance(df_metrics_any, pd.DataFrame) or df_metrics_any.empty:
        raise ValueError("No similarity_metrics rows found for url (version='edited').")
    df_metrics = df_metrics_any
    if "version" in df_metrics.columns:
        df_metrics = df_metrics[df_metrics["version"] == Version.EDITED.value]
    if df_metrics.empty:
        raise ValueError(
            "No similarity_metrics rows remain after filtering version='edited'."
        )

    # Ensure minimal columns exist
    needed_metrics_cols = {"responsibility_key", "requirement_key", "composite_score"}
    missing = needed_metrics_cols - set(df_metrics.columns)
    if missing:
        raise ValueError(
            f"similarity_metrics missing required columns: {sorted(missing)}"
        )

    # 2) Build reference column map
    ref_name: str
    reference_map: Dict[str, str] = {}

    if reference_column_enum == Version.ORIGINAL:
        ref_name = "Original Responsibility"
        df_orig = fetch_flattened_responsibilities(url)

        if not df_orig.empty:
            df_orig = (
                df_orig.loc[:, ["responsibility_key", "responsibility"]]
                .dropna(subset=["responsibility_key", "responsibility"])
                .drop_duplicates(subset=["responsibility_key"])
            )
            df_orig["responsibility_key"] = (
                df_orig["responsibility_key"].astype(str).str.strip()
            )
            df_orig["responsibility"] = (
                df_orig["responsibility"].astype(str).str.strip()
            )

            reference_map = dict(
                zip(df_orig["responsibility_key"], df_orig["responsibility"])
            )

        logger.debug(
            "Original responsibilities sample: %s", list(reference_map.items())[:3]
        )

    elif reference_column_enum == Version.EDITED:
        ref_name = "Edited Responsibility (best match)"
        df_edited = fetch_edited_responsibilities(url)
        if df_edited.empty:
            raise ValueError(
                "Edited responsibilities not found for reference_column='edited'."
            )

        # optional filter if you keep multiple providers/models/versions
        if "version" in df_edited.columns:
            df_edited = df_edited[df_edited["version"] == Version.EDITED.value]

        reference_map = _edited_resps_best_match_map(df_metrics, df_edited)

    else:
        # If you later add Version.NONE, handle it here; for now, keep explicit
        raise ValueError("reference_column must be Version.ORIGINAL or Version.EDITED")

    # Debug sanity: overlap of responsibility keys
    resp_keys_metrics = set(df_metrics["responsibility_key"].astype(str))
    df_flat = fetch_flattened_responsibilities(url)
    resp_keys_flat = (
        set(df_flat["responsibility_key"].astype(str)) if not df_flat.empty else set()
    )
    inter = resp_keys_metrics & resp_keys_flat
    logger.debug(
        "AlignmentReview: keys metrics=%d flat=%d intersect=%d",
        len(resp_keys_metrics),
        len(resp_keys_flat),
        len(inter),
    )
    if not inter:
        logger.warning(
            "AlignmentReview: no overlap between metrics.responsibility_key and flattened keys. "
            "Edited keys may not derive from this extraction. Sample metrics keys: %s",
            sorted(list(resp_keys_metrics))[:5],
        )

    # 3) Build crosstab and export
    xtab = create_resp_req_crosstab_from_df(
        df_metrics,
        score_threshold=score_threshold,
        reference_map=reference_map,  # only affects the left reference column
        reference_column_name=ref_name,  # label for the reference column
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / make_alignment_review_filename(url, ext="xlsx")
    return export_alignment_review_excel(xtab, out_path)


# ----------------------------
# Async per-URL wrapper (URL-only)
# ----------------------------
async def _run_one_url_async(
    url: str,
    *,
    iteration: int,
    worker_id: str,
    score_threshold: float,
    out_dir: str | Path,
    reference_column_enum: Version,
    sem: asyncio.Semaphore,
) -> bool:
    """
    Run the per-URL export in a thread (blocking I/O) and finalize this (url, iteration)
    lease-aware via _finalize_one_row. Caller must have already claimed the row.
    """
    async with sem:
        try:
            out_path = await asyncio.to_thread(
                _export_one_url_alignment_review,
                url,
                score_threshold=score_threshold,
                out_dir=out_dir,
                reference_column_enum=reference_column_enum,
            )
            # Success → finalize this exact row (clears lease, sets COMPLETED)
            ok = await asyncio.to_thread(
                finalize_one_row_in_pipeline_control,
                url,
                iteration,
                worker_id=worker_id,
                ok=True,
                notes=f"Exported: {out_path}",
            )
            if not ok:
                logger.warning(
                    "Finalize returned False for url=%s iter=%s", url, iteration
                )
            return ok
        except Exception as e:
            # Failure → finalize with ERROR
            err = str(e)
            await asyncio.to_thread(
                finalize_one_row_in_pipeline_control,
                url,
                iteration,
                worker_id=worker_id,
                ok=False,
                notes=err[:2000],  # keep notes reasonable
            )
            logger.exception(
                "Export failed for url=%s iter=%s: %s", url, iteration, err
            )
            return False


# ----------------------------
# Orchestrator (URL-only)
# ----------------------------
async def run_alignment_review_pipeline_async_fsm(
    *,
    batch_limit: int | None = None,
    score_threshold: float = 0.0,
    out_dir: str | Path = EXCEL_DIR,
    reference_column_enum: Version = Version.ORIGINAL,
    max_concurrency: int = 4,
    iteration: int = 0,
    retry_errors: bool = False,
) -> int:
    """
    Alignment Review: async FSM-aware runner with human gate + lease control.

    Flow:
      1) Read claimable rows
        (stage/status + task_state='READY' + unleased/expired)
      2) Atomically try to claim each (sets status=IN_PROGRESS on success)
      3) Process concurrently
      4) Release each row with final status (COMPLETED/ERROR)

    Notes:
      - No use of get_urls_ready_for_transition (read-only helpers aren’t
        safe to drive runners)
      - This function assumes the following DB helpers exist in db_utils:
          - generate_worker_id(prefix: str) -> str
          - get_claimable_worklist(stage, status, limit=None) ->
            List[Tuple[str, int]]
          - try_claim_one(url, iteration, worker_id, lease_minutes=15) ->
            Optional[dict]
          - release_one(url, iteration, worker_id, final_status) -> bool
      - If a single URL may have multiple iterations, we always operate on
        (url, iteration) pairs.
    """
    # --- setup ---------------------------------------------------------------
    out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
    stage_enum = PipelineStage.ALIGNMENT_REVIEW
    worker_id = generate_worker_id("align")

    statuses = (
        (PipelineStatus.NEW, PipelineStatus.ERROR)
        if retry_errors
        else (PipelineStatus.NEW,)
    )

    # 1) Pull *claimable* worklist (lease + human gate). Respect batch_limit if provided.
    claimables = get_claimable_worklist(
        stage=stage_enum,
        status=statuses,
        max_rows=(
            batch_limit if (batch_limit is not None and batch_limit >= 0) else None
        ),
    )

    if not claimables:
        logger.info("📭 No claimable rows for stage=%s.", stage_enum.value)
        return 0

    # 2) Atomically claim; only process what we actually claimed
    claimed: list[tuple[str, int]] = []
    for url, iteration in claimables:
        row = try_claim_one(
            url=url, iteration=iteration, worker_id=worker_id, lease_minutes=20
        )
        if row:
            claimed.append((url, iteration))

    if not claimed:
        logger.info("🔒 Nothing claimed (contention or already leased).")
        return 0

    logger.info("🔎 Claimed %d rows for stage=%s", len(claimed), stage_enum.value)

    # 3) Process concurrently. We keep iteration for releasing later.
    sem = asyncio.Semaphore(max_concurrency)

    async def _process_one(url: str, iteration: int) -> tuple[bool, str, int]:
        """
        Run the existing per-URL worker.

        If the worker can accept iteration/worker_id,
        pass them through; otherwise it can ignore extra context.
        """
        async with sem:
            try:
                ok = await _run_one_url_async(
                    url,
                    score_threshold=score_threshold,
                    out_dir=out_dir,
                    reference_column_enum=reference_column_enum,
                    worker_id=worker_id,
                    sem=sem,
                    iteration=iteration,
                )
                return bool(ok), url, iteration
            except Exception:
                logger.exception(
                    "❌ Error processing url=%s iteration=%s", url, iteration
                )
                return False, url, iteration

    results = await asyncio.gather(
        *[_process_one(url, iteration) for (url, iteration) in claimed],
        return_exceptions=False,
    )

    # 4) Release leases with final status
    successes = 0
    for ok, url, iteration in results:
        final_status = PipelineStatus.COMPLETED if ok else PipelineStatus.ERROR
        released = release_one(
            url=url, iteration=iteration, worker_id=worker_id, final_status=final_status
        )
        if not released:
            logger.warning(
                "⚠️ Release failed (mismatched worker or lost lease?) url=%s iter=%s",
                url,
                iteration,
            )
        if ok:
            successes += 1

    logger.info(
        "✅ Alignment review finished. success=%d total=%d", successes, len(results)
    )
    return successes
