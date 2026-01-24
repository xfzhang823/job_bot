# src/job_bot/pipelines/resume_editing/pair_filtering.py
"""
Pair filtering utilities for the resume editing pipeline.

Goal
----
Given a Similarity Metrics table (typically the *original* metrics pass) for a URL,
select which (responsibility_key, requirement_key) pairs are eligible for LLM editing.

This module is intentionally *pipeline-callable* and *pure*:
- It does NOT load from DuckDB.
- It does NOT mutate your resps_dict/reqs_dict.
- It only computes pair eligibility (and optional per-responsibility grouping).

Typical usage in `edit_and_persist_responsibilities_for_url`:
-----------------------------------------------------------
    metrics_df = ...  # load similarity_metrics rows for url
    eligible_pairs = pick_pairs_to_edit(metrics_df, min_score=0.35, top_k=3, min_keep_per_resp=1)
    eligible_map = group_pairs_by_responsibility(eligible_pairs)

    for resp_key, resp_text in resps_dict.items():
        req_keys = eligible_map.get(resp_key, set())
        if not req_keys:
            continue  # skip LLM for this resp
        reqs_sub = subset_requirements(reqs_dict, req_keys)
        ... call modify_* with {resp_key: resp_text} and reqs_sub ...

Design notes
------------
- Filtering is *pair-level*, not dictionary-level.
- You can keep output-table completeness by separately “filling with originals”
  when materializing rows for insertion (that belongs in a separate module).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    Mapping,
    Sequence,
    Set,
    Tuple,
)

import logging

import pandas as pd

logger = logging.getLogger(__name__)

Pair = Tuple[str, str]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairFilterConfig:
    """
    Configuration for selecting (responsibility, requirement) pairs to send to the
    resume-editing LLM.

    This configuration governs *pair-level eligibility* only. It does NOT mutate
    responsibility or requirement dictionaries and does NOT guarantee full coverage
    of all requirements. Its purpose is to:
      - focus each responsibility on a small number of strong matches,
      - prevent semantic overreach during editing,
      - and optionally ensure a minimal editing opportunity per responsibility
        without forcing obviously bad matches.

    Attributes
    ----------
    min_score:
        Primary relevance threshold. A pair must have composite_score >= min_score
        to be eligible under normal circumstances.

    top_k:
        Per-responsibility cap on how many requirement matches are allowed to
        influence editing. For each responsibility, only the top_k highest-scoring
        requirements (ranked by composite_score descending) are considered.

    min_keep_per_resp:
        Safety mechanism. If > 0, allows keeping up to this many top-ranked pairs
        per responsibility even if they fall below min_score.

        This is intended to avoid cases where a responsibility ends up with no
        editable pairs at all, while still respecting overall edit quality.

    min_keep_score_floor:
        Hard floor for the safety keep. If set, the safety keep (min_keep_per_resp)
        is only applied when the *best* composite_score for a responsibility meets
        or exceeds this value.

        If the best available match is below this floor, no pairs are kept for that
        responsibility and the original text should be preserved.

        Set to None to disable this floor and always apply the safety keep.

    score_col:
        Column name for the composite similarity score
        (default: "composite_score").

    resp_key_col:
        Column name for the responsibility key
        (default: "responsibility_key").

    req_key_col:
        Column name for the requirement key
        (default: "requirement_key").
    """

    min_score: float = 0.45
    top_k: int = 2

    # Safety keep
    min_keep_per_resp: int = 0

    # Do not force safety keep if the best score is below this floor.
    # Set to None to disable the floor behavior.
    min_keep_score_floor: float | None = 0.25

    score_col: str = "composite_score"
    resp_key_col: str = "responsibility_key"
    req_key_col: str = "requirement_key"


# ---------------------------------------------------------------------------
# Core selection logic
# ---------------------------------------------------------------------------


def pick_pairs_to_edit(
    df_metrics: pd.DataFrame,
    *,
    min_score: float,
    top_k: int,
    min_keep_per_resp: int = 0,
    min_keep_score_floor: float | None = None,
    resp_key_col: str = "responsibility_key",
    req_key_col: str = "requirement_key",
    score_col: str = "composite_score",
) -> Set[Pair]:
    """
    Select eligible (responsibility_key, requirement_key) pairs for LLM editing.

    This function performs *pair-level gating* based on similarity metrics. It is
    responsibility-centric and intentionally conservative: each responsibility is
    allowed to consider only a small number of its strongest requirement matches.

    The algorithm:
      1. Groups rows by responsibility_key.
      2. Ranks requirements per responsibility by composite_score (descending).
      3. Keeps pairs that satisfy BOTH:
         - rank <= top_k
         - composite_score >= min_score
      4. Optionally applies a safety fallback to retain a minimal number of pairs
         per responsibility, without forcing obviously weak matches.

    Safety fallback behavior:
      - If min_keep_per_resp > 0 and min_keep_score_floor is None:
          Always keep up to min_keep_per_resp top-ranked pairs per responsibility,
          regardless of score.
      - If min_keep_per_resp > 0 and min_keep_score_floor is set:
          Keep up to min_keep_per_resp top-ranked pairs ONLY IF the best
          composite_score for that responsibility is >= min_keep_score_floor.
          Otherwise, no pairs are kept for that responsibility.

    Important notes:
      - This function does NOT guarantee that every requirement is matched or edited.
      - It does NOT mutate responsibility or requirement dictionaries.
      - It does NOT ensure global requirement coverage.
      - Responsibilities with no eligible pairs should retain their original text.

    Parameters
    ----------
    df_metrics:
        Similarity metrics rows for a single URL (or a consistent cohort of rows).
    min_score:
        Primary relevance threshold. Pairs with composite_score below this value
        are excluded unless retained by the safety fallback.
    top_k:
        Per-responsibility cap on the number of requirement matches that may
        influence editing.
    min_keep_per_resp:
        Safety keep count. If > 0, allows retaining up to this many top-ranked
        pairs per responsibility even when they fall below min_score.
    min_keep_score_floor:
        Hard floor for the safety keep. If set, the safety keep is applied only
        when the best composite_score for a responsibility meets or exceeds this
        value. Set to None to disable the floor.
    resp_key_col, req_key_col, score_col:
        Column names for responsibility key, requirement key, and similarity score.

    Returns
    -------
    Set[Tuple[str, str]]:
        Set of (responsibility_key, requirement_key) pairs eligible for editing.

    Raises
    ------
    ValueError:
        If required columns are missing, top_k <= 0, or min_keep_per_resp is invalid.
    """

    if df_metrics is None or len(df_metrics) == 0:
        return set()

    if top_k <= 0:
        raise ValueError(f"top_k must be > 0. Got: {top_k}")
    if min_keep_per_resp < 0:
        raise ValueError(f"min_keep_per_resp must be >= 0. Got: {min_keep_per_resp}")
    if min_keep_per_resp > top_k:
        # Not strictly invalid, but surprising. Fail fast to prevent silent confusion.
        raise ValueError(
            f"min_keep_per_resp ({min_keep_per_resp}) cannot exceed top_k ({top_k})."
        )

    _require_columns(df_metrics, [resp_key_col, req_key_col, score_col])
    df = _coerce_keys_and_scores(
        df_metrics,
        resp_key_col=resp_key_col,
        req_key_col=req_key_col,
        score_col=score_col,
    )

    if df.empty:
        return set()

    df = df.sort_values([resp_key_col, score_col], ascending=[True, False])
    df["_rank"] = df.groupby(resp_key_col).cumcount() + 1

    keep_mask = (df["_rank"] <= top_k) & (df[score_col] >= float(min_score))

    if min_keep_per_resp > 0:
        if min_keep_score_floor is None:
            # old behavior: always keep top N, regardless of score
            fallback_mask = df["_rank"] <= min_keep_per_resp
        else:
            # NEW behavior: keep top N only for responsibilities whose best score clears the floor
            best_score = df.groupby(resp_key_col)[score_col].transform("max")
            fallback_mask = (df["_rank"] <= min_keep_per_resp) & (
                best_score >= float(min_keep_score_floor)
            )

        keep_mask = keep_mask | fallback_mask

    keep = df.loc[keep_mask, [resp_key_col, req_key_col]]
    return set(zip(keep[resp_key_col].tolist(), keep[req_key_col].tolist()))


def pick_pairs_to_edit_with_config(
    df_metrics: pd.DataFrame, cfg: PairFilterConfig
) -> Set[Pair]:
    """
    Convenience wrapper around pick_pairs_to_edit using PairFilterConfig.
    """
    return pick_pairs_to_edit(
        df_metrics,
        min_score=cfg.min_score,
        top_k=cfg.top_k,
        min_keep_per_resp=cfg.min_keep_per_resp,
        min_keep_score_floor=cfg.min_keep_score_floor,
        resp_key_col=cfg.resp_key_col,
        req_key_col=cfg.req_key_col,
        score_col=cfg.score_col,
    )


# ---------------------------------------------------------------------------
# Grouping helpers (pipeline-friendly)
# ---------------------------------------------------------------------------


def group_pairs_by_responsibility(pairs: Iterable[Pair]) -> Dict[str, Set[str]]:
    """
    Convert a pair set into a mapping: resp_key -> set(req_key).

    This is useful for looping per responsibility and building a per-resp requirements
    subset to send to the LLM editor.

    Parameters
    ----------
    pairs:
        Iterable of (resp_key, req_key) tuples.

    Returns
    -------
    Dict[str, Set[str]]:
        Mapping of resp_key to requirement keys.
    """
    out: Dict[str, Set[str]] = {}
    for resp_key, req_key in pairs:
        if resp_key is None or req_key is None:
            continue
        r = str(resp_key)
        q = str(req_key)
        if not r or not q:
            continue
        out.setdefault(r, set()).add(q)
    return out


def subset_requirements(
    reqs_dict: Mapping[str, str],
    keep_req_keys: Iterable[str],
    *,
    strict: bool = True,
) -> Dict[str, str]:
    """
    Build a requirements dict containing only keep_req_keys.

    Parameters
    ----------
    reqs_dict:
        Full requirement dictionary: req_key -> requirement text.
    keep_req_keys:
        Keys to include.
    strict:
        If True, missing keys raise KeyError.
        If False, missing keys are ignored (and logged).

    Returns
    -------
    Dict[str, str]:
        Subset dict suitable for calling modify_*.

    Raises
    ------
    KeyError:
        If strict=True and a requested key is missing.
    """
    out: Dict[str, str] = {}
    missing: list[str] = []

    for k in keep_req_keys:
        kk = str(k)
        if kk in reqs_dict:
            out[kk] = reqs_dict[kk]
        else:
            missing.append(kk)

    if missing:
        msg = f"subset_requirements: missing {len(missing)} requirement keys (example: {missing[:5]})."
        if strict:
            raise KeyError(msg)
        logger.warning(msg)

    return out


def eligible_pairs_stats(pairs: Iterable[Pair]) -> Dict[str, int]:
    """
    Lightweight stats for logging / debugging.

    Returns dict with:
      - num_pairs
      - num_responsibilities
      - num_requirements

    Note: requirement count is unique across all pairs.
    """
    pairs_list = list(pairs)
    resps = {r for r, _ in pairs_list}
    reqs = {q for _, q in pairs_list}
    return {
        "num_pairs": len(pairs_list),
        "num_responsibilities": len(resps),
        "num_requirements": len(reqs),
    }


# ---------------------------------------------------------------------------
# Validation / Coercion
# ---------------------------------------------------------------------------


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Similarity metrics DataFrame missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _coerce_keys_and_scores(
    df: pd.DataFrame,
    *,
    resp_key_col: str,
    req_key_col: str,
    score_col: str,
) -> pd.DataFrame:
    """
    Return a sanitized copy with:
    - keys coerced to string
    - score coerced to float (NaN -> 0.0)
    - rows with empty keys dropped
    """
    out = df.copy()

    # Coerce keys
    out[resp_key_col] = out[resp_key_col].astype(str)
    out[req_key_col] = out[req_key_col].astype(str)

    # Normalize "nan"/None-like string keys to empty and drop
    out[resp_key_col] = out[resp_key_col].fillna("").astype(str)
    out[req_key_col] = out[req_key_col].fillna("").astype(str)
    out = out[(out[resp_key_col].str.len() > 0) & (out[req_key_col].str.len() > 0)]

    # Coerce score
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0)

    return out
