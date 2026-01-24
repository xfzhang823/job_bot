# src/job_bot/utils/gap_report.py
"""
(Job) Requirement gap report utilities.

What this module does
---------------------
Given similarity metrics for a URL, compute a "best evidence" view per requirement:
- best matching responsibility_key (argmax composite_score)
- best_score
Optionally enrich with requirement text + responsibility text (if available)
and summarize gaps via thresholds.

This is intentionally a *reporting / audit* tool:
- It does NOT force edits.
- It does NOT mutate DB state.
- It helps you see which requirements are weakly supported (or intentionally absent
  like React, LangChain, degree).

Implementation notes
--------------------
This module uses the project's `load_table()` (DuckDB loader + registry) rather than
manual SQL/duckdb connections, so it stays consistent with your schema/config and
benefits from your allowed-filter safeguards.

Example
-------
>>> cfg = GapReportConfig(url=url, version="original", iteration=0)
>>> df_best, summary, text = generate_gap_report(cfg, max_rows=25)
>>> logger.info("\\n%s", text)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import json
import re
import pandas as pd

from job_bot.db_io.db_loaders import load_table
from job_bot.db_io.pipeline_enums import (
    TableName,
    Version,
)  # adjust import if different in your project
from job_bot.config.project_config import GAP_REPORTS_DIR

logger = logging.getLogger(__name__)


# ----------------------------- Data models ---------------------------------


@dataclass(frozen=True)
class GapThresholds:
    """
    Thresholds used to classify requirement coverage strength.

    strong: best_score >= strong
    medium: strong > best_score >= medium
    weak/gap: best_score < medium
    """

    strong: float = 0.45
    medium: float = 0.35


@dataclass(frozen=True)
class GapReportConfig:
    """
    Configuration for generating a requirement coverage / gap report.

    Filters are passed through to `load_table(TableName.SIMILARITY_METRICS, ...)`.
    Only filter keys allowed by your loader YAML will be applied.

    Notes:
      - Your current YAML for similarity_metrics uses:
          filters: [url, iteration, version, resp_llm_provider, resp_model_id]
        so this config uses those names to avoid silent predicate drops.

    Example Usage:
        -----------
        from pathlib import Path
        from job_bot.utils.gap_report import GapReportConfig, GapThresholds, generate_gap_report

        cfg = GapReportConfig(
                db_path=Path("/path/to/job_bot.duckdb"),
                url=url,
                version="original",
                iteration=0,                 # optional
                llm_provider=None,           # optional
                model_id=None,               # optional
                thresholds=GapThresholds(strong=0.45, medium=0.35),
        )
    """

    url: str

    # Similarity metrics filters (match your YAML filter keys)
    version: Optional[str] = Version.ORIGINAL.value
    iteration: Optional[int] = None
    resp_llm_provider: Optional[str] = None
    resp_model_id: Optional[str] = None

    thresholds: GapThresholds = GapThresholds()

    # Optional enrichment toggles (safe if tables/columns differ; enrichment is best-effort)
    include_requirement_text: bool = True
    include_responsibility_text: bool = True

    # Candidate text columns to look for (first found wins)
    req_text_cols: Sequence[str] = ("requirement_text", "requirement", "text")
    resp_text_cols: Sequence[str] = ("responsibility_text", "responsibility", "text")


# ----------------------------- Helpers -------------------------------------


def _require_columns(df: pd.DataFrame, cols: Sequence[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in {where}. "
            f"Available columns: {list(df.columns)}"
        )


def _pick_first_text_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _label_score(s: float, th: GapThresholds) -> str:
    if s >= th.strong:
        return "STRONG"
    if s >= th.medium:
        return "MEDIUM"
    return "WEAK/GAP"


# ----------------------------- Core report ----------------------------------


def compute_best_match_per_requirement(cfg: GapReportConfig) -> pd.DataFrame:
    """
    Compute best responsibility match per requirement for a given URL.

    Primary source of truth is `similarity_metrics`:
      - requirement_key
      - responsibility_key
      - composite_score
      - (optionally) requirement, responsibility

    Output columns:
      - requirement_key
      - requirement (if available)
      - best_score
      - best_responsibility_key
      - responsibility (if available)
      - coverage_label

    Compatibility aliases (only if text exists):
      - requirement_text (alias of requirement)
      - responsibility_text (alias of responsibility)
    """
    df_metrics = load_table(
        TableName.SIMILARITY_METRICS,
        url=cfg.url,
        version=cfg.version,
        iteration=cfg.iteration,
        resp_llm_provider=cfg.resp_llm_provider,
        resp_model_id=cfg.resp_model_id,
    )
    if not isinstance(df_metrics, pd.DataFrame):
        raise TypeError(
            "Expected DataFrame from load_table(TableName.SIMILARITY_METRICS). "
            f"Got: {type(df_metrics)}"
        )

    # Empty = return a stable schema
    base_cols = [
        "requirement_key",
        "requirement",
        "best_score",
        "best_responsibility_key",
        "responsibility",
        "coverage_label",
    ]
    if df_metrics.empty:
        return pd.DataFrame(columns=base_cols)

    _require_columns(
        df_metrics,
        ["requirement_key", "responsibility_key", "composite_score"],
        where="similarity_metrics",
    )

    df = df_metrics.copy()

    # Normalize keys + score
    df["requirement_key"] = df["requirement_key"].astype(str).str.strip()
    df["responsibility_key"] = df["responsibility_key"].astype(str).str.strip()
    df["composite_score"] = pd.to_numeric(
        df["composite_score"], errors="coerce"
    ).fillna(0.0)

    # Prefer carrying denormalized text directly from similarity_metrics when present
    has_req_txt = "requirement" in df.columns
    has_resp_txt = "responsibility" in df.columns

    select_cols = ["requirement_key", "responsibility_key", "composite_score"]
    if cfg.include_requirement_text and has_req_txt:
        select_cols.append("requirement")
    if cfg.include_responsibility_text and has_resp_txt:
        select_cols.append("responsibility")

    # Pick best responsibility per requirement (argmax composite_score)
    idx = df.groupby("requirement_key")["composite_score"].idxmax()
    best = df.loc[idx, select_cols].rename(
        columns={
            "responsibility_key": "best_responsibility_key",
            "composite_score": "best_score",
        }
    )

    # Ensure stable dtypes
    best["best_responsibility_key"] = best["best_responsibility_key"].astype(str)
    best["best_score"] = pd.to_numeric(best["best_score"], errors="coerce").fillna(0.0)

    # Coverage label
    best["coverage_label"] = best["best_score"].apply(
        lambda s: _label_score(float(s), cfg.thresholds)
    )

    # Worst first
    best = best.sort_values("best_score", ascending=True).reset_index(drop=True)

    # Nice column order for CSV readability (only keep columns that exist)
    preferred = [
        "requirement_key",
        "requirement",
        "best_score",
        "best_responsibility_key",
        "responsibility",
        "coverage_label",
    ]
    front = [c for c in preferred if c in best.columns]
    rest = [c for c in best.columns if c not in front]
    best = best.loc[:, front + rest]

    return best


def summarize_coverage(
    df_best: pd.DataFrame, thresholds: GapThresholds
) -> Dict[str, Any]:
    """
    Summarize coverage counts given the best-match-per-requirement DataFrame.
    """
    if df_best is None or df_best.empty:
        return {
            "total_requirements": 0,
            "strong": 0,
            "medium": 0,
            "weak_or_gap": 0,
            "avg_best_score": None,
            "median_best_score": None,
        }

    scores = pd.to_numeric(df_best["best_score"], errors="coerce").fillna(0.0)
    strong = int((scores >= thresholds.strong).sum())
    medium = int(((scores >= thresholds.medium) & (scores < thresholds.strong)).sum())
    weak = int((scores < thresholds.medium).sum())

    return {
        "total_requirements": int(len(df_best)),
        "strong": strong,
        "medium": medium,
        "weak_or_gap": weak,
        "avg_best_score": float(scores.mean()),
        "median_best_score": float(scores.median()),
    }


def render_gap_report_text(
    df_best: pd.DataFrame,
    summary: Dict[str, Any],
    *,
    max_rows: int = 20,
) -> str:
    """
    Render a short human-readable report (worst requirements first).
    """
    lines: List[str] = []
    lines.append("Requirement Gap Report")
    lines.append("-" * 80)
    lines.append(
        f"Total requirements: {summary.get('total_requirements', 0)} | "
        f"STRONG: {summary.get('strong', 0)} | "
        f"MEDIUM: {summary.get('medium', 0)} | "
        f"WEAK/GAP: {summary.get('weak_or_gap', 0)}"
    )
    if summary.get("avg_best_score") is not None:
        lines.append(
            f"Avg best score: {summary['avg_best_score']:.3f} | "
            f"Median best score: {summary['median_best_score']:.3f}"
        )
    lines.append("")

    if df_best is None or df_best.empty:
        lines.append("(no rows)")
        return "\n".join(lines)

    show = df_best.head(max_rows).copy()

    lines.append(f"Worst {min(max_rows, len(show))} requirements by best_score:")
    lines.append("-" * 80)

    for _, r in show.iterrows():
        lines.append(
            f"[{r.get('coverage_label')}] score={float(r['best_score']):.3f} "
            f"req={r['requirement_key']} -> resp={r['best_responsibility_key']}"
        )
        # Prefer new columns; fall back to legacy *_text if present
        req_txt = r.get("requirement")
        if (req_txt is None or pd.isna(req_txt)) and "requirement_text" in show.columns:
            req_txt = r.get("requirement_text")

        resp_txt = r.get("responsibility")
        if (
            resp_txt is None or pd.isna(resp_txt)
        ) and "responsibility_text" in show.columns:
            resp_txt = r.get("responsibility_text")

        if req_txt is not None and not pd.isna(req_txt):
            lines.append(f"  req_text: {str(req_txt).strip()}")
        if resp_txt is not None and not pd.isna(resp_txt):
            lines.append(f"  resp_text: {str(resp_txt).strip()}")

        lines.append("")

    return "\n".join(lines)


def generate_gap_report(
    cfg: GapReportConfig,
    *,
    max_rows: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Convenience wrapper:
      - computes best match per requirement
      - summarizes coverage
      - renders a short text report

    Returns (df_best, summary, report_text).
    """
    df_best = compute_best_match_per_requirement(cfg)
    summary = summarize_coverage(df_best, cfg.thresholds)
    text = render_gap_report_text(df_best, summary, max_rows=max_rows)
    return df_best, summary, text


def persist_gap_report(
    *,
    df_best: pd.DataFrame,
    summary: dict,
    report_text: str,
    output_dir: Path = GAP_REPORTS_DIR,  # default to local gap report directory
    url: str,
    version: str | None = None,
    iteration: int | None = None,
) -> Tuple[Path, Path, Path]:
    """
    Persist a requirement gap report to disk (CSV + JSON + TXT).

    This is an explicit, opt-in persistence step intended for auditing,
    review, and regression tracking. It does NOT modify DuckDB state.

    Files written:
      - gap_report_<slug>.csv   : per-requirement best-match table
      - gap_summary_<slug>.json : aggregate coverage statistics
      - gap_report_<slug>.txt   : human-readable report

    Parameters
    ----------
    df_best:
        DataFrame returned by compute_best_match_per_requirement().
    summary:
        Summary dict returned by summarize_coverage().
    report_text:
        Text report returned by render_gap_report_text().
    output_dir:
        Directory where report files will be written.
    url:
        Job posting URL (used for naming).
    version:
        Optional similarity metrics version
        (e.g. "original", "edited", VERSION.ORIGINAL.valuue).
    iteration:
        Optional pipeline iteration number.

    Returns
    -------
    (csv_path, json_path, txt_path):
        Paths to the written files.

    >>> Example:
        from pathlib import Path
        from job_bot.utils.gap_report import (
            generate_gap_report,
            persist_gap_report,
        )

        df_best, summary, text = generate_gap_report(cfg)

        persist_gap_report(
            df_best=df_best,
            summary=summary,
            report_text=text,
            output_dir=Path("pipeline_data/gap_reports"),
            url=cfg.url,
            version=cfg.version,
            iteration=cfg.iteration,
        )
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = _safe_slug(url)
    suffix_parts = [slug]

    if version is not None:
        suffix_parts.append(f"v_{version}")
    if iteration is not None:
        suffix_parts.append(f"iter_{iteration}")

    suffix = "__".join(suffix_parts)
    gap_report_dir = output_dir / f"gap_report__{suffix}"
    gap_report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = gap_report_dir / f"gap_report__{suffix}.csv"
    json_path = gap_report_dir / f"gap_summary__{suffix}.json"
    txt_path = gap_report_dir / f"gap_report__{suffix}.txt"

    # 1) CSV: full per-requirement table
    df_best.to_csv(csv_path, index=False)

    # 2) JSON: summary stats + metadata
    payload = {
        "url": url,
        "version": version,
        "iteration": iteration,
        "summary": summary,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # 3) TXT: human-readable report
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text.rstrip() + "\n")

    logger.info(
        "📊 Gap report persisted: %s | %s | %s",
        csv_path.name,
        json_path.name,
        txt_path.name,
    )

    return csv_path, json_path, txt_path


def _unwrap_grouped(obj: Any, *, url: str) -> Any:
    """
    load_table() may return:
      - DataFrame
      - dict[url] -> Model (when group_by_url: true)
      - Model
    This unwraps dict[url] -> Model safely.
    """
    if isinstance(obj, dict):
        if url in obj:
            return obj[url]
        # fallback: first value
        return next(iter(obj.values()), None)
    return obj


def _model_to_rows(model: Any) -> list[dict]:
    """
    Convert a Pydantic-like model into list-of-dicts rows.
    Supports:
      - model.model_dump()
      - model.dict() (pydantic v1)
    """
    if model is None:
        return []

    if hasattr(model, "model_dump"):
        data = model.model_dump()
    elif hasattr(model, "dict"):
        data = model.dict()
    else:
        return []

    # Most likely shapes in your project:
    #   Requirements: {"requirements": [ ... ]}
    #   Responsibilities: {"responsibilities": [ ... ]}
    if isinstance(data, dict):
        if isinstance(data.get("requirements"), list):
            return data["requirements"]
        if isinstance(data.get("responsibilities"), list):
            return data["responsibilities"]

    return []


def _safe_slug(s: str, max_len: int = 80) -> str:
    """
    Create a filesystem-safe slug from a string (e.g., URL).
    """
    s = s.lower()
    s = re.sub(r"https?://", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:max_len]
