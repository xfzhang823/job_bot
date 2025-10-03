"""
duckdb_table_models.py

Refactored with PURE mixins to avoid MRO issues:
- Mixins (TimestampedMixin, LLMStampedMixin) DO NOT inherit from BaseModel.
- AppBaseModel is the ONLY base that subclasses pydantic.BaseModel.
- Always list mixins BEFORE AppBaseModel in class bases.

Design principles:
- Each table has only the fields it truly needs.
- `pipeline_control` is FSM-only (no history; no provider).
- `job_urls` is a seed list (no iteration).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

# User defined
from job_bot.db_io.pipeline_enums import (
    PipelineStage,
    PipelineStatus,
    PipelineProcessStatus,
    Version,
    LLMProvider,
)

# ──────────────────────────────────────────────────────────────────────────────
# Base & PURE mixins (no BaseModel inheritance in mixins)
# ──────────────────────────────────────────────────────────────────────────────


class AppBaseModel(BaseModel):
    """Project-wide defaults for parsing/serialization behavior."""

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
        use_enum_values=False,  # keep Enums as Enum objects internally
    )


class TimestampedMixin(BaseModel):
    """Opt-in audit timestamps. Use ONLY where you truly need them."""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class LLMStampedMixin:
    """Opt-in LLM metadata for artifacts produced by an LLM."""

    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI)
    model_id: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Control-plane (FSM snapshot ONLY — no history here)
# ──────────────────────────────────────────────────────────────────────────────


class PipelineState(AppBaseModel, TimestampedMixin):
    """
    Represents a single **row** in the `pipeline_control` table.

    - DuckDB table: `pipeline_control` (collection of all states).
    - Model name: `PipelineState` (one state per URL/iteration).
    - FSM logic ensures at most one active row per (url, iteration).

    Fields:
        url (HttpUrl | str):
            Job posting URL (primary key component).
        iteration (int):
            Iteration index for reruns (default = 0).
        stage (PipelineStage):
            Current pipeline stage in the FSM.
        status (PipelineStatus):
            Stage-local FSM status (e.g., NEW, IN_PROGRESS, DONE, ERROR).
        process_status (PipelineProcessStatus):
            High-level lifecycle status for the entire pipeline run
            (NEW, RUNNING, COMPLETED, SKIPPED).
        decision_flag: int | None = None
            1 = go, 0 = no-go
        transition_flag: int = 0
            0=pending (have not applied), 1 = applied
        notes (str | None):
            Optional free-text notes for FSM/debugging.
        source_file (str | None):
            Provenance marker for seeding/debug.
        version (Version | None):
            Optional editorial version marker.

    ✅ Naming convention:
        • Table = `pipeline_control` (plural/collection)
        • Model = `PipelineState` (singular, one record)
        • Intentional difference to emphasize row vs. table
    """

    url: Union[HttpUrl, str]
    iteration: int = 0
    stage: PipelineStage
    status: PipelineStatus = Field(default=PipelineStatus.NEW)
    process_status: PipelineProcessStatus = Field(default=PipelineProcessStatus.NEW)
    decision_flag: int | None = None  # 1=go, 0=no-go, NULL=undecided
    transition_flag: int = 0  # 0=pending, 1=final/applied
    notes: Optional[str] = None
    source_file: Optional[str] = None
    version: Optional[Version] = None


# ──────────────────────────────────────────────────────────────────────────────
# Input table (seed list; no iteration)
# ──────────────────────────────────────────────────────────────────────────────


class JobUrlsRow(AppBaseModel, TimestampedMixin):
    """
    Registry of job posting URLs. Iteration does NOT belong here.
    """

    url: Union[HttpUrl, str]
    # Optional descriptive metadata (kept minimal)
    company: Optional[str] = None
    job_title: Optional[str] = None
    source_file: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Stage/data bases
# ──────────────────────────────────────────────────────────────────────────────


class BaseStageRow(AppBaseModel):
    """
    Minimal shared fields for per-URL, per-iteration stage tables.
    Intentionally excludes timestamps and LLM metadata.
    """

    url: Union[HttpUrl, str]
    iteration: int = 0
    # Optional provenance/versioning knobs (only if you actually use them)
    source_file: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Stage/data tables
# ──────────────────────────────────────────────────────────────────────────────


class JobPostingsRow(BaseStageRow, LLMStampedMixin, TimestampedMixin):
    """
    Raw scraped job posting content in flat format.
    Timestamps are useful here for freshness/debugging.
    """

    job_title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    salary_info: Optional[str] = None
    posted_date: Optional[str] = None
    content: Optional[str] = Field(
        default=None,
        description="Full job description text or serialized JSON content.",
    )


class ExtractedRequirementsRow(BaseStageRow, LLMStampedMixin):
    """
    Structured job requirements extracted from a job posting.
    """

    requirement_key: str
    requirement: str
    # Optional grouping you actually use
    requirement_category: Optional[str] = None
    requirement_category_key: Optional[int] = None  # per-category order


class FlattenedRequirementsRow(BaseStageRow):
    """
    Flattened (normalized) job requirements used for downstream matching.
    """

    requirement_key: str
    requirement: str


class FlattenedResponsibilitiesRow(BaseStageRow):
    """
    Flattened canonical resume responsibilities (provider-agnostic).
    """

    responsibility_key: str
    responsibility: str

    # Optional resume structuring fields (only if used)
    # section: Optional[str] = None
    # role_title: Optional[str] = None
    # start_year: Optional[int] = None
    # end_year: Optional[int] = None


class PrunedResponsibilitiesRow(BaseStageRow):
    """
    Responsibilities pruned via rules/review (still provider-agnostic).
    """

    responsibility_key: str
    responsibility: str
    pruned_by: str = Field(
        ..., description="Who/what performed pruning (e.g., 'llm', 'xfz', 'heuristic')."
    )


class EditedResponsibilitiesRow(BaseStageRow, LLMStampedMixin, TimestampedMixin):
    """
    LLM-edited responsibilities tailored to requirements.
    This is where LLM metadata belongs.
    """

    responsibility_key: str
    requirement_key: str
    responsibility: str


class SimilarityMetricsRow(BaseStageRow):
    """
    Pairwise similarity/entailment between resume responsibilities and
    job requirements.

    Not an LLM artifact; track your metrics backends separately if needed.
    """

    # join keys
    responsibility_key: str
    requirement_key: str

    # denormalized text (handy for inspection/debug)
    responsibility: str
    requirement: str

    # editorial pass (ORIGINAL vs EDITED)
    version: Version = Field(default=Version.ORIGINAL)

    # editor provenance (ONLY populated when version=EDITED)
    resp_llm_provider: Optional[str] = None
    resp_model_id: Optional[str] = None

    # metrics backends (these identify the scoring engines)
    similarity_backend: Optional[str] = (
        None  # e.g. "sentence-transformers/all-mpnet-base-v2"
    )
    nli_backend: Optional[str] = None  # e.g. "microsoft/deberta-v3-large-mnli"

    # raw scores
    bert_score_precision: Optional[float] = None
    soft_similarity: Optional[float] = None
    word_movers_distance: Optional[float] = None
    deberta_entailment_score: Optional[float] = None
    roberta_entailment_score: Optional[float] = None

    # categorized
    bert_score_precision_cat: Optional[str] = None
    soft_similarity_cat: Optional[str] = None
    word_movers_distance_cat: Optional[str] = None
    deberta_entailment_score_cat: Optional[str] = None
    roberta_entailment_score_cat: Optional[str] = None

    # derived/scaled
    scaled_bert_score_precision: Optional[float] = None
    scaled_soft_similarity: Optional[float] = None
    scaled_word_movers_distance: Optional[float] = None
    scaled_deberta_entailment_score: Optional[float] = None
    scaled_roberta_entailment_score: Optional[float] = None
    composite_score: Optional[float] = None
    pca_score: Optional[float] = None


# ──────────────────────────────────────────────────────────────────────────────
# Explicit export list
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "PipelineState",
    "JobUrlsRow",
    "JobPostingsRow",
    "ExtractedRequirementsRow",
    "FlattenedRequirementsRow",
    "FlattenedResponsibilitiesRow",
    "PrunedResponsibilitiesRow",
    "EditedResponsibilitiesRow",
    "SimilarityMetricsRow",
    # bases/mixins (export if the registry/type checks need them)
    "AppBaseModel",
    "BaseStageRow",
    "TimestampedMixin",
    "LLMStampedMixin",
]
