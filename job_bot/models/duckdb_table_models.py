"""
/models/duckdb_table_models.py

Pydantic models representing DuckDB tables used in the resume-job alignment pipeline.
Each model corresponds to one table and includes a shared base class for metadata fields.
"""

from datetime import datetime
from typing import Optional, Union
from pydantic import BaseModel, ConfigDict, HttpUrl, Field


# ✅ Import enums from schema_definitions
from db_io.pipeline_enums import (
    PipelineStage,
    PipelineStatus,
    Version,
    LLMProvider,
)


# 🔁 Shared Base Model with common metadata fields
class BaseDBModel(BaseModel):
    """
    Base model for DuckDB table rows with shared metadata fields:
    `url`, `iteration`, `stage`, `source_file`, `timestamp`, `version`, `llm_provider`
    """

    url: Union[HttpUrl, str] = Field(
        ..., description="Job posting URL this row belongs to."
    )
    iteration: int = Field(
        0,
        description="Which full pipeline cycle this record belongs to (starts at 0, increments on \
reprocessing).",
    )
    stage: PipelineStage = Field(
        ..., description="Pipeline stage that generated this row."
    )
    source_file: Optional[str] = Field(
        None, description="File path this record was loaded from."
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp marking when this record was ingested.",
    )
    version: Version = Field(
        default=Version.ORIGINAL,
        description="Version of the data (e.g., original, edited, final).",
    )
    llm_provider: LLMProvider = Field(
        default=LLMProvider.NONE,
        description="LLM provider used to generate or process this row (e.g., openai).",
    )


class FlattenedResponsibilitiesRow(BaseDBModel):
    """
    Flattened responsibilities extracted from a resume.
    Each row represents one responsibility with a structured key.
    """

    responsibility_key: str = Field(
        ..., description="Key that identifies the source responsibility."
    )
    responsibility: str = Field(
        ..., description="Text of the original resume responsibility."
    )


class FlattenedRequirementsRow(BaseDBModel):
    """
    Flattened job requirements extracted from a job posting.
    Each row represents one requirement with a structured key.
    """

    requirement_key: str = Field(
        ..., description="Key that identifies the job requirement."
    )
    requirement: str = Field(..., description="Text of the job requirement.")


class EditedResponsibilitiesRow(BaseDBModel):
    """
    Responsibilities rewritten by an LLM to align with a specific job requirement.
    """

    responsibility_key: str = Field(
        ..., description="Key of the original responsibility."
    )
    requirement_key: str = Field(..., description="Key of the target requirement.")
    # optimized_text: str = Field(
    #     ..., description="LLM-edited version of the responsibility."
    # )
    responsibility: str = Field(
        ..., description="LLM-edited version of the responsibility."
    )  # new: standardize on responsibility colname


class SimilarityMetricsRow(BaseDBModel):
    """
    Row-level similarity and entailment metrics between a resume responsibility and
    a job requirement.

    Includes raw scores, scaled values, and optional composite/PCA scores.
    """

    responsibility_key: str
    requirement_key: str
    responsibility: str
    requirement: str

    # Raw similarity scores
    bert_score_precision: float
    soft_similarity: float
    word_movers_distance: float
    deberta_entailment_score: float
    roberta_entailment_score: float

    # Categorized (bucketed) values
    bert_score_precision_cat: Optional[str]
    soft_similarity_cat: Optional[str]
    word_movers_distance_cat: Optional[str]
    deberta_entailment_score_cat: Optional[str]
    roberta_entailment_score_cat: Optional[str]

    # Scaled scores
    scaled_bert_score_precision: Optional[float]
    scaled_soft_similarity: Optional[float]
    scaled_word_movers_distance: Optional[float]
    scaled_deberta_entailment_score: Optional[float]
    scaled_roberta_entailment_score: Optional[float]

    composite_score: Optional[float] = Field(
        None, description="Weighted combination of multiple similarity metrics."
    )
    pca_score: Optional[float] = Field(
        None, description="Dimensionality-reduced score via PCA."
    )


class JobPostingsRow(BaseDBModel):
    """
    Raw scraped job posting content in flat format.
    Includes job metadata and full content as a JSON string.
    """

    status: PipelineStatus
    message: Optional[str]
    job_title: str
    company: str
    location: Optional[str]
    salary_info: Optional[str]
    posted_date: Optional[str]
    content: Optional[str] = Field(
        None, description="Full job description content as a JSON string."
    )


class JobUrlsRow(BaseDBModel):
    """
    Master registry of job posting URLs and their associated metadata.
    Used as the entry point for the pipeline.
    """

    company: str
    job_title: str


class ExtractedRequirementRow(BaseDBModel):
    """
    Structured job requirements grouped by category and extracted from job content.
    Includes category indices and individual item order.
    """

    status: PipelineStatus
    message: Optional[str]
    requirement_category: str
    requirement_category_idx: int
    requirement: str
    requirement_idx: int


class PrunedResponsibilitiesRow(BaseDBModel):
    """
    Resume responsibilities that have been filtered or pruned during human review or LLM trimming.
    """

    responsibility_key: str
    responsibility: str
    pruned_by: str = Field(
        ...,
        description="Identifier for who or what performed the pruning (e.g., 'llm', 'xfz').",
    )


# * ✅ Pipeline control table model
class PipelineState(BaseDBModel):
    """Tracks the lifecycle and status of a job posting as it progresses through the pipeline."""

    stage: Optional[PipelineStage] = Field(
        default=None,
        exclude=True,
        description="Excluded. Control model uses `last_stage` instead.",
    )  # * ✅ control table uses last stage to infers last and next stages (state machine)

    status: PipelineStatus = Field(
        default=PipelineStatus.NEW,
        description="Pipeline status (new, in_progress, complete, skipped, error)",
    )
    last_stage: Optional[PipelineStage] = Field(
        default=None, description="Last completed stage."
    )
    is_active: bool = Field(
        default=True, description="Whether this job is currently active."
    )
    notes: Optional[str] = Field(
        default=None, description="Optional notes or debug info."
    )

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
