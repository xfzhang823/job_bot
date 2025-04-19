"""
db_io/pipeline_enums.py

This module defines all core Enum types used across the resume-job
alignment pipeline.

It centralizes canonical value definitions for:
- Pipeline stages (FSM state tracking)
- Pipeline statuses (job progression)
- LLM providers (used during editing/evaluation)
- Versioning (original vs. edited content)
- DuckDB table names (typed reference to all tables)

These enums serve as a single source of truth and are reused across:
- DuckDB schema generation
- Pydantic model validation
- FSM control logic
- Filtering, logging, and pipeline introspection
- Prompt generation and result parsing

Included Components:
- ✅ `PipelineStage`: Defines linear FSM stages
(e.g., job_urls → final_responsibilities)
- ✅ `PipelineStatus`: High-level job status in the control table
(new, in_progress, etc.)
- ✅ `LLMProvider`: Supported LLM provider values
(e.g., openai, anthropic)
- ✅ `Version`: Content lifecycle versions
(original, edited, final)
- ✅ `TableName`: Enum of all DuckDB tables used throughout the pipeline
"""

from enum import Enum
from project_config import OPENAI, ANTHROPIC


# * ✅ Metadata ENUM Classes
class LLMProvider(str, Enum):
    """Name of the LLM provider used during this pipeline pass."""

    OPENAI = OPENAI
    ANTHROPIC = ANTHROPIC
    MISTRAL = "mistral"
    LLAMA = "llama"
    NONE = "none"  # ✅ For non-LLM-driven stages

    @classmethod
    def values(cls) -> list[str]:
        return [e.value for e in cls]


class PipelineStage(str, Enum):
    """
    * Class to define stages (redefined to use table names explicitly as stages)

    Example Usage:
        df["stage"] = PipelineStage.PREPROCESSING.value
        assert stage in PipelineStage.list()
    """

    # ✅ Preprocessing Stages
    JOB_URLS = "job_urls"
    JOB_POSTINGS = "job_postings"
    EXTRACTED_REQUIREMENTS = "extracted_requirements"

    # ✅ Staging
    FLATTENED_REQUIREMENTS = "flattened_requirements"
    FLATTENED_RESPONSIBILITIES = "flattened_responsibilities"

    # ✅ Evaluation
    SIM_METRICS_EVAL = "similarity_metrics_eval"  # original responsibilities vs reqs

    # ✅ Editing (LLM)
    EDITED_RESPONSIBILITIES = "edited_responsibilities"

    # todo: keep it out for now (may not include in final version)
    # PRUNED_RESPONSIBILITIES = "pruned_responsibilities"

    # ✅ Revaluation
    SIM_METRICS_REVAL = "similarity_metrics_reval"  # edited responsibilities vs reqs

    # ✅ Human Review (optional stages to be added)
    CROSSTAB_REVIEW = "crosstab_review"  # cross-tab visualization + feedback

    # ✅ Export
    FINAL_RESPONSIBILITIES = "final_responsibilities"  # manually trimmed/pruned output

    @classmethod
    def list(cls) -> list[str]:
        return [stage.value for stage in cls]


class PipelineStatus(str, Enum):
    """
    * Class to define Status

    ! Pipeline Status is different from status in JobPostings and ExtractedRequirements tables,
    ! which refers LLM API call status

    Example:
        >>> df["status"] = PipelineStatus.NEW.value
        >>> assert df["status"] in PipelineStatus.list()
    """

    NEW = "new"  # Not yet started
    IN_PROGRESS = (
        "in_progress"  # This stage completed successfully, but the pipeline continues
    )
    COMPLETE = "complete"  # Final stage completed successfully (end of pipeline)
    ERROR = "error"  # Current stage failed
    SKIPPED = "skipped"  # Explicitly skipped (optional path or filtered out)

    @classmethod
    def list(cls) -> list[str]:
        return [stage.value for stage in cls]


class TableName(str, Enum):
    """
    Enum representing all DuckDB table names used in the resume-job alignment pipeline.

    Each member corresponds to a specific stage or artifact in the data pipeline,
    and is used to enforce consistent naming across ingestion, transformation, and storage.

    >>> Example Usages:

        for table_name in TableName:
        print(f"Processing table: {table_name.value}")
        # hypothetical read
        df = read_table(table_name.value)
        print(df.shape)

        print("Available DuckDB tables:")
        for table in TableName.list():
            print(f" - {table}")

    """

    PIPELINE_CONTROL = "pipeline_control"
    """Controls pipeline progression with state machine."""

    JOB_URLS = "job_urls"
    """Registry of job posting URLs and associated metadata (company, title)."""

    JOB_POSTINGS = "job_postings"
    """Scraped job descriptions including job title, location, and content."""

    EXTRACTED_REQUIREMENTS = "extracted_requirements"
    """Structured LLM-extracted requirements categorized into tiers (e.g., pie_in_the_sky)."""

    FLATTENED_REQUIREMENTS = "flattened_requirements"
    """Flattened job requirements in key-value format, used for alignment."""

    FLATTENED_RESPONSIBILITIES = "flattened_responsibilities"
    """Original resume responsibilities, flattened for semantic evaluation."""

    PRUNED_RESPONSIBILITIES = "pruned_responsibilities"
    """Responsibilities trimmed or filtered by heuristics or LLMs."""

    EDITED_RESPONSIBILITIES = "edited_responsibilities"
    """LLM-optimized responsibilities aligned with job requirements."""

    SIMILARITY_METRICS = "similarity_metrics"
    """Similarity scores and entailment metrics between responsibilities and requirements."""

    @classmethod
    def list(cls) -> list[str]:
        """
        Returns a list of all table names as strings.
        Useful for validation, logging, and iteration.
        """
        return [t.value for t in cls]

    @classmethod
    def from_value(cls, value: str) -> "TableName":
        """
        Converts a string to the corresponding TableName enum member.

        Args:
            value (str): A valid table name string.

        Returns:
            TableName: The corresponding enum member.
        """
        return cls(value)

    def __str__(self) -> str:
        """
        Returns the string value of the enum member.
        Enables clean usage in f-strings, logs, and file naming.
        """
        return self.value


class Version(str, Enum):
    """Content version (original, LLM-edited, human-finalized)."""

    ORIGINAL = "original"
    EDITED = "edited"
    FINAL = "final"

    @classmethod
    def values(cls) -> list[str]:
        return [e.value for e in cls]
