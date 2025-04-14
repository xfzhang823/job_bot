"""
schema_definitions.py

This module defines the SQL `CREATE TABLE` statements used to set up all DuckDB tables
in the resume-job alignment pipeline. Each table schema is designed for a specific stage
in the pipeline, ensuring standardized structure, traceability, and efficient querying
across preprocessing, editing, evaluation, and cleanup workflows.


Stages in the pipeline:

- job_urls:
    Initial ingestion of job posting URLs. This is the starting point of the pipeline where
    all target URLs are registered and assigned metadata (e.g., company, job title).

- job_postings:
    Crawls or parses each job URL to extract job content. This includes job title, company,
    location, salary, and the full description in structured form.

- extracted_requirements:
    Applies an LLM to extract structured job requirements grouped by logical categories such as
    'pie_in_the_sky', 'down_to_earth', and 'bare_minimum'. Output is a nested requirements model.

- flattened_requirements:
    Flattens the nested job requirements into key-value form, enabling direct alignment with
    resume responsibilities. Keys are structured as `category.index` (e.g., `0.down_to_earth.1`).

- flattened_responsibilities:
    Flattens the original resume responsibilities into key-value pairs for semantic comparison.
    Each bullet point or grouped item becomes one record.

- edited_responsibilities:
    Uses an LLM to rewrite resume responsibilities to better match job requirements. Each match
    pair is edited with optimized language for alignment and tracked by
    responsibility/requirement keys.

- similarity_metrics_eval:
    Computes similarity and entailment metrics (e.g., BERTScore, WMD, DeBERTa/Roberta entailment)
    between original responsibilities and flattened requirements.

- similarity_metrics_reval:
    Recomputes the same set of metrics for the LLM-edited responsibilities to assess improvements
    after editing.

- crosstab_review:
    Produces cross-tab views and comparative visualizations for human QA. May include LLM feedback
    or reviewer notes.

- final_responsibilities:
    Outputs the finalized and optionally trimmed version of each edited responsibility for delivery.
    May be used for presentation, export, or downstream integration with resume builders.


Each table schema includes standardized metadata columns:
- `source_file`: Indicates the file or batch source used for ingestion.
- `stage`: Identifies the processing stage at which the table was generated or
updated.
- `timestamp`: Auto-generated field marking the time of row creation.

All schemas are stored in the `DUCKDB_SCHEMAS` dictionary, keyed by table name,
and can be used to programmatically initialize or validate the DuckDB instance in
a consistent and reproducible way.

---
>>> Usage Example:

import duckdb
from schema_definitions import DUCKDB_SCHEMAS

con = duckdb.connect("pipeline.db")
for name, ddl in DUCKDB_SCHEMAS.items():
    con.execute(ddl)
"""

from enum import Enum

# todo: need to update primary key (makle llm provider a key too)
DUCKDB_SCHEMAS = {
    "pipeline_control": """
        CREATE TABLE IF NOT EXISTS pipeline_control (
            url TEXT PRIMARY KEY,
            llm_provider TEXT DEFAULT 'openai',
            iteration INTEGER DEFAULT 0,
            version TEXT DEFAULT 'original',
            status TEXT DEFAULT 'new',
            is_active BOOLEAN DEFAULT TRUE,
            stage TEXT,
            timestamp TIMESTAMP DEFAULT current_timestamp,
            notes TEXT,
            PRIMARY KEY (url, iteration, version)
        );
    """,
    "job_urls": """
        CREATE TABLE IF NOT EXISTS job_urls (
            url TEXT PRIMARY KEY,
            company TEXT,
            job_title TEXT,
            iteration INTEGER DEFAULT 0,
            source_file TEXT,
            stage TEXT,
            timestamp TIMESTAMP DEFAULT current_timestamp
        );
    """,
    "job_postings": """
        CREATE TABLE IF NOT EXISTS job_postings (
            url TEXT,
            status TEXT,
            message TEXT,
            job_title TEXT,
            company TEXT,
            location TEXT,
            salary_info TEXT,
            posted_date TEXT,
            content TEXT,
            iteration INTEGER DEFAULT 0,
            source_file TEXT,
            stage TEXT,
            timestamp TIMESTAMP DEFAULT current_timestamp
        );
    """,
    "extracted_requirements": """
        CREATE TABLE IF NOT EXISTS extracted_requirements (
            url TEXT,
            status TEXT,
            message TEXT,
            requirement_category TEXT,
            requirement_category_idx INTEGER,
            requirement TEXT,
            requirement_idx INTEGER,
            iteration INTEGER DEFAULT 0,
            source_file TEXT,
            stage TEXT,
            timestamp TIMESTAMP DEFAULT current_timestamp
        );
    """,
    "flattened_responsibilities": """
        CREATE TABLE IF NOT EXISTS flattened_responsibilities (
            url TEXT,
            responsibility_key TEXT,
            responsibility TEXT,
            iteration INTEGER DEFAULT 0,
            source_file TEXT,
            stage TEXT,
            timestamp TIMESTAMP DEFAULT current_timestamp
        );
    """,
    "flattened_requirements": """
        CREATE TABLE IF NOT EXISTS flattened_requirements (
            url TEXT,
            requirement_key TEXT,
            requirement TEXT,
            iteration INTEGER DEFAULT 0,
            source_file TEXT,
            stage TEXT,
            timestamp TIMESTAMP DEFAULT current_timestamp
        );
    """,
    "pruned_responsibilities": """
        CREATE TABLE IF NOT EXISTS pruned_responsibilities (
            url TEXT,
            responsibility_key TEXT,
            responsibility TEXT,
            pruned_by TEXT,
            iteration INTEGER DEFAULT 0,
            source_file TEXT,
            stage TEXT,
            timestamp TIMESTAMP DEFAULT current_timestamp
        );
    """,
    "edited_responsibilities": """
        CREATE TABLE IF NOT EXISTS edited_responsibilities (
            url TEXT,
            responsibility_key TEXT,
            requirement_key TEXT,
            optimized_text TEXT,
            llm_provider TEXT,
            iteration INTEGER DEFAULT 0,
            source_file TEXT,
            stage TEXT,
            timestamp TIMESTAMP DEFAULT current_timestamp
        );
    """,
    "similarity_metrics": """
        CREATE TABLE IF NOT EXISTS similarity_metrics (
            url TEXT,
            responsibility_key TEXT,
            requirement_key TEXT,
            responsibility TEXT,
            requirement TEXT,

            bert_score_precision DOUBLE,
            soft_similarity DOUBLE,
            word_movers_distance DOUBLE,
            deberta_entailment_score DOUBLE,
            roberta_entailment_score DOUBLE,

            bert_score_precision_cat TEXT,
            soft_similarity_cat TEXT,
            word_movers_distance_cat TEXT,
            deberta_entailment_score_cat TEXT,
            roberta_entailment_score_cat TEXT,

            scaled_bert_score_precision DOUBLE,
            scaled_soft_similarity DOUBLE,
            scaled_word_movers_distance DOUBLE,
            scaled_deberta_entailment_score DOUBLE,
            scaled_roberta_entailment_score DOUBLE,

            composite_score DOUBLE,
            pca_score DOUBLE,

            version TEXT,
            llm_provider TEXT,
            iteration INTEGER DEFAULT 0,
            source_file TEXT,
            stage TEXT,
            timestamp TIMESTAMP DEFAULT current_timestamp
        );
    """,
    # TODO: add later
    # "crosstab_review": """
    #     CREATE TABLE IF NOT EXISTS crosstab_review (
    #         url TEXT,
    #         review_notes TEXT,
    #         reviewer TEXT,
    #         source_file TEXT,
    #         stage TEXT,
    #         timestamp TIMESTAMP DEFAULT current_timestamp
    #     );
    # """,
    # "final_responsibilities": """
    #     CREATE TABLE IF NOT EXISTS final_responsibilities (
    #         url TEXT,
    #         responsibility_key TEXT,
    #         responsibility TEXT,
    #         optimized_text TEXT,
    #         trimmed_by TEXT,
    #         source_file TEXT,
    #         stage TEXT,
    #         timestamp TIMESTAMP DEFAULT current_timestamp
    #     );
    # """,
}


# * Column order (table: col names)
DUCKDB_COLUMN_ORDER = {
    "job_urls": [
        "url",
        "company",
        "job_title",
        "iteration",
        "source_file",
        "stage",
        "timestamp",
    ],
    "job_postings": [
        "url",
        "status",
        "message",
        "job_title",
        "company",
        "location",
        "salary_info",
        "posted_date",
        "content",
        "iteration",
        "source_file",
        "stage",
        "timestamp",
    ],
    "extracted_requirements": [
        "url",
        "status",
        "message",
        "requirement_category",
        "requirement_category_idx",
        "requirement",
        "requirement_idx",
        "iteration",
        "source_file",
        "stage",
        "timestamp",
    ],
    "flattened_responsibilities": [
        "url",
        "responsibility_key",
        "responsibility",
        "iteration",
        "source_file",
        "stage",
        "timestamp",
    ],
    "flattened_requirements": [
        "url",
        "requirement_key",
        "requirement",
        "iteration",
        "source_file",
        "stage",
        "timestamp",
    ],
    "pruned_responsibilities": [
        "url",
        "responsibility_key",
        "responsibility",
        "pruned_by",
        "iteration",
        "source_file",
        "stage",
        "timestamp",
    ],
    "edited_responsibilities": [
        "url",
        "responsibility_key",
        "requirement_key",
        "optimized_text",
        "llm_provider",
        "iteration",
        "source_file",
        "stage",
        "timestamp",
    ],
    "similarity_metrics": [
        "url",
        "responsibility_key",
        "requirement_key",
        "responsibility",
        "requirement",
        "bert_score_precision",
        "soft_similarity",
        "word_movers_distance",
        "deberta_entailment_score",
        "roberta_entailment_score",
        "bert_score_precision_cat",
        "soft_similarity_cat",
        "word_movers_distance_cat",
        "deberta_entailment_score_cat",
        "roberta_entailment_score_cat",
        "scaled_bert_score_precision",
        "scaled_soft_similarity",
        "scaled_word_movers_distance",
        "scaled_deberta_entailment_score",
        "scaled_roberta_entailment_score",
        "composite_score",
        "pca_score",
        "version",
        "llm_provider",
        "iteration",
        "source_file",
        "stage",
        "timestamp",
    ],
}


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

    # ✅ Editing (LLM)
    EDITED_RESPONSIBILITIES = "edited_responsibilities"

    # todo: keep it out for now (may not include in final version)
    # PRUNED_RESPONSIBILITIES = "pruned_responsibilities"

    # ✅ Evaluation
    SIM_METRICS_EVAL = "similarity_metrics_eval"  # original responsibilities vs reqs
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
        >>> df["status"] = PipelineStage.PREPROCESSING.value
        >>> assert stage in PipelineStage.list()
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
