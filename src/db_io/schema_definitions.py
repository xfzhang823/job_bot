"""
schema_definitions.py

This module defines the SQL `CREATE TABLE` statements used to set up all DuckDB tables
in the resume-job alignment pipeline. Each table schema is designed for a specific stage
in the pipeline, ensuring standardized structure, traceability, and efficient querying
across preprocessing, editing, evaluation, and cleanup workflows.

---

Stages in the pipeline:
- preprocessing: Initial ingestion and structuring of job posting data
(URLs, content, requirements).
- staging: Transformation of nested structures into flattened representations
(responsibilities, requirements).
- evaluation: Similarity and entailment scoring between resume responsibilities
and job requirements.
- editing: LLM-powered optimization of resume content for alignment with
job requirements.
- revaluation: Re-scoring of edited responsibilities for improvement tracking.
- cleanup: Final filtering, pruning, and consolidation of output data.

---

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


DUCKDB_SCHEMAS = {
    "pipeline_control": """
        CREATE TABLE IF NOT EXISTS pipeline_control (
            url TEXT PRIMARY KEY,                   -- The job posting URL
            llm_provider TEXT DEFAULT 'openai',     -- 'openai', 'anthropic', etc.
            version TEXT DEFAULT 'original',        -- 'original', 'edited', etc.
            status TEXT DEFAULT 'new',              -- 'new', 'in_progress', 'complete', 'skipped'
            is_active BOOLEAN DEFAULT TRUE,         -- Whether this row is in-scope for processing
            stage TEXT,                             -- 'preprocessing', 'staging', 'evaluation', 'revaluation', 'cleanup'
            last_updated TIMESTAMP DEFAULT current_timestamp,
            notes TEXT                              -- Optional free-form metadata (e.g., 'Missing reqs')
        );
    """,
    "job_urls": """
		CREATE TABLE IF NOT EXISTS job_urls (
			url TEXT PRIMARY KEY,
			company TEXT,
			job_title TEXT,
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
            requirement_category TEXT,     -- e.g., 'pie_in_the_sky'
			requirement_category_idx INTEGER,      -- order in the original JSON object
            requirement TEXT,              -- e.g., "Advanced operations skills"
            requirement_idx INTEGER,       -- index within the list
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

            version TEXT,                  -- 'original' or 'edited'
            llm_provider TEXT,             -- optional, filled for edited only
            source_file TEXT,
            stage TEXT,
            timestamp TIMESTAMP DEFAULT current_timestamp
        );
    """,
}


# * Column order (table: col names)
DUCKDB_COLUMN_ORDER = {
    "job_urls": [
        "url",
        "company",
        "job_title",
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
        "source_file",
        "stage",
        "timestamp",
    ],
    "flattened_responsibilities": [
        "url",
        "responsibility_key",
        "responsibility",
        "source_file",
        "stage",
        "timestamp",
    ],
    "flattened_requirements": [
        "url",
        "requirement_key",
        "requirement",
        "source_file",
        "stage",
        "timestamp",
    ],
    "pruned_responsibilities": [
        "url",
        "responsibility_key",
        "responsibility",
        "pruned_by",
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
        "source_file",
        "stage",
        "timestamp",
    ],
}


class PipelineStage(str, Enum):
    """
    Class to define stages

    Example Usage:
        df["stage"] = PipelineStage.PREPROCESSING.value
        assert stage in PipelineStage.list()
    """

    PREPROCESSING = "preprocessing"
    STAGING = "staging"
    EVALUATION = "evaluation"
    EDITING = "editing"
    REVALUATION = "revaluation"
    CLEANUP = "cleanup"

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
