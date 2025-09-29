# db_io/db_table_registry.py

"""
Authoritative registry for all DuckDB tables used in the resume–job alignment pipeline.

This module centralizes:
- ✅ Table names (`TableName` enum) for structured reference
- ✅ Pydantic models (`BaseModel` subclasses) defining schema/validation
- ✅ Primary key definitions per table
- ✅ Explicit per-table column order (authoritative; no auto-derivation)
- ✅ Helper methods to align, deduplicate, and check DataFrame content

By using this registry, the pipeline keeps database shape and validation code in sync,
and guarantees stable INSERT/SELECT ordering across all stages.

---
🧩 COMPONENT RELATIONSHIPS

    +------------------+         +------------------------+        +-------------------+
    |  pipeline_enums  |  --->   |  DUCKDB_SCHEMA_REGISTRY|  --->  |  TableSchema      |
    |   (TableName)    |         |  (central mapping)     |        |  (per table info) |
    +------------------+         +------------------------+        +-------------------+

Each entry in `DUCKDB_SCHEMA_REGISTRY` maps a `TableName` enum to a `TableSchema`.
`TableSchema` encapsulates the Pydantic model, the primary keys, and the explicit
column order to use for IO.

---
✅ USAGE EXAMPLES

>>> from db_io.db_table_registry import DUCKDB_SCHEMA_REGISTRY, TableName
>>> schema = DUCKDB_SCHEMA_REGISTRY[TableName.EXTRACTED_REQUIREMENTS]
>>> schema_columns = DUCKDB_SCHEMA_REGISTRY[TableName.JOB_POSTINGS].column_order

# Align a DataFrame to the canonical order:
>>> df = schema.align_df(df)
>>> df = schema.drop_dupes(df)
>>> schema.log_pk_dupes(df, logger)

# Query by URL with a stable projection:
>>> sql = schema.select_by_url_sql(["url", "requirement_key", "requirement"],
...                                order_by="requirement_key")
>>> con.execute(sql, ("https://example.com/job1",)).df()

This file is a **single source of truth** (SST) for DuckDB table behavior.
"""

# Imports
import logging
from typing import Optional, Type
from pydantic import BaseModel

from job_bot.db_io.pipeline_enums import TableName
from job_bot.db_io.db_utils import align_df_with_schema
from job_bot.models.db_table_models import (
    PipelineState,
    JobUrlsRow,
    JobPostingsRow,
    ExtractedRequirementsRow,
    FlattenedRequirementsRow,
    FlattenedResponsibilitiesRow,
    EditedResponsibilitiesRow,
    SimilarityMetricsRow,
)

logger = logging.getLogger(__name__)

# Which columns, if present, we treat as "standard metadata" in summaries
STANDARD_METADATA_FIELDS = {
    "source_file",
    "iteration",
    "created_at",
    "updated_at",
    "llm_provider",
    "model_id",
    "version",
    "resp_llm_provider",
    "resp_model_id",
}

# --- 🔒 Explicit column order per table (authoritative) ---
# Keep these lists in the exact order you want them SELECT/INSERTed.
DUCKDB_COLUMN_ORDER = {
    TableName.PIPELINE_CONTROL.value: [
        "url",
        "iteration",
        "stage",
        "status",
        "version",
        "source_file",
        "notes",
        "created_at",
        "updated_at",
    ],
    TableName.JOB_URLS.value: [
        "url",
        "company",
        "job_title",
        "source_file",
        "created_at",
        "updated_at",
    ],
    # ✅ Includes llm_provider/model_id
    TableName.JOB_POSTINGS.value: [
        "url",
        "iteration",
        "llm_provider",
        "model_id",
        "job_title",
        "company",
        "location",
        "salary_info",
        "posted_date",
        "content",
        "source_file",
        "created_at",
        "updated_at",
    ],
    # ✅ Includes llm_provider/model_id
    TableName.EXTRACTED_REQUIREMENTS.value: [
        "url",
        "iteration",
        "llm_provider",
        "model_id",
        "requirement_key",
        "requirement",
        "requirement_category",
        "requirement_category_key",
        "source_file",
    ],
    TableName.FLATTENED_REQUIREMENTS.value: [
        "url",
        "iteration",
        "requirement_key",
        "requirement",
        "source_file",
    ],
    TableName.FLATTENED_RESPONSIBILITIES.value: [
        "url",
        "iteration",
        "responsibility_key",
        "responsibility",
        "source_file",
    ],
    TableName.EDITED_RESPONSIBILITIES.value: [
        "url",
        "iteration",
        "llm_provider",
        "model_id",
        "responsibility_key",
        "requirement_key",
        "responsibility",
        "source_file",
        "created_at",
        "updated_at",
    ],
    TableName.SIMILARITY_METRICS.value: [
        "url",
        "iteration",
        "version",  # original/edited
        "resp_llm_provider",  # editor provenance (for edited pass)
        "resp_model_id",
        "similarity_backend",  # scoring engines
        "nli_backend",
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
        "source_file",
    ],
}


class TableSchema:
    """
    Encapsulates DuckDB table schema metadata and helper utilities.

    This class includes:
    - the Pydantic model (for validation elsewhere)
    - primary key definitions
    - explicit column order

    It also provides utilities for DataFrame alignment, deduplication,
    and logging of duplicate primary-key rows.
    """

    def __init__(
        self,
        model: Type[BaseModel],
        table_name: str,
        primary_keys: list[str],
        *,
        column_order: Optional[list[str]] = None,
    ):
        self.model = model
        self.table_name = table_name
        self.primary_keys = primary_keys

        # Enforce explicit order (no fallbacks to model). Either provided or from registry.
        final_cols: list[str]
        if column_order is not None:
            final_cols = column_order
        else:
            # KeyError if missing is intentional: registry must define order.
            final_cols = DUCKDB_COLUMN_ORDER[table_name]
        self.column_order: list[str] = final_cols

    @property
    def metadata_fields(self) -> list[str]:
        return [c for c in self.column_order if c in STANDARD_METADATA_FIELDS]

    def align_df(self, df):
        """
        Align a DataFrame to this table's canonical column order.

        Raises:
            ValueError: If required columns are missing or extra columns exist (strict mode).
        """
        try:
            return align_df_with_schema(df, self.column_order, strict=True)
        except Exception as e:
            logger.error("❌ Align failed for '%s': %s", self.table_name, e)
            raise

    def drop_dupes(self, df):
        """Drop duplicate rows based on this table's primary key."""
        return df.drop_duplicates(subset=self.primary_keys)

    def log_pk_dupes(self, df, logger_):
        """
        Log any rows in `df` that violate the primary key uniqueness.
        """
        if not all(col in df.columns for col in self.primary_keys):
            logger_.warning(
                "Cannot check PK dupes — missing df fields: %s", self.primary_keys
            )
            return

        dupes = df[df.duplicated(subset=self.primary_keys, keep=False)]
        if not dupes.empty:
            logger_.warning(
                "❗ %d duplicate rows by PK %s", len(dupes), self.primary_keys
            )
            try:
                logger_.warning(
                    "Sample duplicate keys:\n%s",
                    dupes[self.primary_keys].drop_duplicates().head(),
                )
            except Exception:
                pass

    def select_by_url_sql(self, cols: list[str], order_by: str | None = None) -> str:
        """
        Build a parameterized SELECT to retrieve all rows for a given URL.

        Args:
            cols: Columns to project (must exist in this table).
            order_by: Optional column to ORDER BY (must exist if provided).

        Returns:
            SQL string using `?` placeholder for the URL parameter.

        Example:
            >>> schema = DUCKDB_SCHEMA_REGISTRY[TableName.FLATTENED_REQUIREMENTS]
            >>> sql = schema.select_by_url_sql(["url", "requirement_key", "requirement"],
            ...                                order_by="requirement_key")
            >>> con.execute(sql, ("https://example.com/job1",)).df()
        """
        unknown = [c for c in cols if c not in self.column_order]
        if unknown:
            raise ValueError(f"{self.table_name}: unknown columns requested: {unknown}")
        if order_by and order_by not in self.column_order:
            raise ValueError(f"{self.table_name}: unknown order_by: {order_by}")

        select_list = ", ".join(cols)
        sql = f"SELECT {select_list} FROM {self.table_name} WHERE url = ?"
        if order_by:
            sql += f" ORDER BY {order_by}"
        return sql

    def summary(self) -> str:
        """
        Return a human-readable summary of this table's schema configuration.
        """
        return (
            f"📊 Table: {self.table_name}\n"
            f"🔑 Primary Keys: {', '.join(self.primary_keys)}\n"
            f"📋 Columns: {', '.join(self.column_order)}\n"
            f"🧩 Metadata Fields: {', '.join(self.metadata_fields)}"
        )

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.table_name} with PK {self.primary_keys}>"

    def __str__(self):
        return f"{self.table_name} table\nPrimary keys: {', '.join(self.primary_keys)}"


# ✅ Centralized registry
DUCKDB_SCHEMA_REGISTRY = {
    # FSM snapshot ONLY: one row per (url, iteration)
    TableName.PIPELINE_CONTROL: TableSchema(
        model=PipelineState,
        table_name=TableName.PIPELINE_CONTROL.value,
        primary_keys=["url", "iteration"],
        column_order=DUCKDB_COLUMN_ORDER[TableName.PIPELINE_CONTROL.value],
    ),
    # Seed list: NO iteration
    TableName.JOB_URLS: TableSchema(
        model=JobUrlsRow,
        table_name=TableName.JOB_URLS.value,
        primary_keys=["url"],
        column_order=DUCKDB_COLUMN_ORDER[TableName.JOB_URLS.value],
    ),
    # Raw scraped content (per pass; provider/model stamped in PK)
    TableName.JOB_POSTINGS: TableSchema(
        model=JobPostingsRow,
        table_name=TableName.JOB_POSTINGS.value,
        primary_keys=["url", "iteration", "llm_provider", "model_id"],
        column_order=DUCKDB_COLUMN_ORDER[TableName.JOB_POSTINGS.value],
    ),
    # Requirements extracted from posting (per pass; provider/model stamped in PK)
    TableName.EXTRACTED_REQUIREMENTS: TableSchema(
        model=ExtractedRequirementsRow,
        table_name=TableName.EXTRACTED_REQUIREMENTS.value,
        primary_keys=[
            "url",
            "iteration",
            "requirement_key",
            "llm_provider",
            "model_id",
        ],
        column_order=DUCKDB_COLUMN_ORDER[TableName.EXTRACTED_REQUIREMENTS.value],
    ),
    # Flattened requirements used for matching (per pass)
    TableName.FLATTENED_REQUIREMENTS: TableSchema(
        model=FlattenedRequirementsRow,
        table_name=TableName.FLATTENED_REQUIREMENTS.value,
        primary_keys=["url", "iteration", "requirement_key"],
        column_order=DUCKDB_COLUMN_ORDER[TableName.FLATTENED_REQUIREMENTS.value],
    ),
    # Canonical resume bullets (per pass)
    TableName.FLATTENED_RESPONSIBILITIES: TableSchema(
        model=FlattenedResponsibilitiesRow,
        table_name=TableName.FLATTENED_RESPONSIBILITIES.value,
        primary_keys=["url", "iteration", "responsibility_key"],
        column_order=DUCKDB_COLUMN_ORDER[TableName.FLATTENED_RESPONSIBILITIES.value],
    ),
    # LLM-edited bullets (per pass; provider-specific)
    TableName.EDITED_RESPONSIBILITIES: TableSchema(
        model=EditedResponsibilitiesRow,
        table_name=TableName.EDITED_RESPONSIBILITIES.value,
        primary_keys=[
            "url",
            "iteration",
            "responsibility_key",
            "requirement_key",
            "llm_provider",
            "model_id",
        ],
        column_order=DUCKDB_COLUMN_ORDER[TableName.EDITED_RESPONSIBILITIES.value],
    ),
    # Pairwise similarity/entailment scores (per pass)
    TableName.SIMILARITY_METRICS: TableSchema(
        model=SimilarityMetricsRow,
        table_name=TableName.SIMILARITY_METRICS.value,
        primary_keys=[
            "url",
            "iteration",
            "responsibility_key",
            "requirement_key",
            "version",
            "resp_llm_provider",
            "resp_model_id",
        ],
        column_order=DUCKDB_COLUMN_ORDER[TableName.SIMILARITY_METRICS.value],
    ),
}
