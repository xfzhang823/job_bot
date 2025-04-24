"""
db_table_registry.py

This module serves as the authoritative registry for all DuckDB tables used
in the resume-job alignment pipeline.

It centralizes schema metadata and utilities including:

- ✅ Table names (`TableName` enum) for structured reference
- ✅ Pydantic models (`BaseModel` subclasses) defining schema and validation
- ✅ Primary key definitions per table
- ✅ Canonical column order (inferred from Pydantic model)
- ✅ Auto-generated DuckDB `CREATE TABLE` statements
- ✅ Helper methods to align, deduplicate, and check DataFrame content

By using this module, the pipeline ensures consistent table schema generation,
validation, and data ingestion across all stages, from preprocessing to editing,
evaluation, and export.

---
🧩 COMPONENT RELATIONSHIPS

    +------------------+         +----------------------+         +------------------+
    |  pipeline_enums  |  --->   |  DUCKDB_SCHEMA_REGISTRY  |  --->  |  TableSchema      |
    |   (TableName)    |         |  (central mapping)    |         |  (per table info) |
    +------------------+         +----------------------+         +------------------+

Each entry in `DUCKDB_SCHEMA_REGISTRY` maps a `TableName` enum member to a
`TableSchema` instance. `TableSchema` encapsulates the model, column order,
primary keys, and generated DDL — all derived from the Pydantic definition.

---
✅ USAGE EXAMPLES

>>> from db_io.db_table_registry import DUCKDB_SCHEMA_REGISTRY, TableName

>>> schema = DUCKDB_SCHEMA_REGISTRY[TableName.EXTRACTED_REQUIREMENTS]
>>> schema_columns = DUCKDB_SCHEMA_REGISTRY[TableName.JOB_POSTINGS].column_order

>>> df = schema.align_df(df)
>>> df = schema.drop_dupes(df)

>>> schema.log_pk_dupes(df, logger)

>>> con.execute(schema.ddl)
>>> con.execute(f"INSERT INTO {schema.table_name} SELECT * FROM df")

---

This file acts as a **single source of truth** (SST) for all DuckDB table behavior,
minimizing duplication and maximizing alignment between database, validation code,
and pipeline logic.
"""

# Imports
import logging
from typing import Type
from pydantic import BaseModel
from db_io.db_utils import (
    generate_table_schema_from_model,
    generate_table_column_order_from_model,
)
from db_io.pipeline_enums import *
from db_io.db_insert import align_df_with_schema
from models.duckdb_table_models import (
    PipelineState,
    JobUrlsRow,
    JobPostingsRow,
    ExtractedRequirementRow,
    FlattenedRequirementsRow,
    FlattenedResponsibilitiesRow,
    EditedResponsibilitiesRow,
    SimilarityMetricsRow,
    PrunedResponsibilitiesRow,
)


logger = logging.getLogger(__name__)

STANDARD_METADATA_FIELDS = {
    "source_file",
    "stage",
    "timestamp",
    "version",
    "llm_provider",
    "iteration",
}


class TableSchema:
    """
    Encapsulates DuckDB table schema metadata and helper utilities.

    This class serves as the canonical source for all information related
    to a DuckDB table, including:
    - the Pydantic model
    - primary key definitions
    - canonical column order
    - the generated CREATE TABLE DDL

    It also includes utility methods for DataFrame alignment, deduplication,
    and logging of duplicate primary key rows.
    """

    def __init__(
        self,
        model: Type[BaseModel],
        table_name: str,
        primary_keys: list[str],
    ):
        self.model = model
        self.table_name = table_name
        self.primary_keys = primary_keys

        try:
            self.column_order = generate_table_column_order_from_model(model)
            self.ddl = generate_table_schema_from_model(
                model,
                table_name=table_name,
                primary_keys=primary_keys,
            )
        except Exception as e:
            logger.exception(
                f"❌ Failed to generate schema for table '{table_name}': {e}"
            )
            raise

    @property
    def metadata_fields(self) -> list[str]:
        """
        Returns a list of standardized ingestion metadata fields present in this table.
        """
        return [col for col in self.column_order if col in STANDARD_METADATA_FIELDS]

    def align_df(self, df):
        """
        Aligns a DataFrame to the expected DuckDB column order.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Schema-aligned DataFrame.

        Raises:
            ValueError: If required columns are missing or extra columns exist.
        """
        try:
            return align_df_with_schema(df, self.column_order, strict=True)
        except Exception as e:
            logger.error(
                f"❌ Failed to align DataFrame to schema for '{self.table_name}': {e}"
            )
            raise

    def drop_dupes(self, df):
        """
        Drops duplicate rows based on the table's primary key.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Deduplicated DataFrame.
        """
        return df.drop_duplicates(subset=self.primary_keys)

    def log_pk_dupes(self, df, logger):
        """
        Logs rows in the DataFrame that violate the table's primary key uniqueness.

        Args:
            df (pd.DataFrame): Input DataFrame.
            logger (logging.Logger): Logger instance for warnings.
        """
        if not all(col in df.columns for col in self.primary_keys):
            logger.warning(
                f"Cannot check PK duplicates — missing fields in df: {self.primary_keys}"
            )
            return

        dupes = df[df.duplicated(subset=self.primary_keys, keep=False)]
        if not dupes.empty:
            logger.warning(
                f"❗ Found {len(dupes)} duplicated rows based on PK: {self.primary_keys}"
            )
            try:
                logger.warning(
                    f"Sample duplicate keys: \n{dupes[self.primary_keys].drop_duplicates().head()}"
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to log sample duplicates: {e}")

    def __repr__(self):
        """
        __repr__ method to returns a concise, unambiguous representation of
        the TableSchema instance. Useful for debugging or logging.

        Usage Example:
        >>> schema = DUCKDB_SCHEMA_REGISTRY[TableName.JOB_POSTINGS]
        >>> print(schema)
        >>> Output: <TableSchema: job_postings with PK ['url', 'iteration']>
        """
        return f"<{self.__class__.__name__}: {self.table_name} with PK {self.primary_keys}>"

    def __str__(self):
        """
        Returns a user-friendly string representation for display purposes,
        showing the table name and primary keys.

        Usage Example:
        >>> schema = DUCKDB_SCHEMA_REGISTRY[TableName.JOB_POSTINGS]
        >>> print(schema)
        job_postings table
        Primary keys: url, iteration
        """
        return f"{self.table_name} table\nPrimary keys: {', '.join(self.primary_keys)}"

    def summary(self) -> str:
        """
        Returns a detailed summary of the table schema, including:
        - Table name
        - Primary keys
        - Column order
        - Metadata fields

        Returns:
            str: Multi-line string with key schema metadata.

        Usage Example:
        >>> schema = DUCKDB_SCHEMA_REGISTRY[TableName.JOB_POSTINGS]
        >>> print(schema.summary())
        📊 Table: job_postings
        🔑 Primary Keys: url, iteration
        📋 Columns: url, iteration, job_title, source_file, stage, timestamp
        🧩 Metadata Fields: source_file, stage, timestamp
        """
        return (
            f"\U0001f4ca Table: {self.table_name}\n"
            f"\U0001f511 Primary Keys: {', '.join(self.primary_keys)}\n"
            f"\U0001f4cb Columns: {', '.join(self.column_order)}\n"
            f"\U0001f9e9 Metadata Fields: {', '.join(self.metadata_fields)}"
        )


# ✅ Centralized schema registry
DUCKDB_SCHEMA_REGISTRY = {
    TableName.PIPELINE_CONTROL: TableSchema(
        model=PipelineState,
        table_name=TableName.PIPELINE_CONTROL.value,
        primary_keys=["url", "iteration", "version", "llm_provider"],
    ),
    TableName.JOB_URLS: TableSchema(
        model=JobUrlsRow,
        table_name=TableName.JOB_URLS.value,
        primary_keys=["url"],
    ),
    TableName.JOB_POSTINGS: TableSchema(
        model=JobPostingsRow,
        table_name=TableName.JOB_POSTINGS.value,
        primary_keys=["url", "iteration"],
    ),
    TableName.EXTRACTED_REQUIREMENTS: TableSchema(
        model=ExtractedRequirementRow,
        table_name=TableName.EXTRACTED_REQUIREMENTS.value,
        primary_keys=[
            "url",
            "requirement_category_idx",
            "requirement_idx",
            "iteration",
            "version",
            "llm_provider",
        ],
    ),
    TableName.FLATTENED_REQUIREMENTS: TableSchema(
        model=FlattenedRequirementsRow,
        table_name=TableName.FLATTENED_REQUIREMENTS.value,
        primary_keys=[
            "url",
            "requirement_key",
            "iteration",
            "version",
        ],
    ),
    TableName.FLATTENED_RESPONSIBILITIES: TableSchema(
        model=FlattenedResponsibilitiesRow,
        table_name=TableName.FLATTENED_RESPONSIBILITIES.value,
        primary_keys=[
            "url",
            "responsibility_key",
            "iteration",
            "version",
        ],
    ),
    TableName.EDITED_RESPONSIBILITIES: TableSchema(
        model=EditedResponsibilitiesRow,
        table_name=TableName.EDITED_RESPONSIBILITIES.value,
        primary_keys=[
            "url",
            "responsibility_key",
            "iteration",
            "version",
            "llm_provider",
        ],
    ),
    TableName.SIMILARITY_METRICS: TableSchema(
        model=SimilarityMetricsRow,
        table_name=TableName.SIMILARITY_METRICS.value,
        primary_keys=[
            "url",
            "requirement_key",
            "responsibility_key",
            "iteration",
            "version",
            "llm_provider",
        ],
    ),
    TableName.PRUNED_RESPONSIBILITIES: TableSchema(
        model=PrunedResponsibilitiesRow,
        table_name=TableName.PRUNED_RESPONSIBILITIES.value,
        primary_keys=[
            "url",
            "responsibility_key",
            "iteration",
            "version",
        ],
    ),
}


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
