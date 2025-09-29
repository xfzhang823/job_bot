"""
db_io/file_ingestion.py

High-level ingestion entrypoints for loading JSON/CSV files into DuckDB.

Each function validates a file, loads it into a Pydantic model, flattens to
a DataFrame, and inserts into the correct DuckDB table with metadata fields
(stage, version, provider, iteration). Deduplication is handled automatically.

Examples
--------
>>> from job_bot.db_io import file_ingestion
>>> file_ingestion.ingest_job_urls_file()
# Loads JOB_POSTING_URLS_FILE into the `job_urls` table

>>> from pathlib import Path
>>> file_ingestion.ingest_flattened_responsibilities_file(
...     Path("data/resume/flattened_responsibilities.json"),
...     mode="replace",
... )
# Refreshes the `flattened_responsibilities` table with the latest resume
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Literal, cast
from pydantic import BaseModel

from job_bot.db_io.db_transform import add_metadata
from job_bot.db_io.flatten_and_rehydrate import flatten_model_to_df
from job_bot.db_io.db_inserters import insert_df_with_config
from job_bot.db_io.pipeline_enums import TableName, LLMProvider, Version
from job_bot.db_io.db_schema_registry import DUCKDB_SCHEMA_REGISTRY

from job_bot.models.model_type import ModelType
from job_bot.utils.pydantic_model_loaders_for_files import (
    load_job_postings_file_model,
    load_job_posting_urls_file_model,
    load_extracted_requirements_model,
    load_requirements_model,
    load_responsibilities_model,
    load_nested_responsibilities_model,
    load_similarity_metrics_model_from_csv,
)
from job_bot.config.project_config import (
    JOB_POSTING_URLS_FILE,
    JOB_DESCRIPTIONS_JSON_FILE,
    JOB_REQUIREMENTS_JSON_FILE,
    RESPS_FILES_ITERATE_0_OPENAI_DIR,
    REQS_FILES_ITERATE_0_OPENAI_DIR,
    SIMILARITY_METRICS_ITERATE_0_OPENAI_DIR,
    SIMILARITY_METRICS_ITERATE_1_OPENAI_DIR,
    RESPS_FILES_ITERATE_1_OPENAI_DIR,
    RESPS_FILES_ITERATE_1_ANTHROPIC_DIR,
    SIMILARITY_METRICS_ITERATE_1_ANTHROPIC_DIR,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ingest_single_file",
    "ingest_flattened_json_file",
    "ingest_job_urls_file",
    "ingest_job_postings_file",
    "ingest_extracted_requirements_file",
    "ingest_flattened_requirements_file",
    "ingest_flattened_responsibilities_file",
    "ingest_similarity_metrics_file",
    "ingest_edited_responsibilities_file",
]

logger = logging.getLogger(__name__)

# * Config for mini pipelines
preprocessing_sources = [
    (
        TableName.JOB_URLS,
        JOB_POSTING_URLS_FILE,
        load_job_posting_urls_file_model,
    ),
    (TableName.JOB_POSTINGS, JOB_DESCRIPTIONS_JSON_FILE, load_job_postings_file_model),
    (
        TableName.EXTRACTED_REQUIREMENTS,
        JOB_REQUIREMENTS_JSON_FILE,
        load_extracted_requirements_model,
    ),
]

staging_sources = [
    (
        TableName.FLATTENED_RESPONSIBILITIES,
        RESPS_FILES_ITERATE_0_OPENAI_DIR,
        load_responsibilities_model,
    ),
    (
        TableName.FLATTENED_REQUIREMENTS,
        REQS_FILES_ITERATE_0_OPENAI_DIR,
        load_requirements_model,
    ),
]


def ingest_single_file(
    table_name: TableName,
    source_file: Path,
    loader_fn: Callable[[Path], BaseModel | None],
    *,
    mode: Literal["append", "replace"] = "replace",
    iteration: int | None = None,
    version: Version | str | None = None,
    llm_provider: LLMProvider | str | None = None,
    model_id: str | None = None,
) -> None:
    """
    Ingest one validated file into a DuckDB table.

    This function:
      1. Validates the file path and loads a Pydantic model via `loader_fn`.
      2. Flattens the model into a DataFrame and stamps only the metadata
         defined for the target table.
      3. Optionally deletes existing rows for the same `source_file` if
         mode='replace' and the table owns a `source_file` column.
      4. Inserts the DataFrame into DuckDB using the standard insert helper
         (`insert_df_with_config`), which handles schema alignment and deduplication.

    Args:
        table_name (TableName): Target DuckDB table.
        source_file (Path): Path to the input file to ingest.
        loader_fn (Callable[[Path], BaseModel | None]): Function that loads
            and validates the file into a Pydantic model.
        mode (Literal["append", "replace"], optional): Insert mode.
            - "append": Adds rows without deleting existing ones.
            - "replace": Deletes prior rows for the same `source_file` before insert
              (if the table tracks `source_file`).
            Defaults to "replace".
        iteration (int, optional): Iteration stamp for reruns or audits.
            If omitted and the table requires it, defaults to 0.
        version (Version | str, optional): Version tag to stamp if supported
            by the table (e.g., "original", "edited").
        llm_provider (LLMProvider | str, optional): LLM provider name for
            artifact tables that track provenance (e.g., "openai", "anthropic").
        model_id (str, optional): LLM model identifier, required for
            LLM-artifact tables such as `edited_responsibilities`.

    Returns:
        None
    """
    logger.info("📥 Ingesting %s from %s", table_name.value, source_file)

    if not source_file.exists():
        logger.warning("⚠️ File not found: %s", source_file)
        return

    model_raw = loader_fn(source_file)
    if model_raw is None:
        logger.warning("⚠️ Skipping %s — model load failed.", table_name.value)
        return

    model = cast(ModelType, model_raw)

    # Enforce LLM provenance where required
    if table_name in {TableName.EDITED_RESPONSIBILITIES}:  # add more if needed
        if llm_provider is None or model_id is None:
            raise ValueError(
                f"{table_name} requires llm_provider and model_id (LLM artifact)."
            )

    # Flatten + table-aware metadata (let add_metadata default iteration when needed)
    df = flatten_model_to_df(
        model=model,
        table_name=table_name,
        source_file=source_file,
        iteration=iteration,  # ← don't force 0
        version=version,
        llm_provider=llm_provider,
        model_id=model_id,
    )

    # 🔍 Preview DF details
    logger.info("🔎 DataFrame for %s → %d rows, %d cols", table_name.value, *df.shape)
    logger.debug("First few rows:\n%s", df.head(3).to_string(index=False))
    logger.debug("Column order:\n%s", list(df.columns))

    # Upsert / dedup insert (align/order/PK merge handled inside)
    insert_df_with_config(df, table_name, mode=mode)


def ingest_flattened_json_file(
    source_file: Path,
    table_name: TableName,
    loader_fn: Callable[[Path], ModelType | None],
    *,
    mode: Literal["append", "replace"] = "replace",
    iteration: int | None = None,
    version: Version | str | None = None,
    llm_provider: LLMProvider | str | None = None,
    model_id: str | None = None,
) -> None:
    """
    Ingest a flattened JSON file into a DuckDB table (table-aware metadata).

    Steps
    -----
    1) Validate path & load a Pydantic model via `loader_fn`.
    2) Flatten to a DataFrame and stamp only the metadata owned by `table_name`
       (source_file, iteration, version/provider/model_id if applicable).
    3) Insert via `insert_df_with_config` (handles schema align + dedup/upsert).
       If mode='replace', the insert helper may delete prior rows for this
       source_file when the table tracks it.

    Args:
        source_file (Path): Path to the flattened JSON file.
        table_name (TableName): Target DuckDB table enum.
        loader_fn (Callable[[Path], ModelType | None]): Loader returning
            a validated model.
        mode (Literal["append","replace"], optional): Insert behavior.
            Defaults to "replace".
        iteration (int | None): Iteration stamp; if omitted and the table
            requires it, defaults to 0 inside the metadata helper.
        version (Version | str | None, optional): Version tag to stamp
            if supported by the table.
        llm_provider (LLMProvider | str | None, optional):
            LLM provider for artifact tables.
        model_id (str | None, optional): Model identifier for artifact tables.

    Notes:
        • No `stage` or timestamps here (FSM/DDL handle those).
        • For LLM-artifact tables (e.g., EDITED_RESPONSIBILITIES), provide
          both `llm_provider` and `model_id`.
    """
    logger.info("📥 Ingesting %s from %s", table_name.value, source_file)

    if not source_file.exists():
        logger.warning("⚠️ File not found: %s", source_file)
        return

    model = loader_fn(source_file)
    if model is None:
        logger.warning("⚠️ Skipping %s — model load failed.", source_file.name)
        return

    # Enforce LLM provenance where required
    if table_name in {TableName.EDITED_RESPONSIBILITIES}:  # extend as needed
        if llm_provider is None or model_id is None:
            raise ValueError(
                f"{table_name} requires llm_provider and model_id (LLM artifact table)."
            )

    # Flatten + table-aware metadata (add_metadata is called inside)
    df = flatten_model_to_df(
        model=cast(ModelType, model),
        table_name=table_name,
        source_file=source_file,
        iteration=iteration,  # None → defaulted to 0 if table owns it
        version=version,
        llm_provider=llm_provider,
        model_id=model_id,
    )

    # Upsert / dedup insert (align/order/PK merge handled inside)
    insert_df_with_config(df, table_name, mode=mode)
    logger.info(
        "✅ Inserted %s from %s (%d rows).", table_name.value, source_file.name, len(df)
    )


# Safer defaults (append) for pipelines
def ingest_job_urls_file(
    mode: Literal["append", "replace"] = "append",
) -> None:
    """
    Ingest job_postings JSON into DuckDB.

    Ensures scraped or cached job descriptions are loaded as structured rows.
    """
    ingest_single_file(
        table_name=TableName.JOB_URLS,
        source_file=JOB_POSTING_URLS_FILE,
        loader_fn=load_job_posting_urls_file_model,
        mode=mode,
    )


# Very unlikely need to do this but keep as an option!
# Current file ETL pipeline does not have separate job_postings files.
def ingest_job_postings_file(
    source_file: Path | str = JOB_DESCRIPTIONS_JSON_FILE,
    *,
    llm_provider: LLMProvider | str,
    model_id: str,
    iteration: int | None = None,
    mode: Literal["append", "replace"] = "append",
) -> None:
    """
    Ingest `job_postings` JSON into DuckDB with LLM provenance.

    Args:
        source_file: Path to job postings JSON (defaults to configured file).
        llm_provider: LLM provider used to parse/enrich postings (e.g., "openai").
        model_id: Model identifier (e.g., "gpt-4.1-mini").
        iteration: Optional iteration stamp (defaults to 0 if table requires it).
        mode: "append" keeps existing; "replace" overwrites matching rows.
    """
    # Optional: assert schema actually owns these fields
    schema = DUCKDB_SCHEMA_REGISTRY[TableName.JOB_POSTINGS]
    needed = {"llm_provider", "model_id"}
    if not needed.issubset(set(schema.metadata_fields)):
        logger.warning(
            "Schema for JOB_POSTINGS missing %s; values will be ignored.",
            needed - set(schema.metadata_fields),
        )

    ingest_single_file(
        table_name=TableName.JOB_POSTINGS,
        source_file=Path(source_file),
        loader_fn=load_job_postings_file_model,  # type: ignore[arg-type]
        iteration=iteration,
        llm_provider=llm_provider,
        model_id=model_id,
        mode=mode,
    )


# Very unlikely need to do this but keep as an option!
# Current file ETL pipeline does not have separate extracted_requirements files.
def ingest_extracted_requirements_file(
    source_file: Path | str,
    *,
    llm_provider: LLMProvider | str,
    model_id: str,
    iteration: int | None = None,
    mode: Literal["append", "replace"] = "append",
) -> None:
    """
    Ingest `extracted_requirements` JSON into DuckDB with LLM provenance.

    Args:
        source_file: Path to extracted requirements JSON.
        llm_provider: LLM provider used for extraction (e.g., "openai").
        model_id: Model identifier (e.g., "gpt-4o-mini").
        iteration: Optional iteration stamp (defaults to 0 if table requires it).
        mode: "append" keeps existing; "replace" overwrites matching rows.
    """
    # Optional: assert schema owns the metadata fields
    schema = DUCKDB_SCHEMA_REGISTRY[TableName.EXTRACTED_REQUIREMENTS]
    needed = {"llm_provider", "model_id"}
    if not needed.issubset(set(schema.metadata_fields)):
        logger.warning(
            "Schema for EXTRACTED_REQUIREMENTS missing %s; values will be ignored.",
            needed - set(schema.metadata_fields),
        )

    ingest_single_file(
        table_name=TableName.EXTRACTED_REQUIREMENTS,
        source_file=Path(source_file),
        loader_fn=load_extracted_requirements_model,  # type: ignore[arg-type]
        iteration=iteration,
        llm_provider=llm_provider,
        model_id=model_id,
        mode=mode,
    )


def ingest_flattened_requirements_file(
    source_file: str | Path,
    *,
    iteration: int | None = None,
    mode: Literal["append", "replace"] = "replace",
) -> None:
    """
    Ingest a flattened requirements JSON file into DuckDB.

    Args:
        file_path: Path to the flattened JSON file.
        iteration: Iteration stamp (defaults to 0 if required by schema).
        mode: Insert mode — 'append' keeps existing, 'replace' overwrites.
    """
    ingest_single_file(
        table_name=TableName.FLATTENED_RESPONSIBILITIES,
        source_file=Path(source_file),
        loader_fn=load_responsibilities_model,
        iteration=iteration,
        mode=mode,
    )


def ingest_flattened_responsibilities_file(
    source_file: str | Path,
    *,
    iteration: int | None = None,
    mode: Literal["append", "replace"] = "replace",
) -> None:
    """
    Ingest a flattened responsibilities JSON file into DuckDB.

    Args:
        file_path: Path to the flattened JSON file.
        iteration: Iteration stamp (defaults to 0 if required by schema).
        mode: Insert mode — 'append' keeps existing, 'replace' overwrites.
    """
    ingest_single_file(
        table_name=TableName.FLATTENED_RESPONSIBILITIES,
        source_file=Path(source_file),
        loader_fn=load_responsibilities_model,
        iteration=iteration,
        mode=mode,
    )


def ingest_similarity_metrics_file(
    source_file: Path,
    *,
    version: Version | str,
    iteration: int | None = None,
    mode: Literal["append", "replace"] = "replace",
) -> None:
    """
    Ingest a similarity metrics CSV into DuckDB (table-aware metadata).

    Notes:
      - Stamps `version` and (optionally) `similarity_backend` / `nli_backend`
        if those columns exist in the schema.
      - Does NOT stamp llm_provider/model_id (metrics aren’t LLM artifacts).
    """
    logger.info("📊 Ingesting similarity metrics from %s", source_file.name)

    df = load_similarity_metrics_model_from_csv(source_file)
    if df is None:
        logger.warning("⚠️ Skipping %s due to validation failure.", source_file.name)
        return

    # Table-aware metadata (source_file, iteration, version if owned)
    df = add_metadata(
        df=df,
        table=TableName.SIMILARITY_METRICS,
        source_file=source_file,
        iteration=iteration,
        version=version,
    )

    # # Stamp backends only if the schema defines them
    # schema = DUCKDB_SCHEMA_REGISTRY[TableName.SIMILARITY_METRICS]

    insert_df_with_config(df=df, table_name=TableName.SIMILARITY_METRICS, mode=mode)
    logger.info("✅ Inserted metrics from %s (%d rows).", source_file.name, len(df))


def ingest_edited_responsibilities_file(
    source_file: Path,
    *,
    llm_provider: LLMProvider | str,
    model_id: str,
    iteration: int | None = None,
    mode: Literal["append", "replace"] = "replace",
) -> None:
    """
    Ingest a single LLM-edited responsibilities JSON file into DuckDB.

    Notes:
      - Requires `llm_provider` and `model_id` (LLM provenance).
      - No `version` or `stage` stamping here; table-aware metadata only.
    """
    logger.info("📝 Ingesting edited responsibilities from %s", source_file.name)

    model = load_nested_responsibilities_model(source_file)
    if model is None:
        logger.warning("⚠️ Skipping %s due to validation failure.", source_file.name)
        return

    # Flatten + table-aware metadata (this will add source_file/iteration;
    # llm_provider/model_id are passed through and only stamped if the table owns them)
    df = flatten_model_to_df(
        model=model,
        table_name=TableName.EDITED_RESPONSIBILITIES,
        source_file=source_file,
        iteration=iteration,
        llm_provider=llm_provider,
        model_id=model_id,
    )

    logger.info("Preview before insert:\n%s", df.head().to_string(index=False))
    insert_df_with_config(
        df=df, table_name=TableName.EDITED_RESPONSIBILITIES, mode=mode
    )
    logger.info(
        "✅ Inserted edited responsibilities from %s (%d rows).",
        source_file.name,
        len(df),
    )
