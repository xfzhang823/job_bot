"""
duckdb_ingestion_pipeline.py

This module orchestrates the ingestion of structured resume-job alignment data
into a DuckDB database.

It is organized around modular 'mini-pipelines' that handle different phases of
the processing pipeline.

Each mini-pipeline:
- Loads validated Pydantic models or DataFrames
- Applies schema alignment and metadata enrichment
- Deduplicates based on key fields
- Inserts aligned data into DuckDB tables

Supported phases:
- Preprocessing: Raw job postings, URLs, and extracted requirements
- Staging: Flattened resume responsibilities and job requirements
- Evaluation: Original and edited semantic similarity metrics
- Editing: LLM-optimized responsibilities
"""

from pathlib import Path
from typing import cast, Type, Callable, List
import logging
from enum import Enum
import pandas as pd
from pydantic import BaseModel
from db_io.db_transform import flatten_model_to_df, add_metadata
from db_io.db_insert import insert_df_dedup
from db_io.setup_duckdb import create_all_duckdb_tables
from db_io.pipeline_enums import PipelineStage, TableName
from db_io.db_schema_registry import DUCKDB_PRIMARY_KEYS
from models.resume_job_description_io_models import (
    JobPostingsBatch,
    JobPostingUrlsBatch,
    ExtractedRequirementsBatch,
    Responsibilities,
    Requirements,
)
from utils.pydantic_model_loaders_from_files import (
    load_job_postings_file_model,
    load_job_posting_urls_file_model,
    load_extracted_requirements_model,
    load_requirements_model,
    load_responsibilities_model,
    load_nested_responsibilities_model,
    load_similarity_metrics_model_from_csv,
)
from project_config import (
    JOB_POSTING_URLS_FILTERED_FILE,
    JOB_DESCRIPTIONS_JSON_FILE,
    JOB_REQUIREMENTS_JSON_FILE,
    RESPS_FILES_ITERATE_0_OPENAI_DIR,
    REQS_FILES_ITERATE_0_OPENAI_DIR,
    SIMILARITY_METRICS_ITERATE_0_OPENAI_DIR,
    SIMILARITY_METRICS_ITERATE_1_OPENAI_DIR,
    RESPS_FILES_ITERATE_1_OPENAI_DIR,
)

logger = logging.getLogger(__name__)

# * Config for mini pipelines
preprocessing_sources = [
    (
        TableName.JOB_URLS,
        JOB_POSTING_URLS_FILTERED_FILE,
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


def ingest_job_urls_file_pipeline():
    """Ingests `job_urls` table from a single JSON file."""
    table_name = TableName.JOB_URLS
    file_path = JOB_POSTING_URLS_FILTERED_FILE
    logger.info(f"📥 Ingesting job URLs from {file_path}")

    if not file_path.exists():
        logger.warning(f"⚠️ File not found: {file_path}")
        return

    model = load_job_posting_urls_file_model(file_path)
    if model is None:
        logger.warning("⚠️ Skipping job URLs due to model loading failure.")
        return

    df = flatten_model_to_df(
        model=model,
        table_name=table_name,
        source_file=file_path,
        stage=PipelineStage.JOB_URLS,
    )
    insert_df_dedup(df, table_name.value)
    logger.info("✅ job_urls ingestion complete.")


def ingest_job_postings_file_pipeline():
    """Ingests `job_postings` table from a single JSON file."""
    table_name = TableName.JOB_POSTINGS
    file_path = JOB_DESCRIPTIONS_JSON_FILE
    logger.info(f"📥 Ingesting job postings from {file_path}")

    if not file_path.exists():
        logger.warning(f"⚠️ File not found: {file_path}")
        return

    model = load_job_postings_file_model(file_path)
    if model is None:
        logger.warning("⚠️ Skipping job postings due to model loading failure.")
        return

    df = flatten_model_to_df(
        model=model,
        table_name=table_name,
        source_file=file_path,
        stage=PipelineStage.JOB_POSTINGS,
    )
    insert_df_dedup(df, table_name.value)
    logger.info("✅ job_postings ingestion complete.")


def ingest_extracted_requirements_file_pipeline():
    """Ingests `extracted_requirements` table from a single JSON file."""
    table_name = TableName.EXTRACTED_REQUIREMENTS
    file_path = JOB_REQUIREMENTS_JSON_FILE
    logger.info(f"📥 Ingesting extracted requirements from {file_path}")

    if not file_path.exists():
        logger.warning(f"⚠️ File not found: {file_path}")
        return

    model = load_extracted_requirements_model(file_path)
    if model is None:
        logger.warning(
            "⚠️ Skipping extracted requirements due to model loading failure."
        )
        return

    df = flatten_model_to_df(
        model=model,
        table_name=table_name,
        source_file=file_path,
        stage=PipelineStage.EXTRACTED_REQUIREMENTS,
    )

    # todo: debug; delete later
    # ✅ debug: Detect duplicated primary keys (within the DataFrame)
    table_key = table_name.value if isinstance(table_name, Enum) else table_name
    pk_fields = DUCKDB_PRIMARY_KEYS.get(table_key)

    missing_cols = [col for col in pk_fields if col not in df.columns]
    if missing_cols:
        logger.error(f"❌ DataFrame missing primary key columns: {missing_cols}")
    else:
        dupes = df[df.duplicated(subset=pk_fields, keep=False)]
        if not dupes.empty:
            duped_urls = dupes["url"].unique().tolist()
            logger.warning(
                f"⚠️ Detected {len(dupes)} duplicated rows based on PK {pk_fields} in table '{table_key}'. "
                f"Problematic URLs: {duped_urls}"
            )
    # todo: debut; delete later

    insert_df_dedup(df, table_name.value)
    logger.info("✅ extracted_requirements ingestion complete.")


# def preprocessing_db_ingestion_mini_pipeline():
#     """
#         Ingests raw preprocessing outputs into DuckDB.

#         This includes:
#         - `job_urls`: URLs of job postings to crawl
#         - `job_postings`: Scraped content of job descriptions
#         - `extracted_requirements`: LLM-extracted requirement categories

#         Each JSON file is loaded via Pydantic, flattened, aligned, and inserted into
#     its corresponding table.
#     """
#     logger.info("\U0001f680 Starting preprocessing db ingestion mini-pipeline")

#     for table_name, file_path, load_fn in preprocessing_sources:
#         logger.info(f"\U0001f4e5 Ingesting {table_name} from {file_path}")
#         if not file_path.exists():
#             logger.warning(f"\u26a0\ufe0f File not found: {file_path}")
#             continue

#         model = load_fn(file_path)
#         if model is None:
#             logger.warning(
#                 f"\u26a0\ufe0f Skipping {table_name} due to model loading failure."
#             )
#             continue

#         df = flatten_model_to_df(
#             model=model,
#             table_name=table_name,
#             source_file=file_path,
#             stage=PipelineStage.PREPROCESSING,
#         )

#         logger.debug(df.head(1))
#         logger.debug(df.columns)
#         logger.debug(df.dtypes)

#         insert_df_dedup(df, table_name.value)

#     logger.info("\u2705 Preprocessing db ingestion mini-pipeline complete.")


def staging_db_ingestion_mini_pipeline():
    """
    Ingests staging-phase outputs from iteration 0 into DuckDB.

    This includes:
    - `flattened_responsibilities`: Extracted resume bullets (flattened)
    - `flattened_requirements`: Job requirement categories (flattened)

    Processes each JSON file using validated models and flattener functions.
    """

    logger.info("\U0001f680 Starting staging db ingestion mini-pipeline")

    for table_name, dir_path, loader_fn in staging_sources:
        if not dir_path.exists():
            logger.warning(f"\u26a0\ufe0f Directory not found: {dir_path}")
            continue

        logger.info(
            f"\U0001f4c2 Scanning directory: {dir_path} for table: {table_name}"
        )

        for file_path in dir_path.glob("*.json"):
            logger.info(f"\U0001f4e5 Ingesting from {file_path.name} into {table_name}")
            model = loader_fn(file_path)
            if model is None:
                logger.warning(
                    f"\u26a0\ufe0f Skipping {file_path.name} due to model load failure."
                )
                continue

            df = flatten_model_to_df(
                model=model,
                table_name=table_name,
                source_file=file_path,
                stage=PipelineStage.STAGING,
            )

            logger.debug(df.head(1))
            logger.debug(df.columns)
            logger.debug(df.dtypes)

            insert_df_dedup(df, table_name.value)

    logger.info("\u2705 Staging db ingestion mini-pipeline complete.")


def evaluation_original_metrics_db_ingestion_mini_pipeline(
    source_dir: Path | str = SIMILARITY_METRICS_ITERATE_0_OPENAI_DIR,
):
    """
    Ingests original similarity metrics (iteration 0) into DuckDB.

    Each file is a CSV of semantic scores between resume and job requirement pairs,
    before any editing.
    Metadata such as version and timestamp is added conditionally based on
    the target schema.
    """

    logger.info(
        "\U0001f680 Starting original similarity metrics ingestion mini-pipeline"
    )

    table_name = TableName.SIMILARITY_METRICS.value
    dir_path = Path(source_dir) if isinstance(source_dir, str) else source_dir

    if not dir_path.exists():
        logger.warning(f"\u26a0\ufe0f Directory not found: {dir_path}")
        return

    for file_path in dir_path.glob("*.csv"):
        logger.info(f"\U0001f4e5 Ingesting similarity metrics from {file_path.name}")
        df = load_similarity_metrics_model_from_csv(file_path)

        if df is None:
            logger.warning(
                f"\u26a0\ufe0f Skipping {file_path.name} due to validation failure."
            )
            continue

        try:
            df = add_metadata(
                df=df,
                source_file=file_path,
                stage=PipelineStage.EVALUATION,
                table=TableName.SIMILARITY_METRICS,
                version="original",
                iteration=0,
            )
            insert_df_dedup(df, table_name)
        except Exception as e:
            logger.exception(f"❌ Failed to insert {file_path.name}: {e}")

        insert_df_dedup(df, table_name)

    logger.info("\u2705 Original similarity metrics ingestion complete.")


def evaluation_edited_metrics_db_ingestion_mini_pipeline(
    source_dir: Path | str = SIMILARITY_METRICS_ITERATE_1_OPENAI_DIR,
    llm_provider: str = "openai",
):
    """
    Ingests similarity metrics for LLM-edited responsibilities into DuckDB.

    These CSVs include re-scored alignment data after LLM-based editing.
    The pipeline appends the LLM provider name and version tag to ensure traceability.
    """
    logger.info("\U0001f680 Starting edited similarity metrics ingestion mini-pipeline")

    table_name = TableName.SIMILARITY_METRICS.value
    dir_path = Path(source_dir) if isinstance(source_dir, str) else source_dir

    if not dir_path.exists():
        logger.warning(f"\u26a0\ufe0f Directory not found: {dir_path}")
        return

    for file_path in dir_path.glob("*.csv"):
        logger.info(f"\U0001f4e5 Ingesting similarity metrics from {file_path.name}")
        df = load_similarity_metrics_model_from_csv(file_path)

        if df is None:
            logger.warning(
                f"\u26a0\ufe0f Skipping {file_path.name} due to validation failure."
            )
            continue

        try:
            df = add_metadata(
                df=df,
                source_file=file_path,
                stage=PipelineStage.REVALUATION,
                table=TableName.SIMILARITY_METRICS,
                version="edited",
                llm_provider=llm_provider,
                iteration=0,
            )
            insert_df_dedup(df, table_name)

        except Exception as e:
            logger.exception(f"❌ Failed to insert {file_path.name}: {e}")
    logger.info("\u2705 Edited similarity metrics ingestion complete.")


def edited_responsibilities_db_ingestion_mini_pipeline(
    source_dir: Path | str = RESPS_FILES_ITERATE_1_OPENAI_DIR,
    llm_provider: str = "openai",
):
    """
    Ingests LLM-generated edits to resume responsibilities into DuckDB.

    Each JSON file includes nested structures of rewritten content aligned to job requirements.
    Adds the provider (e.g., OpenAI) to indicate which model performed the editing.
    """

    logger.info("\U0001f680 Starting edited responsibilities ingestion mini-pipeline")

    table_name = TableName.EDITED_RESPONSIBILITIES
    dir_path = Path(source_dir) if isinstance(source_dir, str) else source_dir

    if not dir_path.exists():
        logger.warning(f"\u26a0\ufe0f Directory not found: {dir_path}")
        return

    for file_path in dir_path.glob("*.json"):
        logger.info(f"\U0001f4e5 Ingesting from {file_path.name} into {table_name}")
        model = load_nested_responsibilities_model(file_path)

        if model is None:
            logger.warning(
                f"\u26a0\ufe0f Skipping {file_path.name} due to model load failure."
            )
            continue

        df = flatten_model_to_df(
            model=model,
            table_name=table_name,
            source_file=file_path,
            stage=PipelineStage.REVALUATION,
            version="edited",
            llm_provider="openai",
        )
        df["llm_provider"] = llm_provider

        logger.debug(df.head(1))
        logger.debug(df.columns)
        logger.debug(df.dtypes)

        insert_df_dedup(df, table_name.value)

    logger.info("\u2705 Edited responsibilities ingestion mini-pipeline complete.")


def run_duckdb_ingestion_pipeline():
    """
    Main orchestrator function for DuckDB ingestion.

    Runs all ingestion stages in order:
    1. Creates all tables using predefined schema
    2. Inserts preprocessing JSON outputs
    3. Loads flattened responsibilities and requirements
    4. Ingests original similarity scores (pre-edit)
    5. Ingests edited responsibilities (LLM-optimized)
    6. Ingests updated similarity scores (post-edit)
    """
    logger.info("\U0001f3d7\ufe0f Creating all DuckDB tables...")
    create_all_duckdb_tables()
    logger.info("\u2705 DuckDB schema setup complete.")

    ingest_job_urls_file_pipeline()
    ingest_job_postings_file_pipeline()
    ingest_extracted_requirements_file_pipeline()
    # staging_db_ingestion_mini_pipeline()
    # evaluation_original_metrics_db_ingestion_mini_pipeline()
    # edited_responsibilities_db_ingestion_mini_pipeline()
    # evaluation_edited_metrics_db_ingestion_mini_pipeline()
