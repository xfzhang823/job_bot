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
from db_io.pipeline_enums import PipelineStage, TableName, LLMProvider, Version

# from db_io.db_schema_registry import DUCKDB_PRIMARY_KEYS
from models.resume_job_description_io_models import (
    JobPostingsBatch,
    JobPostingUrlsBatch,
    ExtractedRequirementsBatch,
    Responsibilities,
    Requirements,
)
from models.model_type import ModelType

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


def ingest_single_file(
    table_name: TableName,
    file_path: Path,
    loader_fn: Callable[[Path], BaseModel | None],
    stage: PipelineStage,
):
    """
    Ingests a single validated JSON file into a DuckDB table.

    This function handles common ingestion logic for files backed by Pydantic models.

    Steps:
    1. Loads the file via `loader_fn`, which returns a validated Pydantic model
       (or None on failure).
    2. If valid, casts the model to `ModelType` for compatibility with the flattening
    dispatcher.
    3. Uses `flatten_model_to_df()` to convert the model to a DataFrame and
       append metadata (e.g. source_file, stage, timestamp).
    4. Inserts the result into DuckDB using `insert_df_dedup()` with deduplication
       and schema alignment.

    Args:
        table_name (TableName): Target DuckDB table.
        file_path (Path): Path to the JSON file.
        loader_fn (Callable): Loader function that returns a Pydantic model or None.
        stage (PipelineStage): Enum indicating the processing stage for metadata.
    """
    logger.info(f"📥 Ingesting {table_name.value} from {file_path}")

    if not file_path.exists():
        logger.warning(f"⚠️ File not found: {file_path}")
        return

    model_raw = loader_fn(file_path)
    if model_raw is None:
        logger.warning(f"⚠️ Skipping {table_name.value} due to model load failure.")
        return

    # Cast to ModelType
    model = cast(ModelType, model_raw)

    df = flatten_model_to_df(
        model=model,
        table_name=table_name,
        source_file=file_path,
        stage=stage,
    )
    insert_df_dedup(df, table_name.value)

    logger.info(f"✅ {table_name.value} ingestion complete.")


def ingest_flattened_json_file(
    file_path: Path,
    table_name: TableName,
    loader_fn: Callable[[Path], ModelType | None],
    stage: PipelineStage,
    version: Version = Version.ORIGINAL,
    llm_provider: LLMProvider = LLMProvider.OPENAI,
    iteration: int = 0,
):
    """
    Loads and inserts a single flattened JSON file into DuckDB with full metadata.
    """
    logger.info(f"📥 Ingesting {table_name.value} from {file_path.name}")

    model = loader_fn(file_path)
    if model is None:
        logger.warning(f"⚠️ Skipping {file_path.name} due to validation failure.")
        return

    df = flatten_model_to_df(
        model=model,
        table_name=table_name,
        source_file=file_path,
        stage=stage,
    )

    df["version"] = version.value
    df["llm_provider"] = llm_provider.value
    df["iteration"] = iteration

    insert_df_dedup(df, table_name.value)
    logger.info(f"✅ Inserted {table_name.value} from {file_path.name}")


def ingest_job_urls_file_pipeline():
    ingest_single_file(
        table_name=TableName.JOB_URLS,
        file_path=JOB_POSTING_URLS_FILTERED_FILE,
        loader_fn=load_job_posting_urls_file_model,
        stage=PipelineStage.JOB_URLS,
    )


def ingest_job_postings_file_pipeline():
    ingest_single_file(
        table_name=TableName.JOB_POSTINGS,
        file_path=JOB_DESCRIPTIONS_JSON_FILE,
        loader_fn=load_job_postings_file_model,  # type: ignore[arg-type]
        stage=PipelineStage.JOB_POSTINGS,
    )


def ingest_extracted_requirements_file_pipeline():
    ingest_single_file(
        table_name=TableName.EXTRACTED_REQUIREMENTS,
        file_path=JOB_REQUIREMENTS_JSON_FILE,
        loader_fn=load_extracted_requirements_model,  # type: ignore[arg-type]
        stage=PipelineStage.EXTRACTED_REQUIREMENTS,
    )


def ingest_flattened_requirements_file(file_path: Path):
    ingest_flattened_json_file(
        file_path=file_path,
        table_name=TableName.FLATTENED_REQUIREMENTS,
        loader_fn=load_requirements_model,
        stage=PipelineStage.FLATTENED_REQUIREMENTS,
        version=Version.ORIGINAL,
        llm_provider=LLMProvider.OPENAI,
        iteration=0,
    )


def ingest_flattened_responsibilities_file(file_path: Path):
    ingest_flattened_json_file(
        file_path=file_path,
        table_name=TableName.FLATTENED_RESPONSIBILITIES,
        loader_fn=load_responsibilities_model,
        stage=PipelineStage.FLATTENED_RESPONSIBILITIES,
        version=Version.ORIGINAL,
        llm_provider=LLMProvider.OPENAI,
        iteration=0,
    )


def ingest_similarity_metrics_file(
    file_path: Path,
    version: Version,
    stage: PipelineStage,
    llm_provider: LLMProvider,
    iteration: int = 0,
):
    """
    Loads and inserts a similarity metrics CSV into DuckDB with metadata.
    """
    logger.info(f"📊 Ingesting similarity metrics from {file_path.name}")

    df = load_similarity_metrics_model_from_csv(file_path)
    if df is None:
        logger.warning(f"⚠️ Skipping {file_path.name} due to validation failure.")
        return

    df = add_metadata(
        df=df,
        file_path=file_path,
        stage=stage,
        table=TableName.SIMILARITY_METRICS,
        version=version,
        llm_provider=llm_provider,
    )
    insert_df_dedup(df, TableName.SIMILARITY_METRICS.value)
    logger.info(f"✅ Inserted metrics from {file_path.name}")


def ingest_edited_responsibilities_file(
    file_path: Path,
    llm_provider: LLMProvider,
    iteration: int = 0,
):
    """
    Ingests a single LLM-edited responsibilities JSON file into the DuckDB database.

    This function is used during the re-evaluation stage of the pipeline, where resume
    responsibilities have been rewritten or optimized by an LLM (e.g., OpenAI or Anthropic)
    and saved in nested JSON format.

    Steps:
    1. Loads the file using `load_nested_responsibilities_model()` to validate its structure
       as a `NestedResponsibilities` Pydantic model.
    2. Flattens the model using `flatten_model_to_df()` into a tabular DataFrame suitable
       for DuckDB, and tags it with the `EDITED_RESPONSIBILITIES` stage.
    3. Adds metadata fields:
        - `version = Version.EDITED`
        - `llm_provider = the provider used for editing`
        - `iteration = the re-evaluation round (default = 0)`
    4. Inserts the DataFrame into the `edited_responsibilities` DuckDB table using
       `insert_df_dedup()`, which handles deduplication and schema enforcement.

    Args:
        file_path (Path): Path to the edited responsibilities JSON file.
        llm_provider (LLMProvider): The LLM provider responsible for generating the edits.
        iteration (int): The pipeline iteration number, defaults to 0.

    Raises:
        Logs a warning and skips insertion if the file fails validation.
    """

    logger.info(f"📝 Ingesting edited responsibilities from {file_path.name}")

    model = load_nested_responsibilities_model(file_path)
    if model is None:
        logger.warning(f"⚠️ Skipping {file_path.name} due to validation failure.")
        return

    df = flatten_model_to_df(
        model=model,
        table_name=TableName.EDITED_RESPONSIBILITIES,
        source_file=file_path,
        stage=PipelineStage.EDITED_RESPONSIBILITIES,
    )
    df["version"] = Version.EDITED.value
    df["llm_provider"] = llm_provider.value
    df["iteration"] = iteration

    insert_df_dedup(df, TableName.EDITED_RESPONSIBILITIES.value)
    logger.info(f"✅ Inserted edited responsibilities from {file_path.name}")


def run_duckdb_ingestion_pipeline():
    """
    Main orchestrator for DuckDB ingestion.

    Ingests all structured outputs from the resume-job alignment pipeline, including:
    - Preprocessing outputs: job URLs, job postings, extracted requirements
    - Staging outputs: flattened responsibilities and requirements (iteration 0)
    - Evaluation outputs: similarity metrics (original and edited)
    - Editing outputs: LLM-optimized responsibilities (iteration 1)

    This function assumes all source files exist in their expected paths
    as defined in `project_config.py`.
    """
    logger.info("🏗️ Creating DuckDB tables...")
    create_all_duckdb_tables()
    logger.info("✅ DuckDB schema setup complete.")

    # 🔹 Preprocessing (single-file tables)
    ingest_job_urls_file_pipeline()
    ingest_job_postings_file_pipeline()
    ingest_extracted_requirements_file_pipeline()

    # 🔹 Flattened requirements & responsibilities (iteration 0)
    for file_path in REQS_FILES_ITERATE_0_OPENAI_DIR.glob("*.json"):
        ingest_flattened_requirements_file(file_path)

    for file_path in RESPS_FILES_ITERATE_0_OPENAI_DIR.glob("*.json"):
        ingest_flattened_responsibilities_file(file_path)

    # 🔹 Original similarity metrics (iteration 0)
    for file_path in SIMILARITY_METRICS_ITERATE_0_OPENAI_DIR.glob("*.csv"):
        ingest_similarity_metrics_file(
            file_path=file_path,
            version=Version.ORIGINAL,
            stage=PipelineStage.SIM_METRICS_EVAL,
            llm_provider=LLMProvider.OPENAI,
            iteration=0,
        )

    # 🔹 Edited responsibilities (iteration 1)
    for file_path in RESPS_FILES_ITERATE_1_OPENAI_DIR.glob("*.json"):
        ingest_edited_responsibilities_file(
            file_path=file_path,
            llm_provider=LLMProvider.OPENAI,
            iteration=0,
        )

    # 🔹 Edited similarity metrics (iteration 1)
    for file_path in SIMILARITY_METRICS_ITERATE_1_OPENAI_DIR.glob("*.csv"):
        ingest_similarity_metrics_file(
            file_path=file_path,
            version=Version.EDITED,
            stage=PipelineStage.SIM_METRICS_REVAL,
            llm_provider=LLMProvider.OPENAI,
            iteration=0,
        )

    logger.info("🏁 DuckDB ingestion pipeline complete.")
