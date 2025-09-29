"""utils/pydantic_model_loaders_for_files.py"""

from pathlib import Path
import logging
from typing import Union, Optional
import json
import pandas as pd
from pydantic import ValidationError, TypeAdapter

from job_bot.models.resume_job_description_io_models import (
    JobPostingsBatch,
    JobPostingUrlMetadata,
    JobPostingUrlsBatch,
    ExtractedRequirementsBatch,
    Requirements,
    Responsibilities,
    NestedResponsibilities,
    SimilarityMetrics,
    JobFileMappings,
)
from job_bot.models.llm_response_models import JobSiteResponse, RequirementsResponse
from job_bot.utils.generic_utils import read_from_json_file

logger = logging.getLogger(__name__)


def load_job_file_mappings_model(
    mapping_file: Union[str, Path],
) -> Optional[JobFileMappings]:
    """
    Load job file mappings from a JSON file using the JobFileMappings model.

    Args:
        mapping_file (str | Path): Path to the JSON mapping file.

    Returns:
        Optional[JobFileMappings]: Job file mappings model or None if validation fails.
    """
    try:
        mapping_file = Path(mapping_file)
        file_mapping = read_from_json_file(mapping_file, key=None)

        if not isinstance(file_mapping, dict):
            logger.error(
                f"Mapping file {mapping_file} does not contain a valid JSON object."
            )
            return None

        job_file_mappings_model = JobFileMappings.model_validate(file_mapping)
        logger.info(f"Loaded and validated mapping file from {mapping_file}")
        return job_file_mappings_model

    except ValidationError as e:
        logger.error(f"Validation error in mapping file {mapping_file}: {e}")
        return None
    except FileNotFoundError as e:
        logger.error(f"Mapping file not found: {mapping_file}. Error: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding failed for file {mapping_file}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading mapping file {mapping_file}: {e}")
        return None


def load_job_postings_file_model(
    file_path: Union[str, Path],
) -> Optional[JobPostingsBatch]:
    """
    Load a validated JobPostingsFile model from a JSON file.

    Args:
        file_path (Union[str, Path]): Path to the job postings JSON file.

    Returns:
        Optional[JobPostingsFile]: Loaded and validated model or None on failure.
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path

    try:
        raw_data = read_from_json_file(file_path)
        validated = {}

        for k, v in raw_data.items():
            if not isinstance(v, dict):
                logger.warning(f"⚠️ Skipping {k} — value is not a dict: {v}")
                continue
            try:
                validated[k] = JobSiteResponse(**v)
            except ValidationError as e:
                logger.warning(f"⚠️ Validation error for {k}: {e}")
                continue
        return JobPostingsBatch(validated)

    except Exception as e:
        logger.error(f"Failed to load JobPostingsFile model from {file_path}: {e}")
        return None


def load_job_posting_urls_file_model(
    file_path: Union[str, Path],
) -> Optional[JobPostingUrlsBatch]:
    """
    Load a validated JobPostingUrlsFile model from a JSON file.

    Args:
        file_path (Union[str, Path]): Path to the job posting URLs JSON file.

    Returns:
        Optional[JobPostingUrlsFile]: Loaded and validated model or None on failure.
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path

    try:
        raw = read_from_json_file(file_path)
        validated = {
            url: (JobPostingUrlMetadata(**val) if isinstance(val, dict) else val)
            for url, val in raw.items()
        }
        return JobPostingUrlsBatch(validated)
    except Exception as e:
        logger.error(f"Failed to load JobPostingUrlsFile model from {file_path}: {e}")
        return None


def load_requirements_model(file_path: Union[str, Path]) -> Optional[Requirements]:
    """
    Load a validated Requirements model from a JSON file.

    Args:
        file_path (Union[str, Path]): Path to the job requirements JSON file.

    Returns:
        Optional[Requirements]: Loaded model or None on validation error.
    """
    try:
        raw = read_from_json_file(file_path)
        return Requirements(**raw)
    except ValidationError as e:
        logger.error(f"Validation error in Requirements file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load Requirements model from {file_path}: {e}")
        return None


def load_extracted_requirements_model(
    file_path: Union[str, Path],
) -> Optional[ExtractedRequirementsBatch]:
    """
    Load a validated ExtractedRequirementsFile model from a JSON file.

    Args:
        file_path (Union[str, Path]): Path to the extracted requirements JSON file.

    Returns:
        Optional[ExtractedRequirementsFile]: Loaded model or None on failure.
    """
    try:
        raw = read_from_json_file(file_path)

        if not isinstance(raw, dict):
            logger.error(f"{file_path} did not return a JSON object.")
            return None

        validated = {
            k: RequirementsResponse(**v) if isinstance(v, dict) else v
            for k, v in raw.items()
        }

        return ExtractedRequirementsBatch(validated)

    except ValidationError as e:
        logger.error(
            f"Validation error in ExtractedRequirementsFile file {file_path}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Failed to load ExtractedRequirementsFile model from {file_path}: {e}"
        )
        return None


def load_responsibilities_model(
    file_path: Union[str, Path],
) -> Optional[Responsibilities]:
    """
    Load a validated Responsibilities model from a JSON file.

    Args:
        file_path (Union[str, Path]): Path to the responsibilities JSON file.

    Returns:
        Optional[Responsibilities]: Loaded model or None on failure.
    """
    try:
        raw = read_from_json_file(file_path)
        return Responsibilities(**raw)
    except ValidationError as e:
        logger.error(f"Validation error in Responsibilities file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load Responsibilities model from {file_path}: {e}")
        return None


def load_nested_responsibilities_model(
    file_path: Union[str, Path],
) -> Optional[NestedResponsibilities]:
    """
    Load a validated NestedResponsibilities model from a JSON file.

    Args:
        file_path (Union[str, Path]): Path to the nested responsibilities JSON file.

    Returns:
        Optional[NestedResponsibilities]: Loaded model or None on failure.
    """
    try:
        raw = read_from_json_file(file_path)
        return NestedResponsibilities(**raw)
    except ValidationError as e:
        logger.error(
            f"Validation error in NestedResponsibilities file {file_path}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Failed to load NestedResponsibilities model from {file_path}: {e}"
        )
        return None


def load_similarity_metrics_model_from_csv(
    file_path: Union[str, Path],
) -> Optional[pd.DataFrame]:
    """
    Load and validate similarity metrics from a CSV file by checking required columns,
    performing light cleanup, and applying Pydantic model validation row-wise.
    """
    try:
        df: pd.DataFrame = pd.read_csv(file_path)
        logger.debug(f"Loaded {len(df)} rows from {file_path}")

        # Clean and coerce values
        def clean_value(v):
            try:
                if pd.isna(v) or (
                    isinstance(v, str) and v.strip() in {"", "N/A", "null"}
                ):
                    return None
                return v
            except Exception:
                return None

        df = df.apply(lambda col: col.map(clean_value))

        numeric_cols = [
            "bert_score_precision",
            "soft_similarity",
            "word_movers_distance",
            "deberta_entailment_score",
            "roberta_entailment_score",
            "scaled_bert_score_precision",
            "scaled_soft_similarity",
            "scaled_word_movers_distance",
            "scaled_deberta_entailment_score",
            "scaled_roberta_entailment_score",
            "composite_score",
            "pca_score",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Validate rows using the Pydantic model
        adapter = TypeAdapter(list[SimilarityMetrics])
        validated = adapter.validate_python(df.to_dict(orient="records"))
        logger.info(
            f"✅ Validated {len(validated)} similarity metrics from {file_path}"
        )
        return pd.DataFrame([row.model_dump() for row in validated])

    except Exception as e:
        logger.error(f"❌ Failed to load similarity metrics CSV from {file_path}: {e}")
        return None
