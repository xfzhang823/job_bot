"""
Module: flatten_and_rehydrate.py
Author: Xiao-Fei Zhang

This module provides utilities to flatten and rehydrate complex nested data structures
used in a resume-job alignment pipeline. It supports:

- Job postings and extracted requirements
- Flattened, pruned, and edited responsibilities
- Pydantic-validated models for both requirements and responsibilities
- Conversion between JSON-like dicts and tabular formats (pandas DataFrames)
- Logging and error-handling for robust pipeline integration

The goal is to allow all pipeline stages to work with relational tables (like DuckDB)
while preserving the ability to round-trip the data back to its original nested form.

---

Includes:

- `flatten_keyed_dict` / `unflatten_to_keyed_dict`: Generic utilities for converting
dict-of-dicts to DataFrame
- `flatten_job_postings_wide` / `rehydrate_job_postings_wide`
- `flatten_extracted_requirements` / `rehydrate_extracted_requirements`
- `flatten_responsibilities` / `rehydrate_responsibilities`
- `flatten_pruned_responsibilities` / `rehydrate_pruned_responsibilities`
- `flatten_requirements_model`: For Pydantic `Requirements` model
- `flatten_nested_responsibilities_model`: For Pydantic `NestedResponsibilities` model

Each function includes logging, validation, and structured keys to support robust
database integration and round-trip JSON transforms.

todo: leave the table data validation with table specific models for rehydrate functions
todo: for now, because:
todo: The ingestion pipeline already adds metadata and aligns columns using align_df_with_schema()
todo: The risk of schema mismatch is low if ingestion is the only entry point into the DBit's.
todo: The risk of schema mismatch is low.
todo: evisit this when batch validation, QA reporting, dashboard or audit tooling,
todo: writing comprehensive unit tests
"""

# Imports
from typing import Any, cast, Dict, List, Union
import json
import logging
import pandas as pd
from pydantic import HttpUrl, ValidationError
from models.resume_job_description_io_models import (
    NestedResponsibilities,
    Requirements,
    Responsibilities,
    JobPostingUrlMetadata,
    JobPostingUrlsBatch,
    JobPostingsBatch,
    ExtractedRequirementsBatch,
)
from models.llm_response_models import (
    JobSiteResponse,
    JobSiteData,
    RequirementsResponse,
    NestedRequirements,
)

# Set up logger for this module
logger = logging.getLogger(__name__)


def flatten_keyed_dict(
    d: Dict[str, Dict[str, Any]], key_col: str = "url"
) -> pd.DataFrame:
    """
    Converts a dict-of-dicts into a flat DataFrame with the outer key as a new column.

    Args:
        d (dict): A dictionary where each value is a dictionary, and the key (e.g. URL)
                  should be lifted into its own column.
        key_col (str): The name of the column to store the outer key.

    Returns:
        pd.DataFrame: Flattened data.
    """
    return pd.DataFrame([{key_col: k, **v} for k, v in d.items()])


def unflatten_to_keyed_dict(
    df: pd.DataFrame, key_col: str = "url"
) -> Dict[str, Dict[str, Any]]:
    """
    Converts a flat DataFrame back into a dict-of-dicts using a column as the key.

    Args:
        df (pd.DataFrame): The DataFrame to convert.
        key_col (str): The column to use as the outer key.

    Returns:
        dict: Nested dictionary.
    """
    return {
        row[key_col]: row.drop(labels=[key_col]).to_dict() for _, row in df.iterrows()
    }


# * ✔️ Jobposting Urls (central registry to keep records of all job posting URLs)
def flatten_job_urls_to_table(model: JobPostingUrlsBatch) -> pd.DataFrame:
    """
    Flattens a validated JobPostingUrlsFile model into a flat DataFrame
    suitable for DuckDB insertion.

    Each row corresponds to one job posting URL entry, containing:
    - url (str)
    - company (str)
    - job_title (str)

    Args:
        model (JobPostingUrlsFile): The root model with URL-keyed
        job posting metadata.

    Returns:
        pd.DataFrame: Flattened representation of job posting URLs.
    """
    rows: List[Dict[str, Any]] = []

    for url_key, metadata in model.root.items():
        try:
            rows.append(
                {
                    "url": str(metadata.url),  # even if it's an HttpUrl, convert to str
                    "company": metadata.company,
                    "job_title": metadata.job_title,
                }
            )
        except ValidationError as e:
            logger.warning(f"Validation error for job URL {url_key}: {e}")
        except Exception as e:
            logger.error(f"Error processing job URL {url_key}: {e}", exc_info=True)

    logger.info(f"✅ Flattened {len(rows)} job posting URLs.")
    return pd.DataFrame(rows)


def rehydrate_job_urls_from_table(df: pd.DataFrame) -> JobPostingUrlsBatch:
    """
    Rehydrates a DataFrame from the `job_urls` table into
    a JobPostingUrlsFile model.

    Args:
        df (pd.DataFrame): Flattened table containing 'url', 'company',
        and 'job_title' columns.

    Returns:
        JobPostingUrlsFile: Root model with validated mapping of URL -> metadata.
    """
    if df.empty:
        return JobPostingUrlsBatch({})

    # * Keep the most recent entry per URL (dedup just to be safe!)
    df = df.sort_values("timestamp").drop_duplicates("url", keep="last")

    validated: dict[str, JobPostingUrlMetadata] = {}

    for row in df.to_dict(orient="records"):
        try:
            metadata = JobPostingUrlMetadata(**dict(row))  # type: ignore
            validated[str(metadata.url)] = metadata  # use URL as key
        except Exception as e:
            continue  # optionally log or raise

    return JobPostingUrlsBatch(validated)


# * ✔️ Jobpostings
def flatten_job_postings_to_table(model: JobPostingsBatch) -> pd.DataFrame:
    """
    Flattens a validated JobPostingsFile model into a wide-format DataFrame,
    where each row corresponds to a job posting.

    This function preserves structure while transforming content into a
    tabular format suitable for DuckDB ingestion.

    * Notes:
    - The 'content' field is stored as a JSON string.
    - Invalid entries are skipped with a warning.

    Each row corresponds to a job posting, and the 'content' field is stored as
    a JSON string in a single column rather than being split into multiple rows.

    Records that do not match the expected format are skipped with a warning.

    Args:
        data (Dict[str, Dict[str, Any]]): A dictionary where each key is a job URL
        and the value is a nested dictionary with job posting metadata and content.

    Returns:
        pd.DataFrame: A DataFrame with columns corresponding to job metadata
        and content, suitable for insertion into a DuckDB 'job_postings' table.
    """
    rows: List[Dict[str, Any]] = []

    for url_key, outer_data in model.items():
        try:
            inner_data = outer_data.data

            rows.append(
                {
                    "url": inner_data.url,
                    "status": outer_data.status,
                    "message": outer_data.message,
                    "job_title": inner_data.job_title,
                    "company": inner_data.company,
                    "location": inner_data.location,
                    "salary_info": inner_data.salary_info,
                    "posted_date": inner_data.posted_date,
                    "content": (
                        json.dumps(inner_data.content, ensure_ascii=False)
                        if inner_data.content
                        else None
                    ),
                }
            )

        except ValidationError as e:
            logger.warning(f"❌ Validation error in job posting at {url_key}: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error at {url_key}: {e}", exc_info=True)

    logger.info(f"✅ Flattened {len(rows)} job postings into wide format.")

    return pd.DataFrame(rows)


def rehydrate_job_postings_from_table(df: pd.DataFrame) -> JobPostingsBatch:
    """
    Reconstructs a validated JobPostingsFile model from a wide-format DataFrame.

    Args:
        df (pd.DataFrame): Flattened job postings with one row per job.

    Returns:
        JobPostingsFile: A validated Pydantic model containing all job postings.
    """
    result: Dict[str, JobSiteResponse] = {}

    for idx, row in df.iterrows():
        try:
            url = row.url
            content = row.content
            parsed_content = json.loads(content) if isinstance(content, str) else {}

            if not all([row.status, row.job_title, row.company]):
                logger.warning(f"⚠️ Missing required fields at {url} — skipping")
                continue

            job_data = JobSiteData(
                url=url,
                job_title=row.job_title,
                company=row.company,
                location=row.location,
                salary_info=row.salary_info,
                posted_date=row.posted_date,
                content=parsed_content,
            )

            result[url] = JobSiteResponse(
                status=row.status,
                message=row.message,
                data=job_data,
            )

        except ValidationError as e:
            logger.warning(
                f"❌ Validation error rehydrating job posting at {row.url}: {e}"
            )
        except Exception as e:
            logger.error(f"❌ Unexpected error at {row.url}: {e}", exc_info=True)

    logger.info(f"✅ Rehydrated {len(result)} job postings from DataFrame.")
    return JobPostingsBatch(cast(dict[Union[HttpUrl, str], JobSiteResponse], result))


# * ☑️ Job Requirements
def flatten_requirements_to_table(model: Requirements) -> pd.DataFrame:
    """
    Flattens a validated Requirements model into a DataFrame for DuckDB insertion.

    Each row represents a job requirement with:
        - url: The job posting URL
        - requirement_key: e.g. '1.down_to_earth.0'
        - requirement: The full text string

    Args:
        model (Requirements): Validated Pydantic Requirements model.

    Returns:
        pd.DataFrame: Long-format table of flattened requirements.
    """
    try:
        rows = [
            {"url": model.url, "requirement_key": key, "requirement": value}
            for key, value in model.requirements.items()
        ]
        logger.info(f"✅ Flattened {len(rows)} requirements from: {model.url}")
        return pd.DataFrame(rows)

    except ValidationError as e:
        logger.error(f"❌ Validation error while flattening requirements: {e}")
        raise

    except Exception as e:
        logger.exception(f"❌ Unexpected error flattening requirements: {e}")
        raise


def rehydrate_requirements_from_table(df: pd.DataFrame) -> Requirements:
    """
    Reconstructs a validated Requirements model from a flattened requirements table.

    This is the inverse of `flatten_requirements_to_table`.

    Args:
        df (pd.DataFrame): DataFrame containing 'url', 'requirement_key', 'requirement'.

    Returns:
        Requirements: A validated Requirements model.
    """
    try:
        if df.empty:
            raise ValueError("Input DataFrame is empty — cannot rehydrate.")

        # Extract URL (assumes single-URL DataFrame, consistent with flattening)
        url = df["url"].iloc[0]

        # Create dictionary of key → value
        requirements = dict(zip(df["requirement_key"], df["requirement"]))

        model = Requirements(url=url, requirements=requirements)
        logger.info(
            f"✅ Rehydrated Requirements model with {len(requirements)} items from: {url}"
        )
        return model

    except ValidationError as e:
        logger.error(f"❌ Validation error while rehydrating requirements: {e}")
        raise

    except Exception as e:
        logger.exception(f"❌ Unexpected error rehydrating requirements: {e}")
        raise


def flatten_extracted_requirements_to_table(
    model: ExtractedRequirementsBatch,
) -> pd.DataFrame:
    """
    Flattens a validated ExtractedRequirementsFile model into a long-format DataFrame.

    Each requirement within a category becomes a row with associated metadata like
    index and status.

    Args:
        model (ExtractedRequirementsFile): The root model representing all extracted
        requirements.

    Returns:
        pd.DataFrame: Flattened format suitable for DuckDB ingestion.
    """
    rows = []

    for url_key, validated in model.items():
        try:
            reqs_dict = (
                validated.data.model_dump()
            )  # * have to use data_dump() b/c there are no fixed attributes in the model

            for cat_idx, (category, items) in enumerate(reqs_dict.items()):
                if not isinstance(items, list):
                    logger.warning(
                        f"Skipping non-list category '{category}' at {url_key}"
                    )
                    continue

                for item_idx, requirement in enumerate(items):
                    if not isinstance(requirement, str):
                        logger.warning(
                            f"Skipping non-string item in {category}[{item_idx}] at {url_key}"
                        )
                        continue

                    rows.append(
                        {
                            "url": url_key,
                            "status": validated.status,
                            "message": validated.message,
                            "requirement_category": category,
                            "requirement_category_idx": cat_idx,
                            "requirement": requirement,
                            "requirement_idx": item_idx,
                        }
                    )

        except ValidationError as e:
            logger.warning(f"Validation failed for {url_key}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error flattening {url_key}: {e}", exc_info=True)

    logger.info(f"✅ Flattened {len(rows)} extracted requirements.")
    return pd.DataFrame(rows)


def rehydrate_extracted_requirements_from_table(
    df: pd.DataFrame,
) -> ExtractedRequirementsBatch:
    """
    Reconstructs a validated ExtractedRequirementsFile model from a flattened
    requirements DataFrame.

    Args:
        df (pd.DataFrame): Flattened requirements with metadata columns.

    Returns:
        ExtractedRequirementsFile: A Pydantic model ready for saving or re-processing.
    """
    result = {}

    for url, group in df.groupby("url"):
        try:
            categories: Dict[str, List[str]] = {}

            for category, cat_group in group.groupby("requirement_category"):
                sorted_items = cat_group.sort_values("requirement_idx")[
                    "requirement"
                ].tolist()
                categories[str(category)] = sorted_items

            nested = NestedRequirements(**categories)

            first = group.iloc[0]
            response = RequirementsResponse(
                status=first["status"],
                message=first["message"],
                data=nested,
            )

            result[url] = response

        except ValidationError as e:
            logger.warning(f"Validation failed during rehydration for {url}: {e}")
        except Exception as e:
            logger.error(
                f"Error rehydrating extracted requirements for {url}: {e}",
                exc_info=True,
            )

    logger.info(f"✅ Rehydrated {len(result)} extracted requirements.")

    return ExtractedRequirementsBatch(
        cast(Dict[Union[str, HttpUrl], RequirementsResponse], result)
    )


# * ✅  Responsibilities (Resume)
def flatten_responsibilities_to_table(model: Responsibilities) -> pd.DataFrame:
    """
    Converts a validated Responsibilities model into a long-format DataFrame.

    Each responsibility becomes a row with its unique key and value.

    Args:
        model (Responsibilities): Validated responsibilities input model.

    Returns:
        pd.DataFrame: Table with columns 'url', 'responsibility_key', and 'responsibility'.
    """
    try:
        rows = [
            {
                "url": str(model.url) if model.url else "Not Available",
                "responsibility_key": key,
                "responsibility": value,
            }
            for key, value in model.responsibilities.items()
        ]
        logger.info(f"✅ Flattened {len(rows)} responsibilities from: {model.url}")
        return pd.DataFrame(rows)

    except ValidationError as e:
        logger.error(f"❌ Validation error while flattening responsibilities: {e}")
        raise
    except Exception as e:
        logger.exception(f"❌ Unexpected error flattening responsibilities: {e}")
        raise


def rehydrate_responsibilities_from_table(df: pd.DataFrame) -> Responsibilities:
    """
    Reconstructs a validated Responsibilities model from a flattened table.

    Assumes a single URL per table (or group).

    Args:
        df (pd.DataFrame): Table with columns 'url', 'responsibility_key', 'responsibility'.

    Returns:
        Responsibilities: A validated Responsibilities model instance.
    """
    try:
        if df.empty:
            raise ValueError("❌ DataFrame is empty — cannot rehydrate.")

        url = df["url"].iloc[0]
        responsibilities = dict(zip(df["responsibility_key"], df["responsibility"]))

        model = Responsibilities(url=url, responsibilities=responsibilities)
        logger.info(
            f"✅ Rehydrated responsibilities with {len(responsibilities)} items from: {url}"
        )
        return model

    except ValidationError as e:
        logger.error(f"❌ Validation error rehydrating responsibilities: {e}")
        raise
    except Exception as e:
        logger.exception(f"❌ Unexpected error rehydrating responsibilities: {e}")
        raise


def flatten_pruned_responsibilities_to_table(
    model: Responsibilities, pruned_by: str
) -> pd.DataFrame:
    """
    Flattens a validated Responsibilities model into a pruned responsibilities table.

    Adds an extra 'pruned_by' column to identify the pruning method used.

    Args:
        model (Responsibilities): Validated input of pruned responsibilities.
        pruned_by (str): Method of pruning (e.g., 'manual', 'llm', 'rule_based').

    Returns:
        pd.DataFrame: Flattened table with 'url', 'responsibility_key', 'responsibility',
        'pruned_by'.
    """
    df = flatten_responsibilities_to_table(model)
    df["pruned_by"] = pruned_by
    return df


def rehydrate_pruned_responsibilities_from_table(df: pd.DataFrame) -> Responsibilities:
    """
    Reconstructs a Responsibilities model from a pruned responsibilities table.

    Drops 'pruned_by' before reconstruction since it's not needed in the nested model.

    Args:
        df (pd.DataFrame): Flattened pruned responsibilities with a 'pruned_by' column.

    Returns:
        Responsibilities: Reconstructed Responsibilities model.
    """
    # Drop metadata columns if they exist
    if "pruned_by" in df.columns:
        df = df.drop(columns=["pruned_by"])

    return rehydrate_responsibilities_from_table(df)


def flatten_nested_responsibilities_to_table(
    model: NestedResponsibilities,
) -> pd.DataFrame:
    """
    Flattens a NestedResponsibilities model into a DataFrame suitable
    for database storage.

    Each row includes:
        - url: The URL of the job posting
        - responsibility_key: Unique key identifying the original responsibility
        - requirement_key: ID of the requirement it's aligned with
        - responsibility: The aligned (edited) responsibility text

    Args:
        model (NestedResponsibilities): A validated NestedResponsibilities model.

    Returns:
        pd.DataFrame: Long-format table of responsibility–requirement alignments.

    Note:
        The model attribute 'optimized_text' is stored as 'responsibility' in the flattened output.
    """

    try:
        rows = []
        for responsibility_key, match_obj in model.responsibilities.items():
            for (
                requirement_key,
                opt_text_obj,
            ) in match_obj.optimized_by_requirements.items():
                rows.append(
                    {
                        "url": str(model.url),
                        "responsibility_key": responsibility_key,
                        "requirement_key": requirement_key,
                        # "optimized_text": opt_text_obj.optimized_text,
                        "responsibility": opt_text_obj.optimized_text,  # ☑️ use responsibility as standard col name
                    }
                )

        logger.info(
            f"✅ Flattened {len(rows)} optimized responsibilities from: {model.url}"
        )
        return pd.DataFrame(rows)

    except ValidationError as e:
        logger.error(
            f"❌ Validation error while flattening NestedResponsibilities: {e}"
        )
        raise
    except Exception as e:
        logger.exception(f"❌ Unexpected error flattening NestedResponsibilities: {e}")
        raise


def rehydrate_nested_responsibilities_from_table(
    df: pd.DataFrame,
) -> NestedResponsibilities:
    """
    Reconstructs a NestedResponsibilities model from a flattened table.

    Args:
        df (pd.DataFrame):
            A DataFrame containing flattened responsibility alignment data,
            with columns: 'url', 'responsibility_key', 'requirement_key', and 'responsibility'.

    Returns:
        NestedResponsibilities: A fully validated NestedResponsibilities model.

    * Note:
        The DataFrame uses the column name 'responsibility', which is renamed to
        'optimized_text' to match the model schema.
    """
    try:
        if df.empty:
            raise ValueError("Input DataFrame is empty — cannot rehydrate.")

        # ☑️ Rename responsibility -> optimized_text
        if "responsibility" in df.columns:
            df = df.rename(columns={"responsibility": "optimized_text"})

        url = df["url"].iloc[0]
        grouped = df.groupby("responsibility_key")

        responsibilities = {}

        for key, group in grouped:
            opt_dict = {
                row["requirement_key"]: {"optimized_text": row["optimized_text"]}
                for _, row in group.iterrows()
            }
            responsibilities[key] = {"optimized_by_requirements": opt_dict}

        model = NestedResponsibilities(url=url, responsibilities=responsibilities)
        logger.info(
            f"✅ Rehydrated NestedResponsibilities from {len(responsibilities)} keys for: {url}"
        )
        return model

    except ValidationError as e:
        logger.error(
            f"❌ Validation error while rehydrating NestedResponsibilities: {e}"
        )
        raise
    except Exception as e:
        logger.exception(f"❌ Unexpected error rehydrating NestedResponsibilities: {e}")
        raise
