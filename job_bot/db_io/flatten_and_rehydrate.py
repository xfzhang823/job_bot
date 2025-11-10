"""
flatten_and_rehydrate.py

Utilities to convert between nested, validated models (Pydantic) and
relational tables (pandas → DuckDB) used in a resume–job alignment pipeline.

Supported data:
- Job posting URLs and postings
- Extracted & flattened requirements
- Flattened, pruned, and edited responsibilities (incl. nested alignments)

Highlights:
- Model → DataFrame “flatteners” for DuckDB inserts
- DataFrame → Model “rehydrators” for round-trips and pipeline reuse
- Consistent keys/columns for stable joins (url, iteration, *_key)
- Logging and light validation at IO boundaries

Notes
-----
Schema enforcement and metadata stamping are handled by the ingestion layer
(e.g., `align_df_with_schema()` and insert config). Rehydrators assume tables
conform to the registry schema and are primarily responsible for structure,
not per-column validation.
"""

from __future__ import annotations

# Imports
from pathlib import Path
import json
import logging
from datetime import datetime, date
from typing import Any, cast, Dict, Callable, Mapping, Type, TypeVar, Tuple, List, Union
import pandas as pd
from pydantic import BaseModel, HttpUrl, ValidationError

# User defined
from job_bot.db_io.pipeline_enums import (
    TableName,
    LLMProvider,
    Version,
)
from job_bot.db_io.db_transform import add_metadata
from job_bot.models.resume_job_description_io_models import (
    NestedResponsibilities,
    Requirements,
    Responsibilities,
    JobPostingUrlMetadata,
    JobPostingUrlsBatch,
    JobPostingsBatch,
    ExtractedRequirementsBatch,
)
from job_bot.models.llm_response_models import (
    JobSiteResponse,
    JobSiteData,
    RequirementsResponse,
    NestedRequirements,
)
from job_bot.models.model_type import ModelType

# from job_bot.models.resume_job_description_io_models import ExtractedRequirementsBatch
# from job_bot.models.llm_response_models import
# Set up logger for this module
logger = logging.getLogger(__name__)


# utils: shared normalizer
T = TypeVar("T", bound=BaseModel)
FlattenFuncTyped = Callable[[Any], pd.DataFrame]  # accepts batch or single
# Accept a single type OR a tuple of types
ExpectedTypes = Union[Type[T], tuple[Type[Any], ...]]
FlattenDispatch = Dict[TableName, Tuple[ExpectedTypes, FlattenFuncTyped]]


def get_flatten_dispatch() -> FlattenDispatch:
    """
    Map DuckDB tables to their expected input model types and flattener functions.

    Returns
    -------
    dict[TableName, tuple[ExpectedTypes, FlattenFuncTyped]]
        A dispatch map used to route a validated model (or batch RootModel) to
        the correct flattening function.
    """
    return {
        TableName.JOB_URLS: (JobPostingUrlsBatch, flatten_job_urls_to_table),
        # 👇 accept BOTH batch and single for these two:
        TableName.JOB_POSTINGS: (
            (JobPostingsBatch, JobSiteResponse),
            flatten_job_postings_to_table,
        ),
        TableName.EXTRACTED_REQUIREMENTS: (
            (ExtractedRequirementsBatch, RequirementsResponse),
            flatten_extracted_requirements_to_table,
        ),
        TableName.FLATTENED_REQUIREMENTS: (
            Requirements,
            flatten_requirements_to_table,
        ),
        TableName.FLATTENED_RESPONSIBILITIES: (
            Responsibilities,
            flatten_responsibilities_to_table,
        ),
        TableName.PRUNED_RESPONSIBILITIES: (
            NestedResponsibilities,
            flatten_nested_responsibilities_to_table,
        ),
        TableName.EDITED_RESPONSIBILITIES: (
            NestedResponsibilities,
            flatten_nested_responsibilities_to_table,
        ),
    }


def _as_url_mapping(model: Any, *, url_of: Callable[[Any], str]) -> Dict[str, Any]:
    """
    Normalize heterogeneous inputs to {url: item}.

    Accepts:
      • RootModel[dict[url, item]] → returns the dict
      • dict[url, item]            → returns as-is
      • single item                → { url_of(item): item }

    Raises
    ------
    ValueError
        If a single item does not yield a URL via `url_of`.
    """
    raw = getattr(model, "root", model)  # handle RootModel
    if isinstance(raw, dict):
        return raw

    url = url_of(model)  # single item
    if not url:
        raise ValueError(
            f"Cannot derive URL from single item of type {type(model).__name__}"
        )
    return {url: model}


def flatten_keyed_dict(
    d: Dict[str, Dict[str, Any]], key_col: str = "url"
) -> pd.DataFrame:
    """
    Convert a dict-of-dicts into a DataFrame, lifting the outer key into `key_col`.

    Parameters
    ----------
    d : dict[str, dict[str, Any]]
        Mapping of identifiers (e.g., URLs) to row dicts.
    key_col : str, default "url"
        Name of the column that stores the outer key.

    Returns
    -------
    pd.DataFrame
    """
    return pd.DataFrame([{key_col: k, **v} for k, v in d.items()])


def unflatten_to_keyed_dict(
    df: pd.DataFrame, key_col: str = "url"
) -> Dict[str, Dict[str, Any]]:
    """
    Convert a DataFrame back into {key: row_dict}, using `key_col` as the key.

    Parameters
    ----------
    df : pd.DataFrame
    key_col : str, default "url"

    Returns
    -------
    dict[str, dict[str, Any]]
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


# * ✔️ Jobpostings
def flatten_job_postings_to_table(
    model: Union["JobPostingsBatch", "JobSiteResponse", Dict[str, "JobSiteResponse"]],
) -> pd.DataFrame:
    """
    Flatten job postings (batch or single) to a wide table.

    Input
    -----
    - RootModel[dict[url, JobSiteResponse]]  OR
    - dict[url, JobSiteResponse]             OR
    - JobSiteResponse

    Output columns
    --------------
    url, job_title, company, location, salary_info, posted_date, content (JSON str)

    Returns
    -------
    pd.DataFrame
    """
    rows: List[Dict[str, Any]] = []

    # --- Normalize to Mapping[str, JobSiteResponse] ---
    # unwrap RootModel if present (a neat Pydantic trick - no need for if/elfi)

    raw = getattr(model, "root", model)  # unwrap RootModel if present

    if isinstance(raw, Mapping):
        # Tell the type checker that values are JobSiteResponse
        mapping: Mapping[str, JobSiteResponse] = cast(
            Mapping[str, JobSiteResponse], raw
        )
    else:
        # single item path
        single = cast(JobSiteResponse, raw)
        url = getattr(single, "url", None) or getattr(
            getattr(single, "data", None), "url", None
        )
        if not url:
            raise ValueError(
                f"Cannot derive URL from single item of type {type(single).__name__}"
            )
        mapping = {url: single}

    for url_key, outer in mapping.items():
        try:
            data = outer.data  # JobSiteResponse.data
            rows.append(
                {
                    "url": getattr(data, "url", url_key),
                    "job_title": getattr(data, "job_title", None),
                    "company": getattr(data, "company", None),
                    "location": getattr(data, "location", None),
                    "salary_info": getattr(data, "salary_info", None),
                    "posted_date": getattr(data, "posted_date", None),
                    "content": (
                        json.dumps(getattr(data, "content", None), ensure_ascii=False)
                        if getattr(data, "content", None) is not None
                        else None
                    ),
                }
            )
        except ValidationError as e:
            logger.warning(f"❌ Validation error in job posting at {url_key}: {e}")
        except Exception as e:
            logger.warning(f"❌ Skipping malformed posting at {url_key}: {e}")

    return pd.DataFrame(rows)


def rehydrate_job_urls_from_table(df: pd.DataFrame) -> JobPostingUrlsBatch:
    """
    Rehydrates a DataFrame from the `job_urls` table into
    a JobPostingUrlsFile model.

    Args:
        df (pd.DataFrame): Flattened table containing 'url', 'company',
        and 'job_title' columns.

    Returns:
        JobPostingUrlsBatch: Root model with validated mapping of URL -> metadata.
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


# def rehydrate_job_postings_from_table(df: pd.DataFrame) -> JobPostingsBatch:
#     """
#     Reconstructs a validated JobPostingsFile model from a wide-format DataFrame.

#     Args:
#         df (pd.DataFrame): Flattened job postings with one row per job.

#     Returns:
#         JobPostingsFile: A validated Pydantic model containing all job postings.
#     """
#     result: Dict[str, JobSiteResponse] = {}

#     for idx, row in df.iterrows():
#         try:
#             url = row.url
#             content = row.content
#             parsed_content = json.loads(content) if isinstance(content, str) else {}

#             if not all([row.status, row.job_title, row.company]):
#                 logger.warning(f"⚠️ Missing required fields at {url} — skipping")
#                 continue

#             job_data = JobSiteData(
#                 url=url,
#                 job_title=row.job_title,
#                 company=row.company,
#                 location=row.location,
#                 salary_info=row.salary_info,
#                 posted_date=row.posted_date,
#                 content=parsed_content,
#             )

#             result[url] = JobSiteResponse(
#                 status=row.status,
#                 message=row.message,
#                 data=job_data,
#             )

#         except ValidationError as e:
#             logger.warning(
#                 f"❌ Validation error rehydrating job posting at {row.url}: {e}"
#             )
#         except Exception as e:
#             logger.error(f"❌ Unexpected error at {row.url}: {e}", exc_info=True)

#     logger.info(f"✅ Rehydrated {len(result)} job postings from DataFrame.")
#     return JobPostingsBatch(cast(dict[Union[HttpUrl, str], JobSiteResponse], result))


def rehydrate_job_postings_from_table(df: pd.DataFrame) -> JobPostingsBatch:
    """
    Reconstruct JobPostingsBatch from a job_postings DataFrame.

    Args:
        df (pd.DataFrame): Flattened job postings with one row per job.

    Returns:
        JobPostingsFile: A validated Pydantic model containing all job postings.

    Since DB does not store `status` or `message`, we fill in dummy defaults:
      - status="success"
      - message="Rehydrated from DB"
    """
    result: Dict[Union[HttpUrl, str], JobSiteResponse] = {}
    # result: dict[str, JobSiteResponse] = {}

    for rec in df.to_dict(orient="records"):
        try:
            url = rec.get("url")
            if not url:
                continue

            job_title = rec.get("job_title")
            company = rec.get("company")

            # 🚦 Guard: skip rows missing required fields
            if not url or not job_title or not company:
                logger.warning(f"⚠️ Missing required fields at {url} — skipping")
                continue

            # Required minimum
            if not all([job_title, company]):
                continue

            content = rec.get("content")
            parsed_content = (
                json.loads(content) if isinstance(content, str) else content or {}
            )

            job_data = JobSiteData(
                url=url,
                job_title=job_title,
                company=company,
                location=rec.get("location"),
                salary_info=rec.get("salary_info"),
                posted_date=_coerce_posted_date(rec.get("posted_date")),
                content=parsed_content,
            )

            # 👇 Dummy values here
            result[url] = JobSiteResponse(
                status="success",
                message="Rehydrated from DB",
                data=job_data,
            )

        except ValidationError as e:
            print(
                f"❌ Validation error rehydrating job posting at {rec.get('url')}: {e}"
            )
        except Exception as e:
            print(f"❌ Unexpected error at {rec.get('url')}: {e}")

    return JobPostingsBatch(result)


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
    model: Union[
        "ExtractedRequirementsBatch",
        "RequirementsResponse",
        Dict[str, "RequirementsResponse"],
    ],
) -> pd.DataFrame:
    """
    Flatten extracted (tiered) requirements into a long table.

    Input
    -----
    - RootModel[dict[url, RequirementsResponse]] OR
    - dict[url, RequirementsResponse]           OR
    - RequirementsResponse

    Output columns
    --------------
    url, status, message, requirement_category, requirement_category_key,
    requirement, requirement_key

    Returns
    -------
    pd.DataFrame
    """
    rows: List[Dict[str, Any]] = []

    # Normalize to dict[url -> item], inline (no helper)
    mapping = getattr(model, "root", model)  # handle RootModel
    if not isinstance(mapping, dict):
        single = mapping
        url = getattr(single, "url", None)
        if url is None:
            data = getattr(single, "data", None)
            url = getattr(data, "url", None)
        if not url:
            raise ValueError(
                f"Cannot derive URL from single item of type {type(single).__name__}"
            )
        mapping = {url: single}

    for url_key, validated in mapping.items():
        try:
            data_obj = getattr(validated, "data", None)
            if not data_obj:
                logger.warning(f"No 'data' found for {url_key}; skipping.")
                continue

            # categories -> list[str]; allow dict or Pydantic model
            reqs_dict = (
                data_obj
                if isinstance(data_obj, dict)
                else data_obj.model_dump(exclude_none=True)
            )
            if not isinstance(reqs_dict, dict):
                logger.warning(f"'data' not dict-like for {url_key}; skipping.")
                continue

            for cat_idx, (category, items) in enumerate(reqs_dict.items()):
                if not isinstance(items, list):
                    logger.warning(
                        f"Skipping non-list category '{category}' at {url_key}"
                    )
                    continue

                for item_idx, requirement in enumerate(items):
                    if not isinstance(requirement, str):
                        logger.warning(
                            f"Skipping non-string {category}[{item_idx}] at {url_key}"
                        )
                        continue

                    # Need to generate unique keys for db tbl (PK rule)
                    category_ = str(category).strip().lower().replace(" ", "_")
                    composite_key = f"{cat_idx}.{category_}.{item_idx}"  # unique key
                    rows.append(
                        {
                            "url": url_key,
                            "requirement_category": category,
                            "requirement_category_key": cat_idx,
                            "requirement": requirement,
                            "requirement_key": composite_key,
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
    Rehydrate a long extracted-requirements table into `ExtractedRequirementsBatch`.

    Grouping & Ordering
    -------------------
    • Rows are grouped by (url, requirement_category).
    • Within each category, rows are sorted numerically by the trailing index
      of the composite `requirement_key` (e.g., "0.down_to_earth.3" → 3).
    • Composite keys are used only for stable ordering; they are not exposed
      in the rehydrated NestedRequirements, which remain {category: [list of strings]}.

    Stripping
    ---------
    The trailing index is extracted after applying `str(x).strip()` to guard
    against whitespace around the composite key.

    Returns
    -------
    ExtractedRequirementsBatch
        A validated batch model containing URL → RequirementsResponse.
    """
    result = {}

    for url, group in df.groupby("url"):
        try:
            categories: Dict[str, List[str]] = {}

            for category, cat_group in group.groupby("requirement_category"):
                cg = cat_group.copy()
                # Sort by numeric tail of composite key, e.g. "0.cat.3" → 3
                cg["_ord"] = cg["requirement_key"].map(
                    lambda x: int(str(x).strip().rsplit(".", 1)[-1])
                )
                sorted_items = cg.sort_values("_ord")["requirement"].tolist()
                categories[str(category)] = sorted_items

            nested = NestedRequirements(**categories)

            first = group.iloc[0]

            # status: always a str; default to "success" if missing/NaN
            raw_status = first["status"] if "status" in first.index else None
            status_str = (
                "success"
                if (raw_status is None or pd.isna(raw_status))
                else str(raw_status)
            )

            # message: Optional[str]; coerce NaN -> None
            raw_msg = first["message"] if "message" in first.index else None
            message_str = (
                None if (raw_msg is None or pd.isna(raw_msg)) else str(raw_msg)
            )

            response = RequirementsResponse(
                status=status_str,
                message=message_str,
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
    Flatten responsibilities into a pruned table with provenance.

    Output columns
    --------------
    url, responsibility_key, responsibility, pruned_by

    Returns
    -------
    pd.DataFrame
    """
    df = flatten_responsibilities_to_table(model)
    df["pruned_by"] = pruned_by
    return df


def rehydrate_pruned_responsibilities_from_table(df: pd.DataFrame) -> Responsibilities:
    """
    Rehydrate pruned responsibilities into `Responsibilities`.

    Notes
    -----
    Drops `pruned_by` before rehydration since it is metadata, not part of the
    nested responsibilities model.

    Returns
    -------
    Responsibilities
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
        The model attribute 'optimized_text' is stored as 'responsibility' in
            the flattened output.
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


# Single choke function to call flatten functions
def flatten_model_to_df(
    model: ModelType,
    table_name: TableName,
    *,
    source_file: Path | str | None = None,
    iteration: int | None = None,
    version: Version | str | None = None,
    llm_provider: LLMProvider | str | None = None,
    model_id: str | None = None,
) -> pd.DataFrame:
    """
    Flatten a validated Pydantic model into a DataFrame and enrich it with
    table-appropriate metadata.

    This function routes the model through the correct flattening function
    (per `FLATTEN_DISPATCH`), then applies metadata stamping according to
    the target table’s schema in the registry.

    Notes:
        • No `stage` stamping here — that belongs only in `pipeline_control`.
        • Timestamps come from DDL/mixins; they are not set explicitly here.
        • LLM artifact tables (e.g., EDITED_RESPONSIBILITIES) require both
          `llm_provider` and `model_id`.

    Args:
        model (ModelType):
            A validated Pydantic model instance (e.g., JobPostingsBatch,
            RequirementsResponse, Responsibilities, etc.).
        table_name (TableName):
            The DuckDB table enum identifying the destination.
        source_file (Path | str | None, optional):
            Originating file path, stamped into metadata if provided.
        iteration (int | None):
            Pipeline iteration index. Required for most non-seed tables.
        version (Version | str | None, optional):
            Editorial version (e.g., ORIGINAL, EDITED). Only stamped if the
            target table has a `version` column.
        llm_provider (LLMProvider | str | None, optional):
            Name of the LLM provider (e.g., "openai", "anthropic").
            Required only for LLM artifact tables.
        model_id (str | None, optional):
            Specific model identifier (e.g., "gpt-4o-mini", "claude-haiku").
            Required only for LLM artifact tables.

    Returns:
        pd.DataFrame:
            A DataFrame containing the flattened table rows plus metadata,
            aligned with the schema defined in the registry.
    """
    logger.info(
        "🪄 Flattening model '%s' → table '%s'", type(model).__name__, table_name
    )

    FLATTEN_DISPATCH = get_flatten_dispatch()

    if table_name not in FLATTEN_DISPATCH:
        raise ValueError(f"Unsupported table: {table_name}")

    expected_types, flatten_func = FLATTEN_DISPATCH[table_name]

    ok = False
    # 1) direct type match(s)
    if isinstance(
        model, expected_types if isinstance(expected_types, tuple) else expected_types
    ):
        ok = True
    # 2) mapping (plain dict) accepted by flexible flatteners
    elif isinstance(model, Mapping):
        ok = True
    # 3) RootModel with a mapping `.root`
    elif hasattr(model, "root") and isinstance(getattr(model, "root"), Mapping):
        ok = True

    if not ok:
        # pretty error message
        if isinstance(expected_types, tuple):
            exp_names = ", ".join(t.__name__ for t in expected_types)
            exp_str = f"({exp_names})"
        else:
            exp_str = expected_types.__name__
        raise TypeError(
            f"Expected {exp_str} (or Mapping / RootModel with mapping .root) for table '{table_name}', "
            f"got '{type(model).__name__}'"
        )

    df = flatten_func(model)

    # Enforce LLM metadata where required
    if table_name in {TableName.EDITED_RESPONSIBILITIES}:  # add others if applicable
        if llm_provider is None or model_id is None:
            raise ValueError(
                f"{table_name} requires llm_provider and model_id (LLM artifact table)."
            )

    # Normalize source_file
    path = (
        Path(source_file)
        if isinstance(source_file, str) and source_file
        else source_file
    )

    # Table-aware stamping only (no global stage/timestamp)
    df = add_metadata(
        df=df,
        table=table_name,
        source_file=path,
        iteration=iteration,
        version=version,  # will be ignored if table doesn't have 'version'
        llm_provider=llm_provider,  # ignored if table doesn't have it
        model_id=model_id,  # ignored if table doesn't have it
    )
    return df


def _coerce_posted_date(val):
    # Treat NaT/NaN/None as None
    if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
        return None
    # pandas Timestamp / python datetime / date → ISO string "YYYY-MM-DD"
    if isinstance(val, (pd.Timestamp, datetime, date)):
        return val.date().isoformat() if hasattr(val, "date") else val.isoformat()
    # Already a string? strip and normalize empty→None
    if isinstance(val, str):
        s = val.strip()
        return s or None
    # Fallback: stringify
    return str(val)
