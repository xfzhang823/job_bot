"""
Module: resps_reqs_crosstab_file_creation_pipeline.py

This module processes similarity metrics CSV files and generates cross-tab
Excel reports matching job responsibilities to job requirements.

Pipeline Steps:
1. Load similarity metrics file paths from a mapping JSON file.
2. Extract company names and job titles using a job postings JSON file.
3. Merge job postings and mapping file on URLs.
4. Generate a cross-tab report for each similarity metrics file.
5. Save results as Excel files in the designated output directory.
6. Uses asyncio for parallel processing to improve performance.

Key Features:
- **Asynchronous execution** for faster processing.
- **Automatic filename generation** based on company and job title.
- **Graceful handling of missing data** (e.g., missing job URLs).
- **Ensures output directories exist** before saving files.

Dependencies:
- `pandas`
- `asyncio`
- `pathlib`
- `logging`
- `json`

Author: [Your Name]
Last Updated: [Date]
"""

import logging
import asyncio
import json
import pandas as pd
from pathlib import Path
from typing import Optional

# User-defined imports
from utils.generic_utils import read_from_json_file
from utils.pydantic_model_loaders_from_files import (
    load_job_file_mappings_model,
)
from evaluation_optimization.evaluation_optimization_utils import create_file_name
from human_review_and_editing.create_resp_req_crosstab import (
    create_resp_req_crosstab,
    save_crosstab,
)
from project_config import (
    RESPS_REQS_MATCHINGS_DIR,
    JOB_POSTING_URLS_FILE,
)

# Set up logger
logger = logging.getLogger(__name__)


### ✅ JSON -> Pandas DataFrame Loading Functions ###
def load_json_to_df(json_file: Path) -> pd.DataFrame:
    """
    Load a JSON file into a Pandas DataFrame.

    Args:
        json_file (Path): Path to the JSON file.

    Returns:
        pd.DataFrame: DataFrame containing the JSON data.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("❌ Expected JSON as a dictionary with URLs as keys.")

    df = pd.DataFrame.from_dict(data, orient="index").reset_index()
    df.rename(columns={"index": "url"}, inplace=True)  # Ensure 'url' is the key

    logger.info(f"✅ Loaded {len(df)} records from {json_file}")
    return df


def load_job_postings_to_df(job_urls_file: Path) -> pd.DataFrame:
    """
    Load job postings from a JSON file and remove duplicate 'url' column.

    Args:
        job_urls_file (Path): Path to the JSON file.

    Returns:
        pd.DataFrame: DataFrame containing ['url', 'company', 'job_title'].
    """
    df = load_json_to_df(job_urls_file)

    # ✅ Remove extra 'url' column if it exists
    if df.columns.duplicated().any():
        logger.warning("⚠️ Duplicate 'url' column found. Removing extra one.")
        df = df.loc[:, ~df.columns.duplicated()]

    # ✅ Keep only required columns
    df = df[["url", "company", "job_title"]].drop_duplicates()

    logger.info(f"✅ Job postings DataFrame created:\n{df.head()}")
    return df


def load_mapping_file_to_df(mapping_file: Path) -> pd.DataFrame:
    """
    Load mapping JSON file and extract ['url', 'sim_metrics'].

    Args:
        mapping_file (Path): Path to the mapping JSON file.

    Returns:
        pd.DataFrame: DataFrame containing ['url', 'sim_metrics'].
    """
    df = load_json_to_df(mapping_file)

    # ✅ Remove duplicate columns if they exist
    if df.columns.duplicated().any():
        logger.warning("⚠️ Duplicate 'url' column found. Removing extra one.")
        df = df.loc[:, ~df.columns.duplicated()]

    # ✅ Keep only required columns
    df = df[["url", "sim_metrics"]].drop_duplicates()

    logger.info(f"✅ Mapping file DataFrame created:\n{df.head()}")
    return df


# ✅ Data Merging Function ###
def merge_mapping_file_wt_job_postings_urls(
    df_mapping: pd.DataFrame, df_job_postings: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge job postings and mapping DataFrames on 'url'.

    Args:
        - df_mapping (pd.DataFrame): Mapping file DataFrame.
        - df_job_postings (pd.DataFrame): Job postings DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame with matched company, job title, and similarity
        metrics file paths.
    """

    # Log before merging
    logger.info(
        f"Before merging: df_mapping={df_mapping.shape}, df_job_postings={df_job_postings.shape}"
    )

    # Perform inner merge to keep only matching URLs
    merged_df = df_mapping.merge(df_job_postings, on="url", how="inner")

    # ✅ Fill missing values for company and job title
    merged_df["company"].fillna("Unknown_Company", inplace=True)
    merged_df["job_title"].fillna("Unknown_Title", inplace=True)

    # ✅ Ensure no NaN values exist in sim_metrics
    assert (
        merged_df["sim_metrics"].isna().sum() == 0
    ), "NaN values still exist in sim_metrics!"

    # Log after merging
    logger.info(f"✅ Merged DataFrame ({merged_df.shape[0]} rows):\n{merged_df.head()}")

    return merged_df


def load_original_responsibilities(
    url: str, mapping_file: Path | str
) -> dict[str, str]:
    """
    Load the original responsibilities mapping for a given URL from the mapping file.
    The mapping file is loaded as a Pydantic model, and for the given URL the corresponding
    'resps' file path is retrieved. That file is then read as JSON to extract
    the responsibilities, which are returned as a dictionary mapping responsibility keys
    to their original text.

    Args:
        url (str): The job posting URL.
        mapping_file (Path): Path to the JSON mapping file.

    Returns:
        dict[str, str]: A dictionary containing the original responsibilities, or an empty
        dictionary if loading fails.
    """
    logger.info(f"Loading original responsibilities for {url}...")

    # Ensure file path is Path if str
    mapping_file = Path(mapping_file) if isinstance(mapping_file, str) else mapping_file
    if not mapping_file.exists():
        logger.error(f"❌ Mapping file does not exist: {mapping_file}")
        return {}

    # Read mapping file
    original_mapping_model = load_job_file_mappings_model(mapping_file)
    if original_mapping_model:
        mapping = (
            original_mapping_model.root
            if hasattr(original_mapping_model, "root")
            else original_mapping_model.model_dump()
        )
        # Ensure the keys are strings
        mapping = {str(k): v for k, v in mapping.items()}
    else:
        mapping = {}

    # Retrieve the responsibilities file path for the provided URL.
    original_entry = mapping.get(url)
    original_resps_file = original_entry.resps if original_entry is not None else None

    # ✅ Verify the file exists before reading
    if not original_resps_file:
        logger.error(f"No responsibilities file found for URL {url} in mapping.")
        return {}

    # ✅ Read the responsibilities JSON file
    try:
        original_resps_dict = read_from_json_file(
            json_file=Path(original_resps_file), key="responsibilities"
        )
        logger.debug(f"Responsibilities file for {url} at {original_resps_file}")
        logger.debug(original_resps_dict)
    except ValueError as e:
        logger.error(
            f"Failed to load responsibilities from {original_resps_file} for {url}: {e}"
        )
        original_resps_dict = {}

    return original_resps_dict


### ✅ Async Processing Function ###
async def process_file_async(
    input_file: Path,
    output_file: Path,
    score_threshold: float = 0,
    original_resps_dict: dict[str, str] | None = None,
) -> None:
    """
    Asynchronously process a similarity metrics CSV file and generate a cross-tab Excel report.

    Args:
        input_file (Path): Path to the similarity metrics CSV file.
        output_file (Path): Path where the generated cross-tab Excel file should be saved.
        score_threshold (float, optional): Threshold for filtering responsibility text.

    Returns:
        None
    """
    try:
        logger.info(f"🔄 Processing: {input_file.name}")

        # ✅ Generate cross-tabulation
        result_df = await asyncio.to_thread(
            create_resp_req_crosstab,
            file_path=input_file,
            score_threshold=score_threshold,
            original_resps_dict=original_resps_dict,
        )

        # 🔍 Debugging: Log the lengths of each DataFrame column
        for col in result_df.columns:
            col_length = len(result_df[col])
            logger.debug(f"Column '{col}' length: {col_length}")

        # ✅ Save to Excel
        await asyncio.to_thread(
            save_crosstab, result_df, output_file, file_format="excel"
        )
        logger.info(f"✅ Crosstab saved: {output_file}")

    except Exception as e:
        logger.error(f"❌ Error processing {input_file.name}: {str(e)}")


### ✅ Main Async Pipeline ###
async def run_resps_reqs_crosstab_pipeline_async(
    mapping_file: Path,
    cross_tab_output_dir: Path = RESPS_REQS_MATCHINGS_DIR,
    score_threshold: float = 0,
    original_mapping_file: Optional[Path] = None,
) -> None:
    """
    Asynchronously process all similarity metrics files and generate cross-tab Excel reports.

    Args:
        mapping_file (Path): Path to the JSON mapping file.
        cross_tab_output_dir (Path): Directory where cross-tab Excel files will be saved.
        score_threshold (float, optional): Threshold for filtering responsibility text.
        original_mapping_file (Optional[Path]): Path to the original responsibilities mapping file.

    Returns:
        None
    """
    logger.info(f"📂 Loading mapping file: {mapping_file}")

    # Ensure output directory exists
    cross_tab_output_dir.mkdir(parents=True, exist_ok=True)

    # Load job postings and mapping file DataFrames
    df_job_postings = load_job_postings_to_df(JOB_POSTING_URLS_FILE)
    df_mapping = load_mapping_file_to_df(mapping_file)

    # Merge DataFrames on 'url'
    merged_df = merge_mapping_file_wt_job_postings_urls(df_mapping, df_job_postings)

    logger.info(f"Merged Dataframe: {len(merged_df)}")
    logger.info(merged_df["sim_metrics"].tolist())  # debug

    # Identify similarity metrics CSV files and schedule processing tasks
    tasks = []
    for _, row in merged_df.iterrows():
        sim_metrics_file = Path(row.at["sim_metrics"])
        job_url = row.at["url"]

        # If an original mapping file is provided, load the original responsibilities for this URL.
        if original_mapping_file is not None:
            original_resps_dict = load_original_responsibilities(
                url=job_url, mapping_file=original_mapping_file
            )
        else:
            original_resps_dict = {}

        if not sim_metrics_file.exists():
            logger.warning(f"⚠️ Skipping: {sim_metrics_file} (File not found)")
            continue

        company, job_title = row.at["company"], row.at["job_title"]

        # Generate output file name
        output_filename = create_file_name(
            company, job_title, suffix="_crosstab", ext=".xlsx"
        )
        if not output_filename:
            logger.warning(
                f"⚠️ Filename could not be generated for {company}, {job_title}. Using default."
            )
            output_filename = "default_filename.xlsx"

        output_file = cross_tab_output_dir / output_filename

        # Process the file asynchronously, passing the original responsibilities dictionary
        tasks.append(
            process_file_async(
                input_file=sim_metrics_file,
                output_file=output_file,
                score_threshold=score_threshold,
                original_resps_dict=original_resps_dict,
            )
        )

    # Run all tasks concurrently
    await asyncio.gather(*tasks)
    logger.info("🎉 ✅ All crosstab files processed successfully!")
