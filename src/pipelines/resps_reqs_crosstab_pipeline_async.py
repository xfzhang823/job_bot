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
from evaluation_optimization.create_mapping_file import load_mappings_model_from_json
from evaluation_optimization.evaluation_optimization_utils import create_file_name
from human_review_and_editing.create_resp_req_crosstab import (
    create_resp_req_crosstab,
    save_crosstab,
)
from project_config import (
    RESPS_REQS_MATCHING_DIR,
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


### ✅ Data Merging Function ###
def merge_job_postings_mapping_file(
    df_job_postings: pd.DataFrame, df_mapping: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge job postings and mapping DataFrames on 'url'.

    Args:
        df_job_postings (pd.DataFrame): Job postings DataFrame.
        df_mapping (pd.DataFrame): Mapping file DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame with matched company, job title, and similarity metrics file paths.
    """
    merged_df = df_job_postings.merge(df_mapping, on="url", how="left").reset_index(
        drop=True
    )

    # ✅ Fill missing values for company and job title
    merged_df.loc[merged_df["company"].isna(), "company"] = "Unknown_Company"
    merged_df.loc[merged_df["job_title"].isna(), "job_title"] = "Unknown_Title"

    logger.info(f"✅ Merged DataFrame:\n{merged_df.head()}")
    return merged_df


### ✅ Async Processing Function ###
async def process_file_async(
    input_file: Path, output_file: Path, score_threshold: float = 0
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
        )

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
    cross_tab_output_dir: Path = RESPS_REQS_MATCHING_DIR,
    score_threshold: float = 0,
) -> None:
    """
    Asynchronously process all similarity metrics files and generate cross-tab Excel reports.

    Args:
        mapping_file (Path): Path to the JSON mapping file.
        cross_tab_output_dir (Path): Directory where cross-tab Excel files will be saved.
        score_threshold (float, optional): Threshold for filtering responsibility text.

    Returns:
        None
    """
    logger.info(f"📂 Loading mapping file: {mapping_file}")

    # ✅ Ensure output directory exists
    cross_tab_output_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Load job postings & mapping file
    df_job_postings = load_job_postings_to_df(JOB_POSTING_URLS_FILE)
    df_mapping = load_mapping_file_to_df(mapping_file)

    # ✅ Merge DataFrames on 'url'
    merged_df = merge_job_postings_mapping_file(df_job_postings, df_mapping)

    # ✅ Identify similarity metrics CSV files
    tasks = []
    for _, row in merged_df.iterrows():
        sim_metrics_file = Path(row.at["sim_metrics"])

        if not sim_metrics_file.exists():
            logger.warning(f"⚠️ Skipping: {sim_metrics_file} (File not found)")
            continue

        company, job_title = row.at["company"], row.at["job_title"]

        # ✅ Generate output file name
        output_filename = create_file_name(
            company, job_title, suffix="_crosstab", ext=".xlsx"
        )

        # ✅ Ensure it's always a string
        if not output_filename or output_filename is None:
            logger.warning(
                f"⚠️ Filename could not be generated for {company}, {job_title}. Using default."
            )
            output_filename = "default_filename.xlsx"

        output_file = cross_tab_output_dir / output_filename

        # ✅ Process the file asynchronously
        tasks.append(process_file_async(sim_metrics_file, output_file, score_threshold))

    # ✅ Run tasks concurrently
    await asyncio.gather(*tasks)
    logger.info("🎉 ✅ All crosstab files processed successfully!")
