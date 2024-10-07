import os
from pathlib import Path
import pandas as pd

import logging
import logging_config

import asyncio
import aiofiles

from preprocessing.resume_preprocessor import ResumeParser
from preprocessing.requirements_preprocessor import JobRequirementsParser
from evaluation_optimization.text_similarity_finder import TextSimilarity
from evaluation_optimization.metrics_calculator import (
    calculate_many_to_many_similarity_metrices,
    categorize_scores_for_df,
    SimilarityScoreCalculator,
)
from evaluation_optimization.multivariate_indexer import MultivariateIndexer
from utils.generic_utils_async import (
    read_from_json_async,
    read_from_csv_async,
    save_to_csv_async,
)
from utils.generic_utils import pretty_print_json, get_company_and_job_title_from_json
from config import METRICS_OUTPUTS_CSV_FILES_DIR, job_descriptions_json_file
from evaluation_optimization.evaluation_optimization_utils import (
    get_new_urls_and_metrics_file_paths,
    get_new_urls_and_flat_json_file_paths,
    find_files_wo_multivariate_indices,
)


# Set up logger
logger = logging.getLogger(__name__)


def add_multivariate_indices(df):
    """
    Add multivariate indices to the DataFrame using the MultivariateIndexer.

    Parameters:
    - df: The input DataFrame.

    Returns:
    - DataFrame with multivariate indices added, or raises an appropriate error.
    """
    try:
        # Check if df is a valid DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input is not a valid DataFrame.")

        # Check if df is empty
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Initialize the MultivariateIndexer and add indices
        indexer = MultivariateIndexer(df)
        df = indexer.add_multivariate_indices_to_df()

        logger.info("Multivariate indices added.")
        return df

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except AttributeError as ae:
        print(f"AttributeError: {ae}. Ensure MultivariateIndexer and its method exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


def unpack_and_combine_json(nested_json, requirements_json):
    """
    Unpacks the nested responsibilities JSON and combines it with matching requirement texts
    from the requirements JSON. Outputs a list of dictionaries with responsibility and requirement texts.

    Args:
        nested_json (dict): JSON-like dictionary containing responsibility text structured in a nested format.
        requirements_json (dict): JSON-like dictionary containing requirement texts keyed by requirement IDs.

    Returns:
        list: A list of dictionaries containing responsibility keys, requirement keys,
              responsibility texts, and matched requirement texts.

    Error Handling:
        - If a requirement key is not found in the requirements JSON, it will skip that entry.
        - If a required field (e.g., 'optimized_text') is missing, it will skip that entry.
        - Logs warnings for missing fields and unmatched keys for better traceability.
    """
    results = []

    for resp_key, values in nested_json.items():  # Unpack the 1st level
        if not isinstance(values, dict):
            logger.info(
                f"Warning: Unexpected data structure under '{resp_key}'. Skipping entry."
            )
            continue

        for req_key, sub_value in values.items():  # Unpack the 2nd level
            if not isinstance(sub_value, dict):
                logger.info(
                    f"Warning: Unexpected data structure under '{req_key}'. Skipping entry."
                )
                continue

            # Extract the optimized_text
            if "optimized_text" in sub_value:
                optimized_text = sub_value["optimized_text"]
            else:
                logger.info(
                    f"Warning: Missing 'optimized_text' for '{req_key}'. Skipping entry."
                )
                continue

            # Perform the lookup in the requirements JSON
            requirement_text = requirements_json.get(req_key)
            if requirement_text is None:
                logger.info(
                    f"Warning: Requirement key '{req_key}' not found in requirements JSON. Skipping entry."
                )
                continue

            # Append results to a list for further processing
            results.append(
                {
                    "responsibility_key": resp_key,
                    "requirement_key": req_key,
                    "responsibility_text": optimized_text,
                    "requirement_text": requirement_text,
                }
            )

    return results


async def metrics_preprocessing_mini_pipeline_async(
    job_descriptions_file, output_dir=METRICS_OUTPUTS_CSV_FILES_DIR / "iteration_0"
):
    """
    Asynchronous preprocess job descriptions for evaluation by identifying new URLs to process.

    Args:
        job_descriptions_file (str or Path): Path to the JSON file containing job descriptions.
        output_dir (str or Path, optional): Directory where output files are stored.
            Defaults to METRICS_OUTPUTS_CSV_FILES_DIR / "iteration_0".

    Returns:
        Dict: A dictionary of new URLs that need to be processed and their file paths to be saved.
    """
    job_descriptions = await read_from_json_async(job_descriptions_file)
    if not job_descriptions:
        logger.error("No job descriptions loaded. Exiting.")
        return {}

    new_urls_and_f_names = get_new_urls_and_metrics_file_paths(
        job_descriptions, output_dir
    )
    logger.info(f"Found {len(new_urls_and_f_names)} new URLs to process.")
    return new_urls_and_f_names  # a dict


async def metrics_processing_pipeline_async(
    url: str, requirements_json_file: str, resume_json_file: str, csv_file: str
):
    """
    1st time run pipeline to calculate responsibility vs requirement alignment scores, asynchronously.

    Args:
        - url (str): URL of the job posting
        (serves as the unique identifier to extract data from a single job posting in the requirement JSON file)
        - requirements_json_file (str): Path of the JSON file containing all the job requirements
        (from job sites)
        - resume_json_file (str): Path of the JSON file containing the resume file
        - csv_file (str): File path of the metric results in CSV (resume responsibilities vs job requirements of
        a single job posting)
    """

    logger.info("Start running responsibility/requirement alignment scoring pipeline.")

    # Step 1: Parse and flatten responsibilities from resume (as a dict)
    resume_parser = ResumeParser(resume_json_file)
    resps_flat = resume_parser.extract_and_flatten_responsibilities()  # dict

    # Step 2: Parse and flatten job requirements (as a dict) or
    # parse/flatten/concatenate into a single string
    job_reqs_parser = JobRequirementsParser(requirements_json_file, url)
    reqs_flat = job_reqs_parser.extract_flatten_reqs()  # dict

    # Check if either responsibilities or requirements are empty
    if not resps_flat or not reqs_flat:
        logger.error(
            "One of the required datasets (responsibilities or requirements) is empty."
        )
        return

    # Step 3. Calculate and display similarity metrics - Segment by Segment
    similarity_df = calculate_many_to_many_similarity_metrices(resps_flat, reqs_flat)
    logger.info("Similarity metrics calculated.")  # Changed to logger

    # Step 4. Add score category values (high, mid, low)
    # Translate DataFrame columns to match expected column names
    similarity_df = categorize_scores_for_df(similarity_df)
    logger.info("Similarity metrics score categories created.")

    # Step 5. Clean up the data by removing newline characters from the DataFrame
    df = similarity_df.applymap(lambda x: str(x).replace("\n", " ").strip())

    # Step 6. Save the CSV file
    async with aiofiles.open(csv_file, "w") as f:
        df.to_csv(csv_file, index=False)  # Save to the correct path
    logger.info(f"Similarity metrics saved to file ({csv_file})")

    logger.info(
        f"Finished running responsibility/requirement alignment scoring pipeline for {url}."
    )
    print("\n\n")


async def multivariate_indices_processing_mini_pipeline_async(data_directory: str):
    """
    Asynchronous mini pipeline that:
    - processes CSV files in a directory, and
    - adds multivariate indices (composite and PCA scores) to files that are missing them.

    Parameters:
    - data_directory: The directory containing the CSV files to process.

    Raises:
    - ValueError if the directory does not exist.
    """
    if not os.path.exists(data_directory):
        raise ValueError(f"The provided directory '{data_directory}' does not exist.")
    try:
        # Find csv files in the right format (with metrics) but missing indices
        logger.info(f"Checking files in {data_directory}")
        files_need_to_process = find_files_wo_multivariate_indices(data_directory)
        logger.info(f"Files that need processing: {files_need_to_process}")

        # Iterate to add composite and pca scores
        for file in files_need_to_process:
            try:
                # Read CSV asynchronously
                df = await read_from_csv_async(file)

                # Add composite scores and PCA scores (multivariate indices)
                df_wt_indices = add_multivariate_indices(df)

                # Write CSV async
                await save_to_csv_async(df_wt_indices, file)
                logger.info(f"Successfully processed and saved {file}")
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")

    except Exception as e:
        logger.error(f"Error during pipeline processing: {e}")

    logger.info(
        f"Successfully added multivariate indices to {len(files_need_to_process)} files."
    )


def re_processing_metrics_pipeline(requirements_file, responsibilities_file, csv_file):
    """
    Re-run pipeline to calculate revised responsibility vs (job) requirement alignment scores
    """

    # Check if resps comparison csv file (the results) ALREADY exists
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)

        print(f"Similarity Metrics Dataframe: {df.head(10)}")

        logger.info("Output already exists, skipping pipeline.")
        return  # Early exit if output exists

    logger.info("Start running responsibility/requirement alignment scoring pipeline.")

    # Step 1: Read JSON files
    resps_dict = read_from_json_async(responsibilities_file)  # Nested dictionary
    reqs_dict = read_from_json_async(requirements_file)

    # Step 2: parse and combine 2 into a single list/dict
    combined_list = unpack_and_combine_json(
        nested_json=resps_dict, requirements_json=reqs_dict
    )

    # Step 3. Iterate through the Calculate and display similarity metrices - Segment by Segment
    similarity_calculator = SimilarityScoreCalculator()
    similarity_df = similarity_calculator.one_to_one(combined_list)

    logger.info("Similarity metrics calcuated.")

    # Step 4. Add score category values (high, mid, low)
    # Translate DataFrame columns to match expected column names**
    similarity_df = categorize_scores_for_df(similarity_df)
    logger.info("Similarity metrics score categories created.")

    # Step 5. Clean and save to csv
    # df_cleaned = bscore_p_df.applymap(lambda x: str(x).replace("\n", " ").strip())
    df = similarity_df.applymap(lambda x: str(x).replace("\n", " ").strip())
    df.to_csv(csv_file, index=False)

    logging.info(f"Similarity metrics saved to location {csv_file}.")

    # Print out for debugging
    print(f"Similarity Metrics Dataframe: {df.head(10)}")

    logger.info(f"Similarity scores saved to csv file ({csv_file})")
