import os
from pathlib import Path
import pandas as pd
import re
import logging
import logging_config
import json
from typing import Callable
from pydantic import ValidationError
from preprocessing.resume_preprocessor import ResumeParser
from preprocessing.requirements_preprocessor import JobRequirementsParser
from evaluation_optimization.text_similarity_finder import TextSimilarity
from evaluation_optimization.metrics_calculator import (
    calculate_many_to_many_similarity_metrices,
    categorize_scores_for_df,
    calculate_text_similarity_metrics,
)
from evaluation_optimization.multivariate_indexer import MultivariateIndexer
from utils.generic_utils import read_from_json_file, validate_json_file
from evaluation_optimization.evaluation_optimization_utils import (
    get_new_urls_and_file_names,
    get_new_urls_and_metrics_file_paths,
    get_files_wo_multivariate_indices,
)
from src.models.resume_and_job_description_models import ResponsibilityMatches

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


def generate_metrics_from_flat_json(
    reqs_flat_file: Path, resps_flat_file: Path, metrics_csv_file: Path
) -> None:
    """
    Generate similarity metrics between flattened responsibilities and requirements
    and save to a CSV file.

    Args:
        reqs_flat_file (Path): Path to the pre-flattened requirements JSON file.
        resps_flat_file (Path): Path to the pre-flattened responsibilities JSON file.
        csv_file (Path): Path where the output CSV file should be saved.

    Returns:
        None
    """
    # Step 1: Load flattened responsibilities from file
    try:
        resps_flat = read_from_json_file(resps_flat_file)
        reqs_flat = read_from_json_file(reqs_flat_file)

        # Check if either responsibilities or requirements are empty
        if not resps_flat or not reqs_flat:
            logger.error(
                "One of the required datasets (responsibilities or requirements) is empty."
            )
            return
        if not isinstance(resps_flat, dict) or not isinstance(reqs_flat, dict):
            logger.error(
                f"Responsibilities or Requirements data is valid dictionary: {type(resps_flat), type(reqs_flat)}"
            )
            return
    except FileNotFoundError as e:
        logger.error(f"File not found: {e.filename}")
        return
    except Exception as e:
        logger.error(f"Error loading files: {e}")
        return

    # Step 4: Calculate similarity metrics - Segment by Segment
    similarity_df = calculate_many_to_many_similarity_metrices(resps_flat, reqs_flat)
    logger.info("Similarity metrics calculated.")

    # Step 5: Add score category values (high, mid, low)
    similarity_df = categorize_scores_for_df(similarity_df)
    logger.info("Similarity metrics score categories created.")

    # Step 6: Clean up the data by removing newline characters from the DataFrame
    df = similarity_df.applymap(lambda x: str(x).replace("\n", " ").strip())

    # Step 7: Ensure the output directory exists and save the CSV file
    df.to_csv(metrics_csv_file, index=False)
    logger.info(f"Similarity metrics saved to file ({metrics_csv_file})")

    logger.info(
        "Finished running responsibility/requirement alignment scoring pipeline."
    )

    # Display the top rows of the DataFrame for verification
    print(df.head(5))


def generate_matching_metrics_from_nested_json(
    reqs_file: Path, resps_file: Path, metrics_csv_file: Path
) -> None:
    """
    Generate similarity metrics between nested responsibilities and requirements and save to a CSV file.

    Args:
        reqs_file (Path): Path to the requirements JSON file.
        resps_file (Path): Path to the nested responsibilities JSON file.
        metrics_csv_file (Path): Path where the output CSV file should be saved.

    Returns:
        None
    """
    # Step 1: Load and validate responsibilities and requirements using Pydantic models
    try:
        resps_data = ResponsibilityMatches.model_validate(resps_file)
        reqs_data = read_from_json_file(
            reqs_file
        )  # Loading requirements directly as a dict

        if not resps_data or not reqs_data:
            logger.error(
                "One of the required datasets (responsibilities or requirements) is empty."
            )
            return

    except FileNotFoundError as e:
        logger.error(f"File not found: {e.filename}")
        return
    except ValidationError as ve:
        logger.error(f"Validation error when parsing JSON files: {ve}")
        return
    except Exception as e:
        logger.error(f"Error loading files: {e}")
        return

    similarity_results = []

    # Step 2: Iterate through the responsibilities and requirements
    for responsibility_key, responsibility_match in resps_data.responsibilities.items():
        for (
            requirement_key,
            optimized_text_obj,
        ) in responsibility_match.optimized_by_requirements.items():
            optimized_text = optimized_text_obj.optimized_text

            # Look up the corresponding requirement using the requirement_key
            requirement_text = reqs_data.get(requirement_key)
            if not requirement_text:
                logger.warning(
                    f"No matching requirement found for requirement_key: {requirement_key}"
                )
                continue

            # Step 3: Calculate similarity metrics between responsibility and requirement
            similarity_metrics = calculate_text_similarity_metrics(
                optimized_text, requirement_text
            )

            # Step 4: Store the result for this responsibility/requirement pair
            similarity_results.append(
                {
                    "responsibility_key": responsibility_key,
                    "requirement_key": requirement_key,
                    "optimized_text": optimized_text,
                    "requirement_text": requirement_text,
                    "similarity_metrics": similarity_metrics,
                }
            )

    # Step 5: Convert results to a DataFrame and categorize scores
    similarity_df = pd.DataFrame(similarity_results)
    similarity_df = categorize_scores_for_df(similarity_df)

    # Step 6: Save the results to a CSV file
    similarity_df.to_csv(metrics_csv_file, index=False)
    logger.info(f"Similarity metrics saved to file ({metrics_csv_file})")

    # Display the top rows of the DataFrame for verification
    print(similarity_df.head(5))


def unpack_and_combine_json(nested_json, requirements_json):
    """
    Unpacks the nested responsibilities JSON and combines it with matching requirement texts
    from the requirements JSON. Outputs a list of dictionaries with responsibility and requirement texts.

    Args:
        nested_json (dict): JSON-like dictionary containing responsibility text structured in a nested format.
        requirements_json (dict): JSON-like dictionary containing requirement texts keyed by requirement IDs.

    Returns:
        list: A list of dictionaries containing responsibility_keys, requirement_keys,
              responsibility texts, and matched requirement texts.

    Error Handling:
        - If a requirement_key is not found in the requirements JSON, it will skip that entry.
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
                    f"Warning: requirement_key '{req_key}' not found in requirements JSON. Skipping entry."
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


def multivariate_indices_processing_mini_pipeline(data_directory: str):
    """
    A mini pipeline that processes CSV files in a directory, adding multivariate indices
    (composite and PCA scores) to files that are missing them.

    Parameters:
    - data_directory: The directory containing the CSV files to process.

    Raises:
    - ValueError if the directory does not exist.
    """
    if not os.path.exists(data_directory):
        raise ValueError(f"The provided directory '{data_directory}' does not exist.")
    try:
        # Find csv files in the right format (with metrics) but missing indices
        files_need_to_process = get_files_wo_multivariate_indices(data_directory)
        logger.info(f"Files that need processing: {files_need_to_process}")

        # Iterate to add composite and pca scores
        for file in files_need_to_process:
            try:
                df = pd.read_csv(file)
                df_wt_indices = add_multivariate_indices(df)
                df_wt_indices.to_csv(file, index=False)
                logger.info(f"Successfully processed and saved {file}")
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")

    except Exception as e:
        logger.error(f"Error during pipeline processing: {e}")

    logger.info(
        f"Successfully added multivariate indices to {len(files_need_to_process)} files."
    )


def metrics_processing_pipeline(
    mapping_file: str,
    generate_metrics: Callable[
        [Path, Path, Path], None
    ] = generate_metrics_from_flat_json,
) -> None:
    """
    Process and create missing sim_metrics files by reading from the mapping file.

    Args:
        mapping_file (str or Path): Path to the JSON mapping file.
        generate_metrics_func (callable): Function to generate the metrics CSV file.

    Returns:
        None
    """
    logger.info("Start running responsibility/requirement alignment scoring pipeline.")

    # Step 1.0: Read the mapping file
    try:
        # Step 1: Read and validate the mapping file
        file_mapping = read_from_json_file(mapping_file)
        if not isinstance(file_mapping, dict):
            raise ValueError(f"Expected a dictionary but got {type(file_mapping)}")

        logger.info(f"Loaded and validated mapping file from {mapping_file}")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error with mapping file {mapping_file}: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error loading mapping file: {e}")
        return

    # Step 2: Check if all sim_metrics files exist
    missing_metrics = {
        url: mapping["sim_metrics"]
        for url, mapping in file_mapping.items()
        if not Path(mapping["sim_metrics"]).exists()
    }

    # *Check if metrics are already processed (no missing files)
    if not missing_metrics:
        logger.info("All sim_metrics files already exist. Exiting pipeline.")
        return  # Early exit if all files exist

    # Step 3: Process missing sim_metrics files
    for url, sim_metrics_file in missing_metrics.items():
        logger.info(f"Processing missing metrics for {url}")

        # Load the requirements and responsibilities files from the mapping
        reqs_file = Path(file_mapping[url]["reqs"])
        resps_file = Path(file_mapping[url]["resps"])

        # Step 3.1: Check if reqs and resps files exist
        if not reqs_file.exists() or not resps_file.exists():
            logger.error(
                f"Missing requirements or responsibilities files for {url}. Skipping."
            )
            continue

        # Step 3.2: Generate the metrics file
        generate_metrics(reqs_file, resps_file, sim_metrics_file)
        logger.info(f"Generated metrics for {url} and saved to {sim_metrics_file}")

    logger.info("Finished processing all missing sim_metrics files.")


# Pipeline to process eval for modified responsibilities
def metrics_reprocessing_pipeline(
    mapping_file,
    generate_metrics: Callable[
        [Path, Path, Path], None
    ] = generate_matching_metrics_from_nested_json,
) -> None:
    """
    Re-run pipeline to process and create missing sim_metrics files by reading from
    the mapping file.

    Args:
        mapping_file (str or Path): Path to the JSON mapping file.
        generate_metrics_func (Callable[[Path, Path, Path], None]): Function to generate
        the metrics CSV file.

    Returns:
        None
    """
    try:
        file_mapping = read_from_json_file(mapping_file)
        if not isinstance(file_mapping, dict):
            raise ValueError(f"Expected a dictionary but got {type(file_mapping)}")

        logger.info(f"Loaded and validated mapping file from {mapping_file}")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error with mapping file {mapping_file}: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error loading mapping file: {e}")
        return

    # Step 2: Check if all sim_metrics files exist
    missing_metrics = {
        url: mapping["sim_metrics"]
        for url, mapping in file_mapping.items()
        if not Path(mapping["sim_metrics"]).exists()
    }

    if not missing_metrics:
        logger.info("All sim_metrics files already exist. Exiting pipeline.")
        return

    # Step 3: Process missing sim_metrics files
    for url, sim_metrics_file in missing_metrics.items():
        logger.info(f"Processing missing metrics for {url}")

        # Load the requirements and responsibilities files from the mapping
        reqs_file = Path(file_mapping[url]["reqs"])
        resps_file = Path(file_mapping[url]["resps"])

        # Step 3.1: Check if reqs and resps files exist and are valid JSON files
        if not validate_json_file(reqs_file) or not validate_json_file(resps_file):
            logger.error(f"Skipping {url} due to invalid files.")
            continue

        # Step 3.2: Generate the metrics file
        generate_metrics(reqs_file, resps_file, sim_metrics_file)
        logger.info(f"Generated metrics for {url} and saved to {sim_metrics_file}")

    logger.info("Finished processing all missing sim_metrics files.")


# def metrics_preprocessing_mini_pipeline(job_descriptions_file, output_dir):
#     """
#     Preprocess job descriptions for evaluation by identifying new URLs to process.

#     Args:
#         job_descriptions_file (str or Path): Path to the JSON file containing job descriptions.
#         output_dir (str or Path, optional): Directory where output files are stored.
#             Defaults to METRICS_OUTPUTS_CSV_FILES_DIR / "iteration_0".

#     Returns:
#         Dict: A dictionary of new URLs that need to be processed and their file paths to be saved.
#     """
#     job_descriptions = read_from_json_file(job_descriptions_file)
#     if not job_descriptions:
#         logger.error("No job descriptions loaded. Exiting.")
#         return {}

#     new_urls_and_f_names = get_new_urls_and_metrics_file_paths(
#         job_descriptions, output_dir
#     )
#     logger.info(f"Found {len(new_urls_and_f_names)} new URLs to process.")
#     return new_urls_and_f_names  # a dict


# def metrics_processing_pipeline(
#     url: str, requirements_json_file: str, resume_json_file: str, csv_file: str
# ):

#     logger.info("Start running responsibility/requirement alignment scoring pipeline.")

#     # Step 1: Parse and flatten responsibilities from resume (as a dict)
#     resume_parser = ResumeParser(resume_json_file)
#     resps_flat = resume_parser.extract_and_flatten_responsibilities()  # dict

#     # Step 2: Parse and flatten job requirements (as a dict) or
#     # parse/flatten/concatenate into a single string
#     job_reqs_parser = JobRequirementsParser(requirements_json_file, url)
#     reqs_flat = job_reqs_parser.extract_flatten_reqs()  # dict

#     # Check if either responsibilities or requirements are empty
#     if not resps_flat or not reqs_flat:
#         logger.error(
#             "One of the required datasets (responsibilities or requirements) is empty."
#         )
#         return

#     # Step 3. Calculate and display similarity metrics - Segment by Segment
#     similarity_df = calculate_many_to_many_similarity_metrices(resps_flat, reqs_flat)
#     logger.info("Similarity metrics calculated.")  # Changed to logger

#     # Step 4. Add score category values (high, mid, low)
#     # Translate DataFrame columns to match expected column names
#     similarity_df = categorize_scores_for_df(similarity_df)
#     logger.info("Similarity metrics score categories created.")

#     # Step 5. Clean up the data by removing newline characters from the DataFrame
#     df = similarity_df.applymap(lambda x: str(x).replace("\n", " ").strip())

#     # Step 6. Ensure the output directory exists and save the CSV file
#     df.to_csv(csv_file, index=False)  # Save to the correct path
#     logger.info(f"Similarity metrics saved to file ({csv_file})")

#     logger.info(
#         "Finished running responsibility/requirement alignment scoring pipeline."
#     )

#     # Display the top rows of the DataFrame for verification
#     print(df.head(5))  # Updated display to print for non-interactive environments
#     logger.info(
#         f"Finished running responsibility/requirement alignment scoring pipeline for {url}."
#     )
#     print("\n\n")


# def get_metrics_file_paths(mapping_file, output_dir):
#     """
#     Get a dictionary of URLs and their corresponding metrics file paths.

#     Args:
#         mapping_file_path (str or Path): Path to the JSON mapping file.
#         output_dir (str or Path): Directory where the new metrics files will be created.

#     Returns:
#         dict: A dictionary where keys are URLs and values are the paths to the metrics files.
#     """

#     sim_metrics_mapping_dict = {
#         url: data["sim_metrics"] for url, data in mapping_dict.items()
#     }
#     return sim_metrics_mapping_dict
