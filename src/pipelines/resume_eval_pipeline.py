import os
from pathlib import Path
import pandas as pd
import re
import logging
import logging_config
import json
import pandas as pd
from pydantic import ValidationError, HttpUrl
from typing import Callable, Union, Optional
from models.resume_job_description_io_models import JobFileMappings
from preprocessing.resume_preprocessor import ResumeParser
from preprocessing.requirements_preprocessor import JobRequirementsParser
from evaluation_optimization.text_similarity_finder import TextSimilarity
from evaluation_optimization.metrics_calculator import (
    calculate_many_to_many_similarity_metrices,
    categorize_scores_for_df,
    calculate_text_similarity_metrics,
)
from evaluation_optimization.multivariate_indexer import MultivariateIndexer
from evaluation_optimization.create_mapping_file import load_mappings_model_from_json
from utils.generic_utils import read_from_json_file, validate_json_file
from evaluation_optimization.evaluation_optimization_utils import (
    get_new_urls_and_file_names,
    get_new_urls_and_metrics_file_paths,
    get_files_wo_multivariate_indices,
)
from models.resume_job_description_io_models import (
    ResponsibilityMatches,
    Responsibilites,
    Requirements,
    SimilarityMetrics,
    PipelineInput,
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


# *Processing the dataframe first and then validate w/t pydantic
def generate_metrics_from_flat_json(
    reqs_flat_file: Path,
    resps_flat_file: Path,
    metrics_csv_file: Path,
    calculate_metrics: Callable[
        [dict, dict], pd.DataFrame
    ] = calculate_many_to_many_similarity_metrices,
    categorize_scores: Callable[
        [pd.DataFrame], pd.DataFrame
    ] = categorize_scores_for_df,
) -> None:
    """
    Generate similarity metrics between flattened responsibilities and requirements
    and save them to a CSV file after validating the input data.

    Args:
        reqs_flat_file (Path): Path to the pre-flattened requirements JSON file.
        resps_flat_file (Path): Path to the pre-flattened responsibilities JSON file.
        metrics_csv_file (Path): Path where the output CSV file should be saved.
        calculate_similarity (Callable[[dict, dict], pd.DataFrame], optional):
            Function to calculate similarity metrics. Defaults to
            calculate_many_to_many_similarity_metrices.
        categorize_scores (Callable[[pd.DataFrame], pd.DataFrame], optional):
            Function to categorize similarity scores. Defaults to
            categorize_scores_for_df.

    Returns:
        None

    Logs:
        - Info messages for successful operations.
        - Error messages for validation failures or processing issues.
    """
    # Step 1: Load flattened responsibilities from file
    try:
        resps_flat_data = read_from_json_file(resps_flat_file)
        validated_resps = Responsibilites(**resps_flat_data)
        resps_flat = validated_resps.responsibilities
        logger.info(f"Validated responsibilities from {resps_flat_file}")
    except ValidationError as ve:
        logger.error(f"Validated responsibilities validation error: {ve}")
        return
    except FileNotFoundError as fe:
        logger.error(f"Responsibilities file not found: {fe.filename}")
        return
    except Exception as e:
        logger.error(f"Unexpected error loading responsibilities: {e}")
        return

    # Step 2: Load and Validate Requirements
    try:
        reqs_flat_data = read_from_json_file(reqs_flat_file)
        validated_reqs = Requirements(**reqs_flat_data)
        reqs_flat = validated_reqs.requirements
        logger.info(f"Validated requirements from {reqs_flat_file}")
    except ValidationError as ve:
        logger.error(f"Requirements validation error: {ve}")
        return
    except FileNotFoundError as fe:
        logger.error(f"Requirements file not found: {fe.filename}")
        return
    except Exception as e:
        logger.error(f"Unexpected error loading requirements: {e}")
        return

    # Step 3: Check for Empty Datasets
    if not resps_flat or not reqs_flat:
        logger.error(
            "One of the required datasets (responsibilities or requirements) is empty."
        )
        return

    # Step 4: Calculate similarity metrics - Segment by Segment
    try:
        similarity_df = calculate_metrics(resps_flat, reqs_flat)
        logger.info("Similarity metrics calculated.")
    except Exception as e:
        logger.error(f"Error calculating similarity metrics: {e}")
        return

    # Step 5: Add score category values (high, mid, low)
    try:
        categorized_df = categorize_scores(similarity_df)
        logger.info("Similarity metrics score categories created.")
    except Exception as e:
        logger.error(f"Error categorizing similarity scores: {e}")
        return

    # Step 6: Validate similarity metrics rows
    try:
        # Iterate through each row and validate using SimilarityMetrics model
        validated_rows = []
        for index, row in categorized_df.iterrows():
            try:
                similarity_metrics = SimilarityMetrics(**row.to_dict())
                validated_rows.append(similarity_metrics.model_dump())
            except ValidationError as ve:
                logger.error(
                    f"Similarity metrics validation error at row {index}: {ve}"
                )
                continue  # Skip invalid rows

            # Convert validated rows back to DataFrame
        if validated_rows:
            final_df = pd.DataFrame(validated_rows)
            logger.info("All similarity metrics rows validated successfully.")
        else:
            logger.error("No valid similarity metrics data to save.")
            return
    except Exception as e:
        logger.error(f"Error during similarity metrics validation: {e}")
        return

    # Step 7: Save the Validated Metrics to CSV
    try:
        # Ensure the output directory exists
        metrics_csv_file.parent.mkdir(parents=True, exist_ok=True)

        # Save the DataFrame to CSV
        final_df.to_csv(metrics_csv_file, index=False)
        logger.info(f"Similarity metrics saved successfully to {metrics_csv_file}")
    except Exception as e:
        logger.error(f"Error saving similarity metrics to CSV: {e}")
        return


# *add metrics by processing each row of the dataframe, validate w/t pydantic
# * and then append the rows to form the dataframe.
def generate_matching_metrics_from_nested_json(
    reqs_file: Union[Path, str],
    resps_file: Union[Path, str],
    metrics_csv_file: Union[Path, str],
) -> None:
    """
    Generate similarity metrics between nested responsibilities and requirements
    and save to a CSV file.

    Args:
        reqs_file (Path or str): Path to the requirements JSON file.
        resps_file (Path or str): Path to the nested responsibilities JSON file.
        metrics_csv_file (Path or str): Path where the output CSV file should be saved.

    Returns:
        None
    """
    # Step 0: Ensure inputs are Path obj.
    reqs_file = Path(reqs_file)
    resps_file = Path(resps_file)
    metrics_csv_file = Path(metrics_csv_file)

    # Step 1: Load and validate responsibilities and requirements using Pydantic models
    try:
        # Read and validate responsibilities
        resps_data = read_from_json_file(resps_file)
        validated_resps_data = ResponsibilityMatches.model_validate(resps_data)

        # Read and validate requirments
        reqs_data = read_from_json_file(
            reqs_file
        )  # Loading requirements directly as a dict
        validated_reqs_data = Requirements.model_validate(reqs_data)

        if not validated_resps_data or not validated_reqs_data:
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

    validated_rows = []

    # Step 2: Iterate through the responsibilities and requirements
    for (
        responsibility_key,
        responsibility_match,
    ) in validated_resps_data.responsibilities.items():
        for (
            requirement_key,
            optimized_text_obj,
        ) in responsibility_match.optimized_by_requirements.items():
            responsibility_text = optimized_text_obj.optimized_text

            # Access the corresponding requirement using the Pydantic model
            try:
                requirement_text = validated_reqs_data.requirements[requirement_key]
            except KeyError:
                logger.warning(
                    f"No matching requirement found for requirement_key: {requirement_key}"
                )
                continue

            # Step 3: Calculate similarity metrics between responsibility and requirement
            similarity_metrics = calculate_text_similarity_metrics(
                responsibility_text, requirement_text
            )

            # Step 4: Validate using SimilarityMetrics model
            try:
                similarity_metrics_model = SimilarityMetrics(
                    responsibility_key=responsibility_key,
                    responsibility=responsibility_text,
                    requirement_key=requirement_key,
                    requirement=requirement_text,
                    bert_score_precision=similarity_metrics[
                        "bert_score_precision"
                    ],  # Explicit mapping
                    soft_similarity=similarity_metrics[
                        "soft_similarity"
                    ],  # Explicit mapping
                    word_movers_distance=similarity_metrics[
                        "word_movers_distance"
                    ],  # Explicit mapping
                    deberta_entailment_score=similarity_metrics[
                        "deberta_entailment_score"
                    ],  # Explicit mapping
                    # Optional fields (if present in similarity metrics)
                    bert_score_precision_cat=similarity_metrics.get(
                        "bert_score_precision_cat"
                    ),
                    soft_similarity_cat=similarity_metrics.get("soft_similarity_cat"),
                    word_movers_distance_cat=similarity_metrics.get(
                        "word_movers_distance_cat"
                    ),
                    deberta_entailment_score_cat=similarity_metrics.get(
                        "deberta_entailment_score_cat"
                    ),
                    scaled_bert_score_precision=similarity_metrics.get(
                        "scaled_bert_score_precision"
                    ),
                    scaled_deberta_entailment_score=similarity_metrics.get(
                        "scaled_deberta_entailment_score"
                    ),
                    scaled_soft_similarity=similarity_metrics.get(
                        "scaled_soft_similarity"
                    ),
                    scaled_word_movers_distance=similarity_metrics.get(
                        "scaled_word_movers_distance"
                    ),
                    composite_score=similarity_metrics.get("composite_score"),
                    pca_score=similarity_metrics.get("pca_score"),
                )
                validated_rows.append(similarity_metrics_model.model_dump())

            except ValidationError as ve:
                logger.error(
                    f"Validation error for responsibility {responsibility_key}: {ve}"
                )
                continue

    # Step 5: Convert validated results to a DataFrame and categorize scores
    if validated_rows:
        final_df = pd.DataFrame(validated_rows)
        final_df = categorize_scores_for_df(final_df)

        # Step 6: Save the validated metrics to a CSV file
        final_df.to_csv(metrics_csv_file, index=False)
        logger.info(f"Similarity metrics saved successfully to {metrics_csv_file}")
    else:
        logger.error("No valid similarity metrics data to save.")
        return

    # Display the top rows of the DataFrame for verification
    print(final_df.head(5))


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


def multivariate_indices_processing_mini_pipeline(
    data_directory: Union[str, Path],
    validate_input: bool = True,
    add_indices_func: Callable[[pd.DataFrame], pd.DataFrame] = add_multivariate_indices,
) -> None:
    """
    A mini pipeline that processes CSV files in a directory, adding multivariate indices
    (composite and PCA scores) to files that are missing them.

    Args:
        - data_directory (str | Path): The directory containing the CSV files to process.
        - validate_input (bool, optional): Whether to validate the input directory. Defaults to True.
        - add_indices_func (Callable[[pd.DataFrame], pd.DataFrame], optional):
            Function to add multivariate indices to the DataFrame.
            Defaults to add_multivariate_indices.

    Raises:
    - ValueError if the directory does not exist.
    """
    # Step 1: Validate the Input Directory
    if validate_input:
        try:
            pipeline_input = PipelineInput(data_directory=data_directory)
            data_directory = pipeline_input.data_directory  # Now a Path object
            logger.info(f"Validated data directory: {data_directory}")
        except ValidationError as ve:
            logger.error(f"Input validation error: {ve}")
            raise ValueError(f"Invalid input directory: {ve}") from ve

    try:
        # Step 2: Find CSV files missing multivariate indices
        files_need_to_process = get_files_wo_multivariate_indices(data_directory)
        logger.info(f"Files that need processing: {files_need_to_process}")

        # *Explicit Check for Empty List
        if not files_need_to_process:
            logger.info("No files require processing. Exiting pipeline.")
            return  # Early exit

        # Step 3: Iterate to add composite and PCA scores
        for file in files_need_to_process:
            logger.info(f"Processing file: {file}")
            try:
                df = pd.read_csv(file)

                # Step 4.1: Validate Each Row in the DataFrame
                validated_rows = []
                for index, row in df.iterrows():
                    try:
                        # Convert the row to a dictionary
                        row_dict = row.to_dict()
                        # Validate the row using the SimilarityMetrics model
                        validated_row = SimilarityMetrics(**row_dict)
                        # Append the validated row as a dictionary
                        validated_rows.append(validated_row.dict())
                    except ValidationError as ve:
                        logger.error(
                            f"Validation error in file '{file}', row {index}: {ve}"
                        )
                        continue  # Skip invalid rows

                if not validated_rows:
                    logger.warning(
                        f"No valid data to process in file '{file}'. Skipping."
                    )
                    continue  # Skip to the next file

                # Convert validated rows back to a DataFrame
                validated_df = pd.DataFrame(validated_rows)
                logger.info(f"Validated data for file '{file}'.")

                # Step 4.2: Add Multivariate Indices
                updated_df = add_indices_func(validated_df)
                logger.info(f"Added multivariate indices to file '{file}'.")

                # Step 4.3: Save the Updated DataFrame to CSV
                updated_df.to_csv(file, index=False)
                logger.info(f"Successfully processed and saved '{file}'.")

            except FileNotFoundError as fe:
                logger.error(f"File not found: {fe.filename}. Skipping.")
                continue
            except pd.errors.EmptyDataError:
                logger.error(f"No data found in file '{file}'. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing file '{file}': {e}")
                continue

    except Exception as e:
        logger.error(f"Error during pipeline processing: {e}")
        raise  # Re-raise the exception after logging

    logger.info(
        f"Successfully added multivariate indices to {len(files_need_to_process)} file(s)."
    )

    # try:
    #     # Step 3: Find CSV files missing multivariate indices
    #     files_need_to_process = get_files_wo_multivariate_indices(data_directory)
    #     logger.info(f"Files that need processing: {files_need_to_process}")

    #     # Step 4: Iterate to add composite and PCA scores
    #     for file in files_need_to_process:
    #         logger.info(f"Processing file: {file}")
    #         try:
    #             df = pd.read_csv(file)
    #     # Iterate to add composite and pca scores
    #     for file in files_need_to_process:
    #         try:
    #             df = pd.read_csv(file)
    #             df_wt_indices = add_multivariate_indices(df)
    #             df_wt_indices.to_csv(file, index=False)
    #             logger.info(f"Successfully processed and saved {file}")
    #         except Exception as e:
    #             logger.error(f"Error processing {file}: {e}")

    # except Exception as e:
    #     logger.error(f"Error during pipeline processing: {e}")

    # logger.info(
    #     f"Successfully added multivariate indices to {len(files_need_to_process)} files."
    # )


# Running pipeline with pydantic model validation
def metrics_processing_pipeline(
    mapping_file: Union[str, Path],
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

    # Ensure that mapping_file is turned into Path obj (if not)
    mapping_file = Path(mapping_file)

    # Step 1: Read and validate the mapping file
    file_mappings_model: Optional[JobFileMappings] = load_mappings_model_from_json(
        mapping_file
    )

    if file_mappings_model is None:
        logger.error("Failed to load and validate the mapping file. Exiting pipeline.")
        return

    file_mappings_dict = file_mappings_model.root

    # Step 2: Check if all sim_metrics files exist
    missing_metrics = {
        url: mapping.sim_metrics
        for url, mapping in file_mappings_dict.items()
        if not Path(mapping.sim_metrics).exists()
    }

    # *Check if metrics are already processed (no missing files)
    if not missing_metrics:
        logger.info("All sim_metrics files already exist. Exiting pipeline.")
        return  # Early exit if all files exist

    # Step 3: Process missing sim_metrics files
    for url, sim_metrics_file in missing_metrics.items():
        logger.info(f"Processing missing metrics for {url}")

        # Load the requirements and responsibilities files from the mapping
        reqs_file = Path(file_mappings_dict[url].reqs)
        resps_file = Path(file_mappings_dict[url].resps)

        # Convert sim_metrics_file to Path object if it's not already
        sim_metrics_file = Path(sim_metrics_file)

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


# Pipeline to process eval again for modified responsibilities
def metrics_re_processing_pipeline(
    mapping_file,
    generate_metrics: Callable[
        [Path, Path, Path], None
    ] = generate_matching_metrics_from_nested_json,
) -> None:
    """
    Re-run the pipeline to process and create missing sim_metrics files by reading from the mapping file.

    Args:
        mapping_file (str | Path): Path to the JSON mapping file.
        generate_metrics (Callable[[Path, Path, Path], None], optional):
            Function to generate the metrics CSV file. Defaults to
            generate_matching_metrics_from_nested_json.

    Returns:
        None
    """
    # Step 1: Read / Validate file mapping
    file_mappings_model: Optional[JobFileMappings] = load_mappings_model_from_json(
        mapping_file
    )

    if file_mappings_model is None:
        logger.error("Failed to load and validate the mapping file. Exiting pipeline.")
        return

    file_mappings_dict = file_mappings_model.root

    # Step 2: Check if all sim_metrics files exist
    missing_metrics = {
        url: mapping.sim_metrics
        for url, mapping in file_mappings_dict.items()
        if not Path(mapping.sim_metrics).exists()
    }

    if not missing_metrics:
        logger.info("All sim_metrics files already exist. Exiting pipeline.")
        return

    # Step 3: Process missing sim_metrics files
    for url, sim_metrics_file in missing_metrics.items():
        logger.info(f"Processing missing metrics for {url}")

        # Convert sim_metrics_file to Path object if it's not already
        sim_metrics_file = Path(sim_metrics_file)

        # Load the requirements and responsibilities files from the mapping
        reqs_file = Path(file_mappings_dict[url].reqs)
        resps_file = Path(file_mappings_dict[url].resps)

        # Step 3.1: Check if reqs and resps files exist and are valid JSON files
        if not reqs_file.exists() or not resps_file.exists():
            if not reqs_file.exists():
                logger.error(f"Skipping {url}: Missing requirements file {reqs_file}")
            if not resps_file.exists():
                logger.error(
                    f"Skipping {url}: Missing responsibilities file {resps_file}"
                )
            continue

        # Step 3.2: Generate the metrics file
        generate_metrics(reqs_file, resps_file, sim_metrics_file)
        logger.info(f"Generated metrics for {url} and saved to {sim_metrics_file}")

    logger.info("Finished processing all missing sim_metrics files.")
