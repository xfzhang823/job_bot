"""
Filename: resume_eval_pipeline_async.py

Async version of pipelines to create matching metrics between responsibilities 
and requirements.
"""

# Import dependencies
import os
from pathlib import Path
import pandas as pd
import json
import logging
import logging_config
import asyncio
import aiofiles
from typing import Any, Callable, Coroutine, Optional, Union
import pandas as pd
from pydantic import ValidationError

from preprocessing.resume_preprocessor import ResumeParser
from preprocessing.requirements_preprocessor import JobRequirementsParser
from models.resume_job_description_io_models import (
    JobFileMappings,
    Requirements,
    ResponsibilityMatches,
    SimilarityMetrics,
)

from evaluation_optimization.create_mapping_file import load_mappings_model_from_json
from evaluation_optimization.metrics_calculator import (
    calculate_many_to_many_similarity_metrices,
    categorize_scores_for_df,
    SimilarityScoreCalculator,
    calculate_text_similarity_metrics,
)
from evaluation_optimization.multivariate_indexer import MultivariateIndexer
from evaluation_optimization.text_similarity_finder import TextSimilarity


from utils.generic_utils import (
    read_from_json_file,
    save_to_json_file,
)
from utils.generic_utils_async import (
    read_from_json_async,
    read_from_csv_async,
    save_to_csv_async,
    save_to_json_file_async,
)

# from config import job_descriptions_json_file
from evaluation_optimization.evaluation_optimization_utils import (
    get_files_wo_multivariate_indices,
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


async def generate_matching_metrics_async(reqs_file, resps_file, sim_metrics_file):
    """
    Generate and save similarity metrics asynchronously.
    """
    logger.info(f"Generating metrics for: {sim_metrics_file}")

    # Load responsibilities and requirements asynchronously
    async with aiofiles.open(reqs_file, mode="r") as f_req:
        reqs_flat = await f_req.read()

    async with aiofiles.open(resps_file, mode="r") as f_resp:
        resps_flat = await f_resp.read()

    # Calculate metrics
    similarity_df = calculate_many_to_many_similarity_metrices(resps_flat, reqs_flat)
    similarity_df = categorize_scores_for_df(similarity_df)
    df = similarity_df.applymap(lambda x: str(x).replace("\n", " ").strip())

    # Save the metrics CSV asynchronously
    async with aiofiles.open(sim_metrics_file, mode="w") as f_csv:
        await df.to_csv(f_csv, index=False)

    logger.info(f"Metrics saved to {sim_metrics_file}")


async def generate_matching_metrics_from_nested_json_async(
    reqs_file: Union[Path, str],
    resps_file: Union[Path, str],
    metrics_csv_file: Union[Path, str],
) -> None:
    """
    Generate similarity metrics between nested responsibilities and requirements
    and save to a CSV file asynchronously.

    Args:
        reqs_file (Path or str): Path to the requirements JSON file.
        resps_file (Path or str): Path to the nested responsibilities JSON file.
        metrics_csv_file (Path or str): Path where the output CSV file should be saved.

    Returns:
        None
    """
    # Step 0: Ensure inputs are Path objects.
    reqs_file = Path(reqs_file)
    resps_file = Path(resps_file)
    metrics_csv_file = Path(metrics_csv_file)

    # Step 1: Load and validate responsibilities and requirements using Pydantic models asynchronously
    try:
        resps_data = await asyncio.to_thread(read_from_json_file, resps_file)
        validated_resps_data = ResponsibilityMatches.model_validate(resps_data)

        reqs_data = await asyncio.to_thread(read_from_json_file, reqs_file)
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

            # Step 3: Calculate similarity metrics asynchronously
            similarity_metrics = await asyncio.to_thread(
                calculate_text_similarity_metrics, responsibility_text, requirement_text
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
        final_df = await asyncio.to_thread(pd.DataFrame, validated_rows)
        final_df = await asyncio.to_thread(categorize_scores_for_df, final_df)

        # Step 6: Save the validated metrics to a CSV file asynchronously
        await asyncio.to_thread(final_df.to_csv, metrics_csv_file, index=False)
        logger.info(f"Similarity metrics saved successfully to {metrics_csv_file}")
    else:
        logger.error("No valid similarity metrics data to save.")
        return

    # Display the top rows of the DataFrame for verification
    print(final_df.head(5))


# def unpack_and_combine_json(nested_json, requirements_json):
#     """
#     Unpacks the nested responsibilities JSON and combines it with matching requirement texts
#     from the requirements JSON. Outputs a list of dictionaries with responsibility and requirement texts.

#     Args:
#         nested_json (dict): JSON-like dictionary containing responsibility text structured in a nested format.
#         requirements_json (dict): JSON-like dictionary containing requirement texts keyed by requirement IDs.

#     Returns:
#         list: A list of dictionaries containing responsibility_keys, requirement_keys,
#               responsibility texts, and matched requirement texts.

#     Error Handling:
#         - If a requirement_key is not found in the requirements JSON, it will skip that entry.
#         - If a required field (e.g., 'optimized_text') is missing, it will skip that entry.
#         - Logs warnings for missing fields and unmatched keys for better traceability.
#     """
#     results = []

#     for resp_key, values in nested_json.items():  # Unpack the 1st level
#         if not isinstance(values, dict):
#             logger.info(
#                 f"Warning: Unexpected data structure under '{resp_key}'. Skipping entry."
#             )
#             continue

#         for req_key, sub_value in values.items():  # Unpack the 2nd level
#             if not isinstance(sub_value, dict):
#                 logger.info(
#                     f"Warning: Unexpected data structure under '{req_key}'. Skipping entry."
#                 )
#                 continue

#             # Extract the optimized_text
#             if "optimized_text" in sub_value:
#                 optimized_text = sub_value["optimized_text"]
#             else:
#                 logger.info(
#                     f"Warning: Missing 'optimized_text' for '{req_key}'. Skipping entry."
#                 )
#                 continue

#             # Perform the lookup in the requirements JSON
#             requirement_text = requirements_json.get(req_key)
#             if requirement_text is None:
#                 logger.info(
#                     f"Warning: requirement_key '{req_key}' not found in requirements JSON. Skipping entry."
#                 )
#                 continue

#             # Append results to a list for further processing
#             results.append(
#                 {
#                     "responsibility_key": resp_key,
#                     "requirement_key": req_key,
#                     "responsibility_text": optimized_text,
#                     "requirement_text": requirement_text,
#                 }
#             )

#     return results


async def metrics_processing_pipeline_async(
    mapping_file: str, generate_metrics: Callable = generate_matching_metrics_async
):
    """
    Asynchronous version of the pipeline to process and create missing sim_metrics files by reading from the mapping file.

    Args:
        mapping_file (str or Path): Path to the JSON mapping file.
        generate_metrics_func (callable): Function to generate the metrics CSV file.

    Returns:
        None
    """

    logger.info(
        "Start running responsibility/requirement alignment scoring pipeline (async)."
    )

    # Step 1: Read the mapping file asynchronously
    try:
        async with aiofiles.open(mapping_file, mode="r") as f:
            content = await f.read()
            file_mapping = json.loads(content)
        logger.info(f"Loaded mapping file from {mapping_file}")
    except FileNotFoundError:
        logger.error(f"Mapping file not found: {mapping_file}")
        return
    except Exception as e:
        logger.error(f"Error loading mapping file: {e}")
        return

    # Step 2: Check for missing sim_metrics files
    missing_metrics = {
        url: mapping["sim_metrics"]
        for url, mapping in file_mapping.items()
        if not Path(mapping["sim_metrics"]).exists()
    }

    if not missing_metrics:
        logger.info("All sim_metrics files already exist. Exiting pipeline.")
        return  # Early exit if all files exist

    # Step 3: Process missing sim_metrics files concurrently using asyncio.gather
    tasks = []
    for url, sim_metrics_file in missing_metrics.items():
        logger.info(f"Processing missing metrics for {url}")

        # Load the requirements and responsibilities files from the mapping
        reqs_file = Path(file_mapping[url]["reqs_flat"])
        resps_file = Path(file_mapping[url]["resps_flat"])

        # Step 3.1: Check if reqs and resps files exist
        if not reqs_file.exists() or not resps_file.exists():
            logger.error(
                f"Missing requirements or responsibilities files for {url}. Skipping."
            )
            continue

        # Step 3.2: Create async task for each URL
        tasks.append(generate_metrics(reqs_file, resps_file, sim_metrics_file))

    # Step 4: Run tasks concurrently
    await asyncio.gather(*tasks)

    logger.info("Finished processing all missing sim_metrics files.")


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
        files_need_to_process = get_files_wo_multivariate_indices(data_directory)
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


async def metrics_re_processing_pipeline_async(
    mapping_file: Path,
    generate_metrics: Callable[
        [Path, Path, Path], Coroutine[Any, Any, None]
    ] = generate_matching_metrics_from_nested_json_async,
) -> None:
    """
    Re-run the pipeline to process and create missing sim_metrics files by reading from the mapping file asynchronously.

    Args:
        mapping_file (str | Path): Path to the JSON mapping file.
        generate_metrics (Callable[[Path, Path, Path], Coroutine[Any, Any, None]], optional):
            Asynchronous function to generate the metrics CSV file. Defaults to
            generate_matching_metrics_from_nested_json_async.

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

    # Step 3: Process missing sim_metrics files asynchronously
    tasks = []
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

        # Step 3.2: Generate the metrics file asynchronously
        tasks.append(generate_metrics(reqs_file, resps_file, sim_metrics_file))
        logger.info(
            f"Queued metrics generation for {url} to be saved to {sim_metrics_file}"
        )

    # Run all the tasks concurrently
    if tasks:
        await asyncio.gather(*tasks)
        logger.info("Finished processing all missing sim_metrics files.")
    else:
        logger.info("No valid files were found to process.")
