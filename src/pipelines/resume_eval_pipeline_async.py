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
import asyncio
import aiofiles
from typing import Any, Callable, Coroutine, Optional, Union

import pandas as pd
from pydantic import HttpUrl, ValidationError, TypeAdapter

# User defined
from preprocessing.resume_preprocessor import ResumeParser
from preprocessing.requirements_preprocessor import JobRequirementsParser
from models.resume_job_description_io_models import (
    JobFileMappings,
    Requirements,
    Responsibilities,
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
    save_df_to_csv_file_async,
    save_data_to_json_file_async,
)

# from config import job_descriptions_json_file
from evaluation_optimization.evaluation_optimization_utils import (
    get_files_wo_multivariate_indices,
)


# Set up logger
logger = logging.getLogger(__name__)


def add_multivariate_indices(df: pd.DataFrame) -> pd.DataFrame:
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
        # Log or print as desired, then re-raise to adhere to the return contract
        logger.error(f"ValueError: {ve}")
        raise
    except AttributeError as ae:
        logger.error(f"AttributeError: {ae}")
        raise
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


async def generate_metrics_from_flat_json_async(
    reqs_file: Union[Path, str],
    resps_file: Union[Path, str],
    sim_metrics_file: Union[Path, str],
    url: str,  # Job posting URL for traceability
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Asynchronously loads, validates, and computes similarity metrics
    between resume responsibilities and job requirements, saving the results
    as a CSV file.

    **Processing Steps:**
    1. **Load JSON files asynchronously** (`reqs_file`, `resps_file`).
    2. **Validate job requirements and responsibilities** using Pydantic models.
    3. **Ensure loaded JSON data is a dictionary** and contains the expected keys.
    4. **Compute similarity metrics** using
    `calculate_many_to_many_similarity_metrices()`.
    5. **Categorize similarity scores** using `categorize_scores_for_df()`.
    6. **Clean the DataFrame** (remove newline characters, trim whitespace).
    7. **Insert job posting URL** as a new column in the DataFrame.
    8. **Save the DataFrame asynchronously** as a CSV file using
    `save_df_to_csv_file_async()`.

    *Concurrency Control:
    *- Uses `asyncio.Semaphore` to limit concurrent executions, preventing
    * excessive parallel processing.
    - Ensures controlled execution, reducing memory spikes and preventing
    system overload.

    **Args:**
        - `reqs_file` (Union[Path, str]):
          Path to the flattened job requirements JSON file.
        - `resps_file` (Union[Path, str]):
          Path to the flattened resume responsibilities JSON file.
        - `sim_metrics_file` (Union[Path, str]):
          Path where the computed similarity metrics CSV will be saved.
        - `url` (str):
          The job posting URL for traceability; added as a column in the output CSV.
        - `semaphore` (asyncio.Semaphore):
          A semaphore to control the number of concurrent executions.

    Returns:
        - `None`: The function does not return anything but saves the processed
        similarity metrics CSV file.

    Raises:
        - ValidationError: If the job requirements or responsibilities JSON
        does not match the expected format.
        - FileNotFoundError: If any required input file is missing.
        - ValueError: If input data is not structured as a dictionary.
        - Exception: Catches and logs any unexpected errors encountered during execution.

    **Example Usage:**
    ```python
    import asyncio
    from pathlib import Path

    semaphore = asyncio.Semaphore(5)  # Limit concurrency to 5 tasks
    await generate_metrics_from_flat_json_async(
        reqs_file=Path("requirements.json"),
        resps_file=Path("responsibilities.json"),
        sim_metrics_file=Path("output.csv"),
        url="https://example.com/job123",
        semaphore=semaphore
    )
    ```
    """
    async with semaphore:
        logger.info(f"Generating metrics for: {sim_metrics_file}")

        # Convert file paths to Path objects
        reqs_file, resps_file, sim_metrics_file = map(
            Path, [reqs_file, resps_file, sim_metrics_file]
        )

        # Step 1: Ensure files exist
        for file in [reqs_file, resps_file]:
            if not file.exists():
                logger.error(f"File not found: {file}")
                return

        # Step 2: Load JSON files asynchronously
        try:
            async with aiofiles.open(reqs_file, "r") as f_req, aiofiles.open(
                resps_file, "r"
            ) as f_resp:
                reqs_data, resps_data = await asyncio.gather(
                    f_req.read(), f_resp.read()
                )

            reqs_data, resps_data = json.loads(reqs_data), json.loads(resps_data)

            # Step 3: Validate JSON structure using Pydantic models
            reqs_model = Requirements(**reqs_data)
            resps_model = Responsibilities(**resps_data)

            # Step 4: Extract validated data (responsibilities, requirements)
            reqs_flat = reqs_model.requirements
            resps_flat = resps_model.responsibilities

            logger.info(f"Validated requirements from {reqs_file}")
            logger.info(f"Validated responsibilities from {resps_file}")

        except ValidationError as ve:
            logger.error(f"Validation error in {reqs_file} or {resps_file}: {ve}")
            return
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading JSON files: {e}")
            return
        except Exception as e:
            logger.error(
                f"Unexpected error loading files {reqs_file} or {resps_file}: {e}"
            )
            return

        # Step 5: Compute similarity metrics in a separate thread
        try:
            similarity_df = await asyncio.to_thread(
                calculate_many_to_many_similarity_metrices,
                responsibilities=resps_flat,
                requirements=reqs_flat,
            )

            similarity_df = await asyncio.to_thread(
                categorize_scores_for_df, similarity_df
            )

            # Step 6: Apply final cleaning asynchronously
            df = await asyncio.to_thread(
                lambda df: df.applymap(lambda x: str(x).replace("\n", " ").strip()),
                similarity_df,
            )

            # Step 7: Add the job posting URL as the first column
            df.insert(0, "job_posting_url", url)

            # Step 8: Save the metrics CSV asynchronously
            await save_df_to_csv_file_async(df, sim_metrics_file)

            logger.info(f"Metrics saved to {sim_metrics_file} with URL column added.")

        except Exception as e:
            logger.error(f"Error during similarity computation or saving CSV: {e}")


# todo: need to fix and add url
async def generate_metrics_from_nested_json_async(
    reqs_file: Union[Path, str],
    resps_file: Union[Path, str],
    metrics_csv_file: Union[Path, str],
) -> None:
    """
    Generate similarity metrics between nested responsibilities and requirements
    and save to a CSV file asynchronously.

    Args:
        - reqs_file (Path or str): Path to the requirements JSON file.
        - resps_file (Path or str): Path to the nested responsibilities JSON file.
        - metrics_csv_file (Path or str): Path where the output CSV file should be saved.
        - url (str): The job posting URL to be included in the output DataFrame.

    Returns:
        None
    """
    logger.info(f"Generating metrics for: {metrics_csv_file}")

    # Step 0: Ensure inputs are Path objects.
    reqs_file = Path(reqs_file)
    resps_file = Path(resps_file)
    metrics_csv_file = Path(metrics_csv_file)

    # Step 1: Load and validate responsibilities and requirements using
    # Pydantic models asynchronously
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


async def run_metrics_processing_pipeline_async(
    mapping_file: Union[Path, str],
    generate_metrics: Callable,
    batch_size: int = 7,  # Adjust batch size as needed
) -> None:
    """
    Asynchronous pipeline to process and create missing similarity metrics files
    by reading from the job file mapping JSON.

    Args:
        - mapping_file (Union[Path, str]): Path to the JSON mapping file.
        - generate_metrics (Callable): Function to generate the metrics CSV file.
        - batch_size (int): Number of tasks to process concurrently.

    Returns:
        None
    """
    logger.info("Starting async metrics processing pipeline...")

    mapping_file = Path(mapping_file)  # Ensure file path is a Path object.

    # Step 1: Read the mapping file (synchronously)
    file_mapping_model = load_mappings_model_from_json(
        mapping_file
    )  # Returns pyd model

    if file_mapping_model is None:
        logger.error("Failed to load mapping file. Exiting pipeline.")
        return

    logger.debug(f"Loaded mapping data from {mapping_file}")

    # Step 2: Identify missing similarity metrics files
    missing_metrics = {
        str(url): job_paths.sim_metrics
        for url, job_paths in file_mapping_model.root.items()
        if not Path(job_paths.sim_metrics).exists()  # Directly check existence
    }

    if not missing_metrics:
        logger.info("All similarity metrics files exist. Exiting pipeline.")
        return

    logger.debug(
        f"Missing similarity metrics files: {len(missing_metrics)} items found."
    )

    # Step 3: Process missing sim_metrics files concurrently with batching

    semaphore = asyncio.Semaphore(batch_size)
    tasks = []

    for url, sim_metrics_file in missing_metrics.items():
        logger.info(f"Processing missing metrics for {url}")

        try:
            # Convert `url` to `HttpUrl` using TypeAdapter (Explicit Validation)
            job_url = TypeAdapter(HttpUrl).validate_python(url)
            job_paths = file_mapping_model.root[job_url]  # Use converted HttpUrl
        except ValidationError as e:
            logger.error(f"Invalid URL format: {url}. Skipping. Error: {e}")
            continue

        # Load the requirements and responsibilities files from the mapping
        reqs_file = Path(job_paths.reqs)
        resps_file = Path(job_paths.resps)

        # Step 3.1: Check if reqs and resps files exist
        if not reqs_file.exists() or not resps_file.exists():
            logger.error(f"Missing files for {url}. Skipping.")
            continue

        # Step 3.2: Create async task for each URL
        tasks.append(
            generate_metrics(
                reqs_file=reqs_file,
                resps_file=resps_file,
                sim_metrics_file=sim_metrics_file,
                url=url,
                semaphore=semaphore,
            )
        )

    # Step 4: Process tasks in batches
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        logger.info(
            f"Processing batch {i // batch_size + 1}/{(len(tasks) + batch_size - 1) // batch_size} "
            f"with {len(batch)} tasks..."
        )

        # Ensuring batch continues even if one task fails
        results = await asyncio.gather(*batch, return_exceptions=True)

        # Log errors in the batch
        for j, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i + j} failed with error: {result}")

    logger.info("Finished processing all missing similarity metrics files.")


async def run_multivariate_indices_processing_mini_pipeline_async(
    mapping_file: Union[str, Path],
    add_indices_func: Callable[[pd.DataFrame], pd.DataFrame] = add_multivariate_indices,
) -> None:
    """
    Asynchronously reads a mapping file and adds multivariate indices
    (composite and PCA scores) to any 'sim_metrics' CSV files that
    do not already have them.

    Args:
        - mapping_file (str | Path): Path to the JSON mapping file that includes
            paths to sim_metrics files to be processed.
        - add_indices_func (Callable[[pd.DataFrame], pd.DataFrame], optional):
            The function to add multivariate indices to the DataFrame.
            Defaults to add_multivariate_indices.

    Raises:
        ValueError: If the mapping file does not exist.
    """
    mapping_file = Path(mapping_file)  # Ensure it's a Path object
    if not mapping_file.exists():
        raise ValueError(f"The file '{mapping_file}' does not exist.")

    # Step 1: Load the mapping file into a Pydantic model
    # Don't really need to use async for loading json file
    file_mapping_model = load_mappings_model_from_json(mapping_file)
    if file_mapping_model is None:
        logger.error(f"Failed to load the mapping file: {mapping_file}")
        return

    # Gather the sim_metrics files from each URL entry
    sim_metrics_files = {
        str(url): Path(paths.sim_metrics)
        for url, paths in file_mapping_model.root.items()
    }

    # Check for non-existent sim_metrics files
    missing_files = [fp for fp in sim_metrics_files.values() if not fp.exists()]
    missing_file_count = len(missing_files)
    if missing_file_count > 0:
        logger.warning(
            f"Missing sim_metrics files (found in mapping but not on disk): {missing_files}"
        )

    # Filter out any that don't exist on disk so we don't try to read them
    existing_files = [fp for fp in sim_metrics_files.values() if fp.exists()]

    # Step 2: Find which CSV files actually need multivariate indices
    files_need_to_process = get_files_wo_multivariate_indices(
        data_sources=existing_files
    )
    if not files_need_to_process:
        logger.info("No files require adding multivariate indices. Exiting pipeline.")
        return

    # Step 3: For each file that needs indices, read & update it asynchronously
    for file_path in files_need_to_process:
        try:
            df = await read_from_csv_async(file_path)

            # Verify required columns
            required_columns = {
                "responsibility_key",
                "responsibility",
                "requirement_key",
                "requirement",
                "bert_score_precision",
                "soft_similarity",
                "word_movers_distance",
                "deberta_entailment_score",
            }
            missing_cols = required_columns - set(df.columns)
            if missing_cols:
                logger.error(
                    f"File '{file_path}' is missing required columns: {missing_cols}"
                )
                continue  # Skip this file

            # Row-level validation using SimilarityMetrics pydantic model
            validated_rows = []
            for idx, row in df.iterrows():
                try:
                    validated_row = SimilarityMetrics(**row.to_dict())
                    validated_rows.append(validated_row.model_dump())
                except ValidationError as ve:
                    logger.warning(
                        f"Validation error in row {idx} of '{file_path}': {ve}"
                    )
                    # You could skip or drop these rows, as you see fit
                    continue

            if not validated_rows:
                logger.warning(f"No valid data in file '{file_path}'. Skipping.")
                continue

            validated_df = pd.DataFrame(validated_rows)

            # Apply the function to add multivariate indices
            updated_df = add_indices_func(validated_df)
            if updated_df is None:
                logger.error(
                    f"'{add_indices_func.__name__}' returned None for file '{file_path}'. Skipping."
                )
                continue

            # Save the updated DataFrame asynchronously
            await save_df_to_csv_file_async(updated_df, file_path)
            logger.info(f"Successfully processed and saved '{file_path}'.")

        except FileNotFoundError:
            logger.error(f"File not found: '{file_path}'. Skipping.")
            continue
        except pd.errors.EmptyDataError:
            logger.error(f"File '{file_path}' is empty. Skipping.")
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing file '{file_path}': {e}")
            continue

    logger.info(
        f"Successfully added multivariate indices to {len(files_need_to_process)} file(s)."
    )

    # Final summary if any files in the mapping did not exist at all
    if missing_file_count > 0:
        logger.info(
            f"Pipeline completed, but {missing_file_count} sim_metrics file(s) in the mapping "
            "did not exist on disk."
        )


async def run_metrics_re_processing_pipeline_async(
    mapping_file: Path,
    generate_metrics: Callable[
        [Path, Path, Path], Coroutine[Any, Any, None]
    ] = generate_metrics_from_nested_json_async,
) -> None:
    """
    Re-run the pipeline to process and create missing sim_metrics files by reading from
    the mapping file asynchronously.

    Args:
        - mapping_file (str | Path): Path to the JSON mapping file.
        - generate_metrics (Callable[[Path, Path, Path], Coroutine[Any, Any, None]], optional):
            Asynchronous function to generate the metrics CSV file.
            Defaults to generate_matching_metrics_from_nested_json_async.

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

        # Step 3.2: validate the requirements and responsibilities files with pyd models
        try:
            reqs_data = read_from_json_file(reqs_file)

            # Validate using the Requirements model
            validated_requirements = Requirements.model_validate(reqs_data)
            logger.info(f"Loaded and validated requirements from {reqs_file}")
        except ValidationError as e:
            logger.error(f"Validation error for requirements: {e}")
            continue

        # Attempt to load and validate the responsibilities JSON with ResponsibilityMatches model
        try:
            resps_data = read_from_json_file(resps_file)

            # If the file lacks a top-level "responsibilities" key, wrap it
            if "responsibilities" not in resps_data:
                resps_data = {"responsibilities": resps_data}

            # If the file lacks a top-level "responsibilities" key, wrap it
            if "responsibilities" not in resps_data:
                resps_data = {"responsibilities": resps_data}

            validated_responsibilities = ResponsibilityMatches.model_validate(
                resps_data
            )
            logger.info(f"Loaded and validated responsibilities from {resps_file}")
        except ValidationError as e:
            logger.error(f"Validation error when parsing JSON files: {e}")
            continue

        # Step 3.3: Generate the metrics file asynchronously
        tasks.append(generate_metrics(reqs_file, resps_file, sim_metrics_file))
        logger.info(
            f"Queued metrics generation for {url} to be saved to {sim_metrics_file}"
        )

    # Step 4: run all the tasks concurrently
    if tasks:
        await asyncio.gather(*tasks)
        logger.info("Finished processing all missing sim_metrics files.")
    else:
        logger.info("No valid files were found to process.")
