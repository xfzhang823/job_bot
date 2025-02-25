"""
Filename: resume_eval_pipeline_async.py

Async version of pipelines to create matching metrics between responsibilities 
and requirements.
"""

# Import dependencies
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional, Union
import json
import logging
import asyncio
import aiofiles
import numpy as np
import pandas as pd
from pydantic import HttpUrl, ValidationError, TypeAdapter
from IPython.display import display

# User defined
from preprocessing.resume_preprocessor import ResumeParser
from preprocessing.requirements_preprocessor import JobRequirementsParser
from models.resume_job_description_io_models import (
    JobFileMappings,
    Requirements,
    Responsibilities,
    NestedResponsibilities,
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
from evaluation_optimization.evaluation_optimization_utils import (
    add_multivariate_indices,
)
from utils.generic_utils import (
    read_from_json_file,
    save_to_json_file,
)
from utils.generic_utils_async import (
    read_json_file_async,
    read_csv_file_async,
    read_and_validate_json_async,
    save_df_to_csv_file_async,
    save_data_to_json_file_async,
)

# from config import job_descriptions_json_file
from evaluation_optimization.evaluation_optimization_utils import (
    get_files_wo_multivariate_indices,
)


# Set up logger
logger = logging.getLogger(__name__)


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

    Processing Steps:
    1. Load JSON files asynchronously (`reqs_file`, `resps_file`).
    2. Validate job requirements and responsibilities using Pydantic models.
    3. Ensure loaded JSON data is a dictionary** and contains the expected keys.
    4. Compute similarity metrics using
    `calculate_many_to_many_similarity_metrices()`.
    5. Categorize similarity scores** using `categorize_scores_for_df()`.
    6. Clean the DataFrame (remove newline characters, trim whitespace).
    7. Insert job posting URL as a new column in the DataFrame.
    8. Save the DataFrame asynchronously** as a CSV file using
    `save_df_to_csv_file_async()`.

    *Concurrency Control:
    *- Uses `asyncio.Semaphore` to limit concurrent executions, preventing
    * excessive parallel processing.
    - Ensures controlled execution, reducing memory spikes and preventing
    system overload.

    Args:
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

            # Raw str -> Dict (pydantic model can't read raw str.)
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

            # Step 7: Ensure URL column is first and converted to string
            df["job_posting_url"] = str(url)  # Assign first
            df = df[
                ["job_posting_url"]
                + [col for col in df.columns if col != "job_posting_url"]
            ]

            # Step 8: Validate before saving
            if df.empty:
                logger.warning(
                    f"Skipping CSV save: DataFrame is empty for {sim_metrics_file}"
                )
                return

            # Count Empty Rows
            empty_rows = df[df.isnull().all(axis=1)]
            logger.debug(f"Empty Rows Before Saving: {len(empty_rows)}")

            # Step 9: Save the metrics CSV asynchronously
            await save_df_to_csv_file_async(df, sim_metrics_file)

            # # Simple save
            # df.to_csv(sim_metrics_file, index=False, encoding="utf-8")

            logger.info(f"Metrics saved to {sim_metrics_file} with URL column added.")

        except Exception as e:
            logger.error(f"Error during similarity computation or saving CSV: {e}")


async def generate_metrics_from_nested_json_async(
    reqs_file: Union[Path, str],
    resps_file: Union[Path, str],
    sim_metrics_file: Union[Path, str],
    url: str,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    * The method is almost identical to the generate_metrics_from_flat_json_async
    * but reading nested responsibilities (edited resps) w/t many-to-many mapping.

    Generate similarity metrics between nested responsibilities and requirements
    and save to a CSV file asynchronously.

    Args:
        - reqs_file (Path or str): Path to the requirements JSON file.
        - resps_file (Path or str): Path to the nested responsibilities JSON file.
        - metrics_csv_file (Path or str): Path where the output CSV file should be saved.
        - url (str): The job posting URL to be included in the output DataFrame.
        - semaphore (asyncio.Semaphore): Controls the number of concurrent executions.

    Returns:
        None

    Key Features:
    ✅ Uses `aiofiles.open()` for async file reading.
    ✅ Uses `asyncio.to_thread()` for CPU-bound operations.
    ✅ Handles many-to-many mapping of responsibilities to requirements.
    ✅ Efficient error handling and structured logging.
    """
    logger.info(f"Generating metrics for: {sim_metrics_file}")

    async with semaphore:

        # Step 1: Ensure inputs are Path objects.
        reqs_file, resps_file, sim_metrics_file = (
            Path(reqs_file),
            Path(resps_file),
            Path(sim_metrics_file),
        )

        # Step 2: Load and validate responsibilities and requirements asynchronously
        # (w/t pyd models)
        try:
            # Read responsibilities & requirements from JSON files
            async with aiofiles.open(reqs_file, "r") as f_req, aiofiles.open(
                resps_file, "r"
            ) as f_resp:
                reqs_data, resps_data = await asyncio.gather(
                    f_req.read(), f_resp.read()
                )

            # Convert JSON strings to dictionaries (pyd models can't read raw str)
            reqs_data, resps_data = json.loads(reqs_data), json.loads(resps_data)

            # Validate with correct models
            validated_reqs_data = Requirements(**reqs_data)
            validated_resps_data = NestedResponsibilities(
                **resps_data
            )  # Uses NestedResponsibilities (responsibilities are edited & in nested format)

            if (
                not validated_resps_data.responsibilities
                or not validated_reqs_data.requirements
            ):
                logger.error("Responsibilities or requirements are empty. Exiting.")
                return

        except ValidationError as ve:
            logger.error(f"Validation error in JSON files: {ve}")
            return
        except FileNotFoundError as e:
            logger.error(f"File not found: {e.filename}")
            return
        except Exception as e:
            logger.error(f"Error loading files: {e}")
            return

        validated_rows = []

        # Step 3: Process similarity metrics by
        # iterate through the responsibilities and requirements
        for (
            responsibility_key,
            responsibility_matches,
        ) in validated_resps_data.responsibilities.items():
            for (
                requirement_key,
                optimized_text_obj,
            ) in responsibility_matches.optimized_by_requirements.items():

                # Fetch responsibility text
                responsibility_text = optimized_text_obj.optimized_text

                # Fetch matching requirement text
                requirement_text = validated_reqs_data.requirements.get(
                    requirement_key, None
                )
                if not requirement_text:
                    logger.warning(
                        f"No matching requirement found for requirement_key: {requirement_key}"
                    )
                    continue

                # Step 4: Compute similarity metrics asynchronously
                similarity_metrics = await asyncio.to_thread(
                    calculate_text_similarity_metrics,
                    responsibility_text,
                    requirement_text,
                )

                # Step 5: Validate Using SimilarityMetrics Model
                # * This is needed b/c it matches multiple responsibilities to multiple requirements
                # * (many-to-many mapping).
                try:
                    similarity_metrics_model = SimilarityMetrics(
                        job_posting_url=url,
                        responsibility_key=responsibility_key,
                        responsibility=responsibility_text,
                        requirement_key=requirement_key,
                        requirement=requirement_text,
                        **similarity_metrics,  # ✅ Automatically adds all fields from the dict
                        # bert_score_precision=similarity_metrics[
                        #     "bert_score_precision"
                        # ],  # Explicit mapping
                        # soft_similarity=similarity_metrics[
                        #     "soft_similarity"
                        # ],  # Explicit mapping
                        # word_movers_distance=similarity_metrics[
                        #     "word_movers_distance"
                        # ],  # Explicit mapping
                        # deberta_entailment_score=similarity_metrics[
                        #     "deberta_entailment_score"
                        # ],  # Explicit mapping
                        # # Optional fields (if present in similarity metrics)
                        # bert_score_precision_cat=similarity_metrics.get(
                        #     "bert_score_precision_cat"
                        # ),
                        # soft_similarity_cat=similarity_metrics.get(
                        #     "soft_similarity_cat"
                        # ),
                        # word_movers_distance_cat=similarity_metrics.get(
                        #     "word_movers_distance_cat"
                        # ),
                        # deberta_entailment_score_cat=similarity_metrics.get(
                        #     "deberta_entailment_score_cat"
                        # ),
                        # scaled_bert_score_precision=similarity_metrics.get(
                        #     "scaled_bert_score_precision"
                        # ),
                        # scaled_deberta_entailment_score=similarity_metrics.get(
                        #     "scaled_deberta_entailment_score"
                        # ),
                        # scaled_soft_similarity=similarity_metrics.get(
                        #     "scaled_soft_similarity"
                        # ),
                        # scaled_word_movers_distance=similarity_metrics.get(
                        #     "scaled_word_movers_distance"
                        # ),
                        # composite_score=similarity_metrics.get("composite_score"),
                        # pca_score=similarity_metrics.get("pca_score"),
                    )

                    validated_rows.append(similarity_metrics_model.model_dump())

                except ValidationError as ve:
                    logger.error(
                        f"Validation error for responsibility {responsibility_key}: {ve}"
                    )
                    continue

        # Step 6: Convert Validated Data to DataFrame & Save
        if validated_rows:
            df = await asyncio.to_thread(pd.DataFrame, validated_rows)
            df = await asyncio.to_thread(categorize_scores_for_df, df)

            # Step 4: Save the validated metrics to a CSV file asynchronously
            logger.info(
                f"DataFrame columns before saving: {df.columns}"
            )  # todo: debug; delete later
            logger.info(
                f"DataFrame first few rows:\n{df.head()}"
            )  # todo: debug; delete later

            await save_df_to_csv_file_async(df=df, filepath=sim_metrics_file)
            # await asyncio.to_thread(final_df.to_csv, metrics_csv_file, index=False)
            logger.info(f"Similarity metrics saved successfully to {sim_metrics_file}")
        else:
            logger.error("No valid similarity metrics data to save.")
            return

        # Display the top rows of the DataFrame for verification
        display(df.head(5))


async def run_metrics_processing_pipeline_async(
    mapping_file: Path | str,
    generate_metrics: Callable = generate_metrics_from_flat_json_async,
    batch_size: int = 10,  # Limit group size for batching (# of tasks/batch)
    max_concurrent: int = 6,  # Limit number of concurrent tasks
) -> None:
    """
    * Asynchronous pipeline to process and create missing similarity metrics files
    * by reading from the job file mapping JSON.

    Args:
        - mapping_file (str | Path): Path to the JSON mapping file.
        - generate_metrics (Callable): Function to generate the metrics CSV file.
        - batch_size (int): Number of tasks to process concurrently.
        - max_concurrent (int): Number of concurrent tasks.

    Returns:
        None
    """
    strict_mode = False  # Set to False if you want to skip bad files

    logger.info("Starting async metrics processing pipeline...")

    # Ensure mapping_file is a Path object
    if not isinstance(mapping_file, Path):
        mapping_file = Path(mapping_file)

    if not mapping_file.exists():
        raise ValueError(f"The file '{mapping_file}' does not exist.")

    logger.info(f"Loading mapping file: {mapping_file}")

    # Step 1: Read the mapping file (synchronously)
    file_mapping_model = load_mappings_model_from_json(
        mapping_file
    )  # Returns Pydantic model
    if file_mapping_model is None:
        logger.error("Failed to load mapping file. Exiting pipeline.")
        return

    logger.debug(f"Loaded mapping data from {mapping_file}")

    # Step 2: Identify missing similarity metrics files
    missing_metrics = {
        str(url): Path(job_paths.sim_metrics)  # Convert to Path directly
        for url, job_paths in file_mapping_model.root.items()
        if not Path(job_paths.sim_metrics).exists()
    }

    if not missing_metrics:
        logger.info("All similarity metrics files exist. Exiting pipeline.")
        return

    logger.info(f"Found {len(missing_metrics)} missing similarity metrics files.")

    # Step 3: Process missing sim_metrics files concurrently with batching
    # Step 3.1: Setup placeholders
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []

    # Step 3.2: Iterate through missing sim_metrics files
    for url, sim_metrics_file in missing_metrics.items():
        logger.info(f"Processing missing metrics for {url}")

        try:
            # Convert `url` to `HttpUrl` using TypeAdapter (Explicit Validation)
            job_url = TypeAdapter(HttpUrl).validate_python(url)
            job_paths = file_mapping_model.root[job_url]
        except ValidationError as e:
            logger.error(f"Invalid URL format: {url}. Skipping. Error: {e}")
            continue

        # Load & validate the requirements and responsibilities files from mapping
        # w/t read_and_validate_json function
        reqs_file = Path(job_paths.reqs)
        resps_file = Path(job_paths.resps)

        validated_requirements = await read_and_validate_json_async(
            reqs_file, Requirements
        )
        validated_responsibilities = await read_and_validate_json_async(
            resps_file, Responsibilities
        )

        if not validated_requirements or not validated_responsibilities:
            logger.error(f"Validation failed for {reqs_file} or {resps_file}.")

            if strict_mode:  # If True, then stop the process...
                raise ValueError(
                    f"Stopping execution due to invalid JSON in {reqs_file} or {resps_file}."
                )
            else:
                logger.warning("Skipping and continuing processing.")
                continue

        # Queue tasks after validation succeeds
        tasks.append(
            await generate_metrics(
                reqs_file=reqs_file,
                resps_file=resps_file,
                sim_metrics_file=sim_metrics_file,
                url=url,
                semaphore=semaphore,
            )
        )
        logger.info(f"📝 Queued metrics generation for {url} -> {sim_metrics_file}")

    # Step 4: Process tasks in batches
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]

        logger.info(
            f"Starting batch {i // batch_size + 1} of {len(tasks) // batch_size + 1} "
            f"({len(batch)} tasks in this batch)..."
        )

        # Run the batch asynchronously and handle failures
        results = await asyncio.gather(*batch, return_exceptions=True)

        # Log errors in the batch
        for j, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i + j} failed with error: {result}")

    # Log completion with accurate task count
    logger.info(f"Successfully processed {len(tasks)} similarity metrics files.")


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

    logger.info(f"Loading mapping file: {mapping_file}")

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
            logger.info(f"Processing file: {file_path}")

            df = await read_csv_file_async(file_path)

            # todo: debugging, delete later
            logger.debug(
                f"DataFrame from {file_path}: shape={df.shape}, columns={df.columns.tolist()}"
            )

            # Check for empty rows
            empty_rows = df[df.isnull().all(axis=1)]
            empty_row_count = len(empty_rows)

            if empty_row_count > 0:
                logger.warning(
                    f"File '{file_path}' contains {empty_row_count} completely empty row(s)."
                )
            # todo: delete later

            # Verify required columns
            required_columns = {
                "job_posting_url",
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

            # todo: debugging, check for url col and empty rows
            logger.debug(
                f"DataFrame from {file_path}: shape={df.shape}, columns={df.columns.tolist()}"
            )
            # Check for empty rows
            empty_rows = df[df.isnull().all(axis=1)]
            empty_row_count = len(empty_rows)

            if empty_row_count > 0:
                logger.warning(
                    f"File '{file_path}' contains {empty_row_count} completely empty row(s)."
                )

            logger.debug(
                f"Original row count: {df.shape[0]}, Validated row count: {len(validated_rows)}"
            )
            # todo: delete later

            # Apply the function to add multivariate indices
            logger.debug(
                f"Before adding indices: {validated_df.shape}"
            )  # todo: debugging; delete later
            updated_df = add_indices_func(validated_df)
            logger.debug(f"updated df: {updated_df}")  # todo: debug; delete later
            if updated_df is None:
                logger.error(
                    f"'{add_indices_func.__name__}' returned None for file '{file_path}'. Skipping."
                )
                continue
            logger.debug(
                f"After adding indices: {updated_df.shape}"
            )  # todo: debugging; delete later
            logger.info(
                f"Columns after adding indices: {df.columns}"
            )  # todo: debugging; delete later
            logger.info(
                f"First few rows:\n{df.head()}"
            )  # todo: debugging; delete later

            # Remove fully empty rows (where all columns are NaN) and replace empty strings with NaN
            updated_df = updated_df.replace(r"^\s*$", np.nan, regex=True).dropna(
                how="all"
            )

            # Log before saving
            logger.debug(
                f"Final DataFrame shape before saving: {updated_df.shape}"
            )  # TODO: Remove later

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
    mapping_file: Union[Path, str],
    generate_metrics: Callable = generate_metrics_from_nested_json_async,
    batch_size: int = 10,
    max_concurrent: int = 6,
) -> None:
    """
    * Re-run the pipeline asynchronously to process and create missing sim_metrics files
    * by reading from the mapping file.

    Args:
        - mapping_file (str | Path): Path to the JSON mapping file.
        - generate_metrics (Callable[[Path, Path, Path], Coroutine[Any, Any, None]],
        optional):
            Asynchronous function to generate the metrics CSV file.
            Defaults to generate_metrics_from_nested_json_async.
        - batch_size (int): Number of tasks to process concurrently.
        - max_concurrent (int): Number of concurrent tasks allowed.

    Returns:
        None
    """
    strict_mode = False  # If True, fail pipeline on validation errors

    logger.info("🚀 Starting re-processing pipeline for missing similarity metrics...")

    # Step 1: Ensure mapping_file is a Path object (if it's a string, convert it)
    if not isinstance(mapping_file, Path):
        mapping_file = Path(mapping_file)

    if not mapping_file.exists():
        raise ValueError(f"❌ Mapping file '{mapping_file}' does not exist.")

    logger.info(f"📂 Loading mapping file: {mapping_file}")

    # Step 2: Load the mapping file into a validated Pydantic model
    # This ensures the file structure and expected fields are correct before proceeding.
    file_mappings_model: JobFileMappings = load_mappings_model_from_json(mapping_file)
    if (
        file_mappings_model is None
    ):  # Error handling: Exit if mapping file fails validation.
        logger.error("Failed to load and validate the mapping file. Exiting pipeline.")
        return

    logger.debug(f"Loaded mapping data from {mapping_file}")

    # Step 3: Identify missing sim_metrics files
    # (i.e., job posting URLs where the similarity metrics file does not exist)
    missing_metrics = {
        str(url): Path(
            job_paths.sim_metrics
        )  # Convert sim_metrics path directly to Path object
        for url, job_paths in file_mappings_model.root.items()
        if not Path(
            job_paths.sim_metrics
        ).exists()  # Check if the sim_metrics file exists
    }

    # If no missing metrics, exit early to save computational resources.
    if not missing_metrics:
        logger.info("All sim_metrics files already exist. Exiting pipeline.")
        return

    # Log the total number of missing files to provide an overview of the workload.
    logger.info(f"🔍 Found {len(missing_metrics)} missing similarity metrics files.")

    # Step 4: Process missing sim_metrics files asynchronously

    # Step 4.1: Set up concurrency control
    # ✅ `semaphore` ensures that at most `max_concurrent` tasks run simultaneously.
    semaphore = asyncio.Semaphore(max_concurrent)

    # ✅ List of tasks to be executed asynchronously.
    tasks = []

    # Step 4.2: Iterate through missing sim_metrics files
    for url, sim_metrics_file in missing_metrics.items():
        logger.info(f"Processing missing metrics for {url}")

        try:
            # Convert `url` string to a validated `HttpUrl` object.
            job_url = TypeAdapter(HttpUrl).validate_python(url)
            # Fetch the corresponding file paths from the mapping.
            job_paths = file_mappings_model.root[job_url]
        except ValidationError as e:
            logger.error(f"Invalid URL format: {url}. Skipping. Error: {e}")
            continue  # Skip this iteration if URL validation fails.

        # Step 4.3: Validate the requirements JSON file before processing
        # ✅ Load the file paths for requirements and responsibilities.
        reqs_file = Path(job_paths.reqs)  # Convert to Path object
        resps_file = Path(job_paths.resps)  # Convert to Path object

        # ✅ Validate JSON files asynchronously
        validated_requirements = await read_and_validate_json_async(
            reqs_file, Requirements
        )
        validated_responsibilities = await read_and_validate_json_async(
            resps_file, NestedResponsibilities, "responsibilities"
        )

        if not validated_requirements or not validated_responsibilities:
            logger.error(f"❌ Validation failed for: {reqs_file} or {resps_file}.")

            if strict_mode:
                raise ValueError(
                    f"❌ Stopping execution due to invalid JSON in {reqs_file} or {resps_file}."
                )
            else:
                logger.warning("⚠️ Skipping and continuing processing.")
                continue

        # Step 4.5: Queue task for each missing sim_metrics file - runs asynchronously
        task = asyncio.create_task(
            generate_metrics(
                reqs_file=reqs_file,
                resps_file=resps_file,
                sim_metrics_file=sim_metrics_file,
                url=url,
                semaphore=semaphore,
            )
        )
        tasks.append(task)
        logger.info(f"📝 Queued metrics generation for {url} -> {sim_metrics_file}")

    # Step 5: Process tasks in batches
    # ✅ Prevents memory overload by processing `batch_size` tasks at a time.
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]  # Select a batch of tasks

        # Log batch progress (which batch is being processed).
        logger.info(
            f"🚀 Processing batch {i // batch_size + 1}/{(len(tasks) + batch_size - 1) // batch_size} "
            f"with {len(batch)} tasks..."
        )

        # Run the batch asynchronously and handle failures gracefully.
        results = await asyncio.gather(*batch, return_exceptions=True)

        # Log errors encountered in the batch.
        for j, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i + j} failed with error: {result}")

    # Final log message to indicate completion of the processing pipeline.
    logger.info(f"Successfully processed {len(tasks)} similarity metrics files.")
