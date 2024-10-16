import os
from pathlib import Path
import logging
from typing import Union
from tqdm import tqdm
from joblib import Parallel, delayed
from src.models.resume_and_job_description_models import (
    OptimizedText,
    ResponsibilityMatch,
    ResponsibilityMatches,
    ResponsibilityInput,
    RequirementsInput,
)
from evaluation_optimization.resume_editor import TextEditor
from evaluation_optimization.text_similarity_finder import AsymmetricTextSimilarity
from evaluation_optimization.metrics_calculator import categorize_scores
from evaluation_optimization.resumes_editing import modify_multi_resps_based_on_reqs
from utils.generic_utils import (
    read_from_json_file,
    save_to_json_file,
    verify_dir,
    verify_file,
)


# Set up logging
logger = logging.getLogger(__name__)


def check_mapping_keys(file_mapping_prev: dict, file_mapping_curr: dict) -> dict:
    """
    Check if the keys (URLs) in the previous and current mapping files are the same.

    Args:
        file_mapping_prev (dict): Dictionary loaded from the previous mapping file.
        file_mapping_curr (dict): Dictionary loaded from the current mapping file.

    Returns:
        dict: A dictionary containing the keys that are only in the previous or only
        in the current file.

    Raises:
        ValueError: If there are differences in the keys between the two mapping files.
    """
    prev_keys = set(file_mapping_prev.keys())
    curr_keys = set(file_mapping_curr.keys())

    # Find keys that are only in one of the mappings
    missing_in_prev = curr_keys - prev_keys
    missing_in_curr = prev_keys - curr_keys

    if missing_in_prev or missing_in_curr:
        error_message = (
            f"Key mismatch detected:\n"
            f"Missing in previous mapping: {missing_in_prev}\n"
            f"Missing in current mapping: {missing_in_curr}"
        )
        raise ValueError(error_message)

    return {
        "missing_in_prev": missing_in_prev,
        "missing_in_curr": missing_in_curr,
    }


def set_directory_paths(
    mapping_file_prev: Union[str, Path], mapping_file_curr: Union[str, Path]
) -> dict:
    """
    Run the pipeline to modify responsibilities based on the previous and current mapping files.
    The pipeline uses joblib to process jobs in parallel.

    Args:
        mapping_file_prev (Union[str, Path]): Path to the mapping file for the previous iteration.
        mapping_file_curr (Union[str, Path]): Path to the mapping file for the current iteration.
        model (str, optional): Model name to be used for modifications. Defaults to 'openai'.
        model_id (str, optional): Specific model version to be used. Defaults to 'gpt-3.5-turbo'.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1 (use all available).

    Returns:
        None: The function modifies responsibilities and saves the results to output files.
        If the output file already exists for a URL, the function will skip processing
        for that URL.
    """

    # Load the mapping files
    mapping_file_curr = Path(mapping_file_curr)
    mapping_file_prev = Path(mapping_file_prev)
    try:
        file_mapping_prev = read_from_json_file(mapping_file_prev)
        file_mapping_curr = read_from_json_file(mapping_file_curr)
        logger.info(
            f"Loaded mapping files from {mapping_file_prev} and {mapping_file_curr}"
        )
    except FileNotFoundError:
        logger.error(
            f"One of the mapping files not found: {mapping_file_prev} or {mapping_file_curr}"
        )
        return {}
    except Exception as e:
        logger.error(f"Error loading mapping files: {e}")
        return {}

    # Ensure both mapping files are valid dictionaries (not None)
    if not isinstance(file_mapping_prev, dict) or not isinstance(
        file_mapping_curr, dict
    ):
        logger.error(
            "Failed to load one or both mapping files. They are not dictionaries."
        )
        return {}

    # Check if the keys (URLs) are the same in both mappings
    try:
        check_mapping_keys(file_mapping_prev, file_mapping_curr)
    except ValueError as e:
        logger.error(f"Error during key validation: {e}")
        return {}

    # Set directory paths using both the previous and current mapping files
    paths_dict = {}
    for url, prev_paths in file_mapping_prev.items():
        if url not in file_mapping_curr:
            logger.warning(
                f"URL {url} not found in the current iteration's mapping file."
            )
            continue

        curr_paths = file_mapping_curr[url]
        paths_dict[url] = {
            "requirements_input": Path(prev_paths["reqs"]),
            "responsibilities_input": Path(prev_paths["pruned_resps"]),
            "responsibilities_output": Path(curr_paths["resps"]),
        }

        # Verify the file paths and directory paths
        if not verify_file(paths_dict[url]["requirements_input"]):
            logger.error(f"Missing requirements file for {url}. Skipping URL.")
            continue
        if not verify_file(paths_dict[url]["responsibilities_input"]):
            logger.error(
                f"Missing pruned responsibilities file for {url}. Skipping URL."
            )
            continue

    logger.info(f"Directory path dictionary:\n{paths_dict}")
    return paths_dict


def run_pipeline(
    mapping_file_prev: Union[str, Path],
    mapping_file_curr: Union[str, Path],
    model: str = "openai",
    model_id: str = "gpt-3.5-turbo",
    n_jobs: int = -1,
):
    """
    Run the pipeline to modify responsibilities based on the previous and current mapping files.
    The piple uses joblib to process jobs in parallel.

    Args:
        mapping_file_prev (Union[str, Path]): Path to the mapping file for the previous iteration.
        mapping_file_curr (Union[str, Path]): Path to the mapping file for the current iteration.
        model (str, optional): Model name to be used for modifications. Defaults to 'openai'.
        model_id (str, optional): Specific model version to be used. Defaults to 'gpt-3.5-turbo'.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1 (use all available).

    Returns:
        None
    """
    # Step 0: Ensure mapping file paths are Path objects
    mapping_file_prev = Path(mapping_file_prev)
    mapping_file_curr = Path(mapping_file_curr)

    # Step 1: Setup directory and file paths
    paths_dict = set_directory_paths(mapping_file_prev, mapping_file_curr)
    if not paths_dict:
        logger.error("Failed to set up directory paths.")
        return

    # Step 2: Process each job posting URL and modify responsibilities
    for url, paths in paths_dict.items():
        logger.info(f"Processing job posting from {url}")

        # *Early return if the output file already exists
        output_file = paths["responsibilities_output"]
        if output_file.exists():
            logger.info(f"Output file already exists for {url}, skipping processing.")
            continue  # Skip further processing for this URL

        try:
            # Extract file paths for requirements and pruned responsibilities
            reqs_file = paths["requirements_input"]
            resps_file = paths["responsibilities_input"]

            # Load responsibilities and requirements files
            if not reqs_file.exists() or not resps_file.exists():
                raise FileNotFoundError(f"Files not found for {url}")

            responsibilities = read_from_json_file(resps_file)
            requirements = read_from_json_file(reqs_file)

            if not responsibilities or not requirements:
                raise ValueError(f"Files are empty for {url}")

            # Use Pydantic for validation
            validated_responsibilities = ResponsibilityInput(
                responsibilities=responsibilities
            )
            validated_requirements = RequirementsInput(requirements=requirements)

        except (FileNotFoundError, ValueError) as error:
            logger.error(error)
            continue

        # Step 3: Modify responsibilities based on requirements
        with tqdm(
            total=len(responsibilities), desc=f"Modifying responsibilities for {url}"
        ) as pbar:
            modified_resps = modify_multi_resps_based_on_reqs(
                responsibilities=validated_responsibilities.responsibilities,
                requirements=validated_requirements.requirements,
                model=model,
                model_id=model_id,
                n_jobs=n_jobs,
            )  # the function returns a pyd object
            pbar.update(1)

        # Step 4: Save the modified responsibilities
        output_file = paths["resps_curr"]
        save_to_json_file(modified_resps.model_dump(), output_file)
        logger.info(f"Modified responsibilities for {url} saved to {output_file}")

    logger.info("Pipeline execution completed.")
