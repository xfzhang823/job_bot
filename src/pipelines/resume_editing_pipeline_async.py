import os
from pathlib import Path
import logging
from typing import Union
from tqdm import tqdm
import asyncio

# from joblib import Parallel, delayed
from models.resume_job_description_io_models import (
    OptimizedText,
    ResponsibilityMatch,
    ResponsibilityMatches,
    Responsibilites,
    Requirements,
)
from evaluation_optimization.resume_editor import TextEditor
from evaluation_optimization.text_similarity_finder import AsymmetricTextSimilarity
from evaluation_optimization.metrics_calculator import categorize_scores

# Import from non parallell version for now!
# from evaluation_optimization.resumes_editing import modify_multi_resps_based_on_reqs
from evaluation_optimization.resumes_editing_async import (
    modify_multi_resps_based_on_reqs_async,
)

from evaluation_optimization.create_mapping_file import load_mappings_model_from_json
from utils.generic_utils import (
    read_from_json_file,
    verify_dir,
    verify_file,
)
from utils.generic_utils_async import save_to_json_file_async


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
    Set the directory paths for processing responsibilities and requirements based on the
    previous and current mapping files, using Pydantic models for validation.

    This function loads the previous and current iteration mapping files as Pydantic models,
    validates the mapping keys (URLs), and constructs a dictionary of paths for input/output
    responsibilities and requirements. It checks the validity of the file paths and logs
    warnings for any URLs that are missing in the current mapping file.

    Args:
        mapping_file_prev (Union[str, Path]): Path to the mapping file for the previous iteration.
        mapping_file_curr (Union[str, Path]): Path to the mapping file for the current iteration.

    Returns:
        dict: A dictionary where each key is a URL from the mapping, and the value is another
        dictionary containing:
            - 'requirements_input': Path to the requirements file from the previous iteration.
            - 'responsibilities_input': Path to the pruned responsibilities file from the previous iteration.
            - 'responsibilities_output': Path to the responsibilities file for the current iteration.

        If any file is missing or an error occurs, the function logs the issue and skips
        processing for that URL.
    """
    # Convert mapping files to Path objects (if they aren't already)
    mapping_file_prev = Path(mapping_file_prev)
    mapping_file_curr = Path(mapping_file_curr)

    # Load the mapping files using the Pydantic model loader
    file_mapping_prev = load_mappings_model_from_json(mapping_file_prev)
    file_mapping_curr = load_mappings_model_from_json(mapping_file_curr)

    if file_mapping_prev is None or file_mapping_curr is None:
        logger.error(
            f"Failed to load one or both mapping files: {mapping_file_prev}, {mapping_file_curr}"
        )
        return {}

    # Extract mappings from the Pydantic models' root attribute
    file_mapping_prev = file_mapping_prev.root
    file_mapping_curr = file_mapping_curr.root

    # Initialize dictionary to hold paths
    paths_dict = {}

    # Iterate through URLs from the previous mapping file
    for url, prev_paths in file_mapping_prev.items():
        if url not in file_mapping_curr:
            logger.warning(
                f"URL {url} not found in the current iteration's mapping file."
            )
            continue

        curr_paths = file_mapping_curr[url]

        # Create the dictionary of paths for the current URL
        paths_dict[url] = {
            "requirements_input": Path(
                prev_paths.reqs
            ),  # Directly access from the Pydantic model
            "responsibilities_input": Path(
                prev_paths.pruned_resps
            ),  # Access pruned responsibilities
            "responsibilities_output": Path(
                curr_paths.resps
            ),  # Current iteration responsibilities
        }

        # Verify the file paths and log errors for missing files
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


def verify_directory_paths(mapping_file_prev, mapping_file_curr) -> bool:
    """Function to test input files exists and output folder exists."""
    # Get the directory paths
    paths_dict = set_directory_paths(mapping_file_prev, mapping_file_curr)

    # Flag to track if all verifications pass
    all_valid = True

    # Check each URL entry and verify input/output paths
    for url, paths in paths_dict.items():
        requirements_input = paths["requirements_input"]
        responsibilities_input = paths["responsibilities_input"]
        responsibilities_output = paths["responsibilities_output"]

        # Verify if input files exist
        if not requirements_input.exists():
            print(
                f"Error: Requirements input file does not exist for URL {url}: {requirements_input}"
            )
            all_valid = False  # Set to False if any file is missing
        if not responsibilities_input.exists():
            print(
                f"Error: Responsibilities input file does not exist for URL {url}: {responsibilities_input}"
            )
            all_valid = False

        # Verify if the output directory exists (or create it)
        if not responsibilities_output.parent.exists():
            try:
                print(
                    f"Creating output directory for URL {url}: {responsibilities_output.parent}"
                )
                responsibilities_output.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Failed to create output directory for URL {url}: {e}")
                all_valid = False

        # Test writing a small dummy file to the output directory
        try:
            test_file = responsibilities_output.with_name("test_output.txt")
            with test_file.open("w") as f:
                f.write("Test output")
            print(f"Successfully wrote test file to {test_file}")
            test_file.unlink()  # Optionally remove the test file after verification
        except Exception as e:
            print(f"Failed to write to output directory for URL {url}: {e}")
            all_valid = False

    return all_valid


async def run_pipeline_async(
    mapping_file_prev: Union[str, Path],
    mapping_file_curr: Union[str, Path],
    model: str = "openai",
    model_id: str = "gpt-3.5-turbo",
    # n_jobs: int = -1, # Run non-parellel for now
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

    # Step 0: Test the file/dir paths
    if not verify_directory_paths(mapping_file_prev, mapping_file_curr):
        logger.info("Path verification failed, stopping execution.")
        return  # early return

    logger.info("I/O dir/file paths are correct. Proceed with processing.")

    # Step 1: Setup directory and file paths
    paths_dict = set_directory_paths(mapping_file_prev, mapping_file_curr)
    if not paths_dict:
        logger.error("Failed to set up directory paths.")
        return

    logger.info(f"paths_dict:\n{paths_dict}")

    # Step 2: Process each job posting URL and modify responsibilities
    # Use async.gather
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
                raise FileNotFoundError(f"Files ({reqs_file} not found for {url}")

            responsibilities = read_from_json_file(resps_file)
            requirements = read_from_json_file(reqs_file)

            if not responsibilities or not requirements:
                raise ValueError(f"Files are empty for {url}")

            # Use Pydantic for validation
            validated_responsibilities = Responsibilites(
                responsibilities=responsibilities
            )
            validated_requirements = Requirements(requirements=requirements)

        except (FileNotFoundError, ValueError) as error:
            logger.error(error)
            continue

        # Step 3: Modify responsibilities based on requirements
        with tqdm(
            total=len(responsibilities), desc=f"Modifying responsibilities for {url}"
        ) as pbar:
            modified_resps = await modify_multi_resps_based_on_reqs_async(
                responsibilities=validated_responsibilities.responsibilities,
                requirements=validated_requirements.requirements,
                model=model,
                model_id=model_id,
                # n_jobs=n_jobs,
            )  # the function returns a pyd object
            pbar.update(1)

        # Step 4: Save the modified responsibilities
        output_file = paths["responsibilities_output"]
        await save_to_json_file_async(modified_resps, output_file)
        logger.info(f"Modified responsibilities for {url} saved to {output_file}")

    logger.info("Pipeline execution completed.")
