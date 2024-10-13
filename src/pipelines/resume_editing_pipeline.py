import os
from pathlib import Path
import logging
from typing import Union
from tqdm import tqdm
from joblib import Parallel, delayed
from evaluation_optimization.resume_editor import TextEditor
from evaluation_optimization.text_similarity_finder import AsymmetricTextSimilarity
from evaluation_optimization.metrics_calculator import categorize_scores
from evaluation_optimization.evaluation_optimization_utils import (
    get_new_urls_and_metrics_file_paths,
    get_new_urls_and_flat_json_file_paths,
    process_and_save_requirements_by_url,
    process_and_save_responsibilities_from_resume,
)
from evaluation_optimization.resumes_editing import modify_multi_resps_based_on_reqs
from utils.generic_utils import read_from_json_file, save_to_json_file


# Set up logging
logger = logging.getLogger(__name__)


def set_directory_paths(
    mapping_file_prev: Union[str, Path], mapping_file_curr: Union[str, Path]
) -> dict:
    """
    Set up directory and file paths based on the mapping files from the previous and current iterations.

    Args:
        mapping_file_prev (Union[str, Path]): Path to the mapping file from the previous iteration.
        mapping_file_curr (Union[str, Path]): Path to the mapping file for the current iteration.

    Returns:
        dict: A dictionary containing paths for requirements, responsibilities, and output directories.
    """
    # Load the mapping files
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
            "reqs_flat": Path(prev_paths["reqs_flat"]),
            "pruned_resps_flat": Path(prev_paths["pruned_resps_flat"]),
            "resps_modified": Path(curr_paths["resps_flat"]).parent / "resps_modified",
        }

        # Ensure the output directory exists;
        # exist_ok=True allows the mkdir function to proceed without any issue
        # if the directory already exists.
        paths_dict[url]["resps_modified"].mkdir(parents=True, exist_ok=True)

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
    # Step 1: Set up directory and file paths
    paths_dict = set_directory_paths(mapping_file_prev, mapping_file_curr)
    if not paths_dict:
        logger.error("Failed to set up directory paths.")
        return

    # Step 2: Process each job posting URL and modify responsibilities
    for url, paths in paths_dict.items():
        logger.info(f"Processing job posting from {url}")

        try:
            # Extract file paths for requirements and pruned responsibilities
            reqs_file = paths["reqs_flat"]
            resps_file = paths["pruned_resps_flat"]

            # Load responsibilities and requirements files
            if not reqs_file.exists() or not resps_file.exists():
                raise FileNotFoundError(f"Files not found for {url}")

            responsibilities = read_from_json_file(resps_file)
            requirements = read_from_json_file(reqs_file)

            if not responsibilities or not requirements:
                raise ValueError(f"Files are empty for {url}")

        except (FileNotFoundError, ValueError) as error:
            logger.error(error)
            continue

        # Step 3: Modify responsibilities based on requirements
        with tqdm(
            total=len(responsibilities), desc=f"Modifying responsibilities for {url}"
        ) as pbar:
            modified_resps = modify_multi_resps_based_on_reqs(
                responsibilities=responsibilities,
                requirements=requirements,
                model=model,
                model_id=model_id,
                n_jobs=n_jobs,
            )
            pbar.update(1)

        # Step 4: Save the modified responsibilities
        output_file = paths["resps_modified"] / resps_file.name.replace(
            "pruned_resps_flat", "modified_resps_flat"
        )
        save_to_json_file(modified_resps, output_file)
        logger.info(f"Modified responsibilities for {url} saved to {output_file}")

    logger.info("Pipeline execution completed.")


# def run_pipeline(
#     responsibilities_flat_json_file,
#     requirements_flat_json_file,
#     modified_resps_flat_json_file,
# ):
#     """Pipeline to modify resume"""

#     # Step 1. Read responsibility vs requirement similarity metrics JSON files
#     if not (
#         os.path.exists(responsibilities_flat_json_file)
#         and os.path.exists(requirements_flat_json_file)
#     ):
#         raise FileNotFoundError("One or both of the JSON files do not exist.")

#     else:
#         # Step 1. Read both dictionaries into
#         resps_flat = read_from_json_file(responsibilities_flat_json_file)
#         reqs_flat = read_from_json_file(requirements_flat_json_file)

#         # Step 2. Exclude certain responsibilities from modification
#         # (to be added back afterwards-factual statements like "promoted to ... in ...")


#         # Step 3: Modify responsibility texts
#         gpt3 = "gpt-3.5-turbo"
#         gpt4 = "gpt-4-turbo"

#         # modified_resps = modify_responsibilities_based_on_requirements(
#         #     responsibilities=resps_flat,
#         #     requirements=reqs_flat,
#         #     TextEditor=TextEditor,  # Pass the class instance
#         #     model="openai",
#         #     model_id=gpt3,
#         # )
#         with tqdm(total=len(resps_flat), desc="Modifying Responsibilities") as pbar:
#             modified_resps = modify_multi_resps_based_on_reqs(
#                 responsibilities=resps_flat,
#                 requirements=reqs_flat,
#                 TextEditor=TextEditor,  # Pass your TextEditor class here
#                 model="openai",
#                 model_id="gpt-3.5-turbo",
#                 n_jobs=-1,
#             )
#             pbar.update(1)  # Update progress bar

#         # Step 4: Save modified responsibilities
#         save_to_json_file(modified_resps, modified_resps_flat_json_file)
