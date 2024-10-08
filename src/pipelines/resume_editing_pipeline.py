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


# Pipeline to edit multiple responsibilities files
def multi_files_modification_pipeline(
    mapping_file: Union[str, Path],
    output_dir: Union[str, Path],
    model: str = "openai",
    model_id: str = "gpt-3.5-turbo",
    n_jobs: int = -1,
):
    """
    Run the pipeline to modify responsibilities for multiple job postings by matching
    them with corresponding requirements based on the mapping file.

    Args:
        mapping_file (str): Path to the mapping file containing the URLs and
                            the corresponding file paths for responsibilities and requirements.
        output_dir (str or Path): Directory where modified responsibility files should be saved.
        model (str, optional): Model name to be used for modifications.
                                Defaults to 'openai'.
        model_id (str, optional): Specific model version to be used.
                                Defaults to 'gpt-3.5-turbo'.

    Returns:
        None
    """
    # Step 0: Validate / ensure output_dir is path and exists
    output_dir = Path(output_dir)  # Ensure output directory is a Path object
    if not output_dir.exists():
        os.makedirs(output_dir)

    # Step 1: Load the mapping file
    try:
        file_mapping = read_from_json_file(mapping_file)
        logger.info(f"Loaded mapping file from {mapping_file}")
    except FileNotFoundError:
        logger.error(f"Mapping file not found: {mapping_file}")
        return
    except Exception as e:
        logger.error(f"Error loading mapping file: {e}")
        return

    # Step 2: Parallel processing to extract responsibilities and requirements paths,
    # read them, and modify them (editing each)
    for url, paths in file_mapping.items():
        logger.info(f"Processing job posting from {url}")

        # Step 2.1: Extract file paths for responsibilities and requirements
        resps_file = Path(paths["resps_flat"])
        reqs_file = Path(paths["reqs_flat"])

        # Step 2.2: Check if the files exist and are valid
        try:
            # Ensure both files exist
            if not Path(resps_file).exists() or not Path(reqs_file).exists():
                raise FileNotFoundError(
                    f"One or both files not found for job posting {url}"
                )

            # Step 2.3: Load responsibilities and requirements files
            responsibilities = read_from_json_file(resps_file)
            requirements = read_from_json_file(reqs_file)

            # Check if either file is empty
            if not responsibilities:
                raise ValueError(
                    f"Responsibilities file {resps_file} is empty for job posting {url}"
                )
            if not requirements:
                raise ValueError(
                    f"Requirements file {reqs_file} is empty for job posting {url}"
                )

        except FileNotFoundError as fnf_error:
            logger.error(fnf_error)
            continue  # Skip to the next job posting if files are missing
        except ValueError as val_error:
            logger.error(val_error)
            continue  # Skip to the next job posting if files are empty
        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {e}")
            continue

        # Step 2.4: Modify responsibilities based on requirements
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
            pbar.update(1)  # Update progress bar

        # Step 5: Save the modified responsibilities to the output directory
        output_file_name = Path(output_dir) / Path(resps_file).name.replace(
            "resps_flat", "modified_resps_flat"
        )
        save_to_json_file(modified_resps, output_file_name)
        logger.info(
            f"Modified responsibilities for URL {url} saved to {output_file_name}"
        )

    logger.info("Finished modifying all responsibilities based on the mapping file.")


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
#         excluded_key = "3.responsibilities.5"
#         # resps_flat = {k: v for k, v in resps_flat.items() if k not in excluded_keys}
#         resps_flat.pop(excluded_key, None)

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
