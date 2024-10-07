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
from utils.generic_utils import read_from_json_file, save_to_json_file


# Set up logging
logger = logging.getLogger(__name__)


def modify_one_resp_based_on_reqs(resp_key, resp, reqs, model, model_id):
    """Modify a single responsibility by matching to multiple job requirements"""
    text_editor = TextEditor(model=model, model_id=model_id, max_tokens=1024)

    local_modifications = {}
    for req_key, req in reqs.items():
        logger.info(f"Modifying Responsibility: {resp} \nwith Requirement: {req}")

        # Step 1: Align Semantic
        revised = text_editor.edit_for_semantics(
            candidate_text=resp,
            reference_text=req,
            text_id=f"{resp_key}_{req_key}",
            temperature=0.5,
        )
        revised_text_1 = revised["optimized_text"]

        # Step 2: Align Entailment
        revised = text_editor.edit_for_entailment(
            premise_text=revised_text_1,
            hypothesis_text=req,
            text_id=f"{resp_key}_{req_key}",
            temperature=0.6,
        )
        revised_text_2 = revised["optimized_text"]

        # Step 3: Align Original Sentence's DP
        revised = text_editor.edit_for_dp(
            target_text=revised_text_2,
            source_text=resp,
            text_id=f"{resp_key}_{req_key}",
            temperature=0.9,
        )
        revised_text_3 = revised["optimized_text"]

        # Store the modification for this requirement
        local_modifications[req_key] = {"optimized_text": revised_text_3}

    return (resp_key, local_modifications)


def modify_multi_resps_based_on_reqs(
    responsibilities: dict[str, str],
    requirements: dict[str, str],
    TextEditor: callable,  # Add TextEditor as an argument here
    model: str = "openai",
    model_id: str = "gpt-3.5-turbo",
    n_jobs: int = -1,
):
    """
    Modify multiple-responsibilities based on matching each of them
    with each job requirement in parallel batches using Joblib.
    """

    # Debugging: Check length of responsibilities and requirements
    logger.info(f"Number of responsibilities: {len(responsibilities)}")
    logger.info(f"Number of requirements: {len(requirements)}")

    # Initialize the result dictionary
    modified_responsibilities = {}

    # Progress bar setup
    total = len(responsibilities)
    with tqdm(total=total, desc="Modifying Responsibilities") as pbar:
        # Run the processing in parallel using joblib's Parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(modify_one_resp_based_on_reqs)(
                resp_key, resp, requirements, model, model_id
            )
            for resp_key, resp in responsibilities.items()
        )

        # Update the progress bar after all results are gathered
        pbar.update(len(results))

    # Aggregate the results
    for resp_key, modifications in results:
        modified_responsibilities[resp_key] = modifications

    return modified_responsibilities


def filter_responsibilities_by_low_scores(df, fields):
    """
    Filters out responsibilities where all specified fields have 'Low' scores for all requirements.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing responsibility and requirement data.
    - fields (list): A list of column names to check for 'Low' values.

    Returns:
    - responsibilities_to_optimize (np.array): An array of unique responsibilities
    that do not have 'Low' scores for all requirements in the specified fields.
    """
    # Group the data by Responsibility and aggregate the counts of "Low" scores for each specified field
    aggregation_dict = {field: lambda x: (x == "Low").sum() for field in fields}
    aggregation_dict["Requirement Key"] = (
        "count"  # Count the number of requirements associated with each responsibility
    )

    grouped = df.groupby("Responsibility").agg(aggregation_dict).reset_index()

    # Filter responsibilities where all specified fields have "Low" scores for all requirements
    filter_condition = grouped[fields[0]] == grouped["Requirement Key"]
    for field in fields[1:]:
        filter_condition &= grouped[field] == grouped["Requirement Key"]

    filtered_responsibilities = grouped[filter_condition]

    # Get responsibilities that don't match the above criteria
    responsibilities_to_optimize = df[
        ~df["Responsibility"].isin(filtered_responsibilities["Responsibility"])
    ]["Responsibility"].unique()

    return responsibilities_to_optimize


def flat_json_files_processing_mini_pipeline(
    job_descriptions_file: Union[str, Path],
    job_requirements_file: Union[str, Path],
    resume_json_file: Union[str, Path],
    flat_json_output_files_dir: Union[str, Path],
):
    """
    Preprocessing mini-pipeline for flattening and saving JSON files for responsibilities and requirements.

    This pipeline processes the resume and job descriptions by flattening the JSON structure of responsibilities
    and job requirements and saving the flattened data to separate JSON files.

    The pipeline:
    1. Finds new URLs (from job descriptions).
    2. Creates and saves flattened responsibilities (resume) and requirements (job posting) JSON files.

    Args:
        job_descriptions_file (str or Path): Path to the JSON file containing job descriptions.
        job_requirements_file (str or Path): Path to the extracted requirements (from job postings) JSON file.
        resume_json_file (str or Path): Path to the resume JSON file.
        flat_json_output_files_dir (str or Path): Directory where output files are stored.

    Returns:
        None
    """
    # Convert inputs to Path objects if they are not already
    if not isinstance(job_descriptions_file, Path):
        job_descriptions_file = Path(job_descriptions_file)
    if not isinstance(job_requirements_file, Path):
        job_requirements_file = Path(job_requirements_file)
    if not isinstance(resume_json_file, Path):
        resume_json_file = Path(resume_json_file)
    if not isinstance(flat_json_output_files_dir, Path):
        flat_json_output_files_dir = Path(flat_json_output_files_dir)

    logger.info(f"Job descriptions file path: {job_descriptions_file}")
    logger.info(f"Job requirements file path: {job_requirements_file}")
    logger.info(f"Resume JSON file path: {resume_json_file}")
    logger.info(f"Flat JSON output files dir: {flat_json_output_files_dir}")

    # Step 1: Read job descriptions
    try:
        job_descriptions = read_from_json_file(job_descriptions_file)
        logger.info(f"job_descriptions:\n{job_descriptions}")
        if not job_descriptions:
            logger.error("No job descriptions loaded. Exiting.")
            return
    except FileNotFoundError as e:
        logger.error(f"Job descriptions file not found: {e}")
        return
    except Exception as e:
        logger.error(f"Error reading job descriptions: {e}")
        return

    # Step 2: Find new URLs and corresponding file paths
    try:
        new_urls_and_file_paths = get_new_urls_and_flat_json_file_paths(
            job_descriptions, flat_json_output_files_dir
        )
        if not new_urls_and_file_paths:
            logger.info("No new flat JSON files to process. Exiting.")
            return
        logger.info(
            f"Found {len(new_urls_and_file_paths)} new flat JSON files to process."
        )
    except Exception as e:
        logger.error(f"Error retrieving new URLs and file paths: {e}")
        return

    # Step 3: Create and save flattened job requirements files for each URL
    try:
        for url, file_path in new_urls_and_file_paths.items():
            process_and_save_requirements_by_url(
                requirements_json_file=job_requirements_file,
                url=url,
                requirements_flat_json_file=file_path,
            )
            logger.info(f"Requirements flat file saved: {file_path}")
        logger.info("All requirements flat files saved.")
    except Exception as e:
        logger.error(f"Error processing requirements files: {e}")
        return

    # Step 4: Create and save flattened responsibilities file from the resume
    try:
        responsibilities_flat_json_file = (
            flat_json_output_files_dir / "responsibilities_flat.json"
        )
        process_and_save_responsibilities_from_resume(
            resume_json_file=resume_json_file,
            responsibilities_flat_json_file=responsibilities_flat_json_file,
        )
        logger.info(
            f"Responsibilities flat file saved: {responsibilities_flat_json_file}"
        )
    except Exception as e:
        logger.error(f"Error processing responsibilities file: {e}")
        return

    logger.info(
        "All responsibilities and requirements flattened files created and saved."
    )


def run_pipeline(
    responsibilities_flat_json_file,
    requirements_flat_json_file,
    modified_resps_flat_json_file,
):
    """Pipeline to modify resume"""

    # Step 0. Create flat JSON files
    # If not, then create them from resume and job requirements
    # Step 1. Read responsibility vs requirement similarity metrics JSON files
    if not (
        os.path.exists(responsibilities_flat_json_file)
        and os.path.exists(requirements_flat_json_file)
    ):
        raise FileNotFoundError("One or both of the JSON files do not exist.")

    else:
        # Step 1. Read both dictionaries into
        resps_flat = read_from_json_file(responsibilities_flat_json_file)
        reqs_flat = read_from_json_file(requirements_flat_json_file)

        # Step 2. Exclude certain responsibilities from modification
        # (to be added back afterwards-factual statements like "promoted to ... in ...")
        excluded_key = "3.responsibilities.5"
        # resps_flat = {k: v for k, v in resps_flat.items() if k not in excluded_keys}
        resps_flat.pop(excluded_key, None)

        # Step 3: Modify responsibility texts
        gpt3 = "gpt-3.5-turbo"
        gpt4 = "gpt-4-turbo"

        # modified_resps = modify_responsibilities_based_on_requirements(
        #     responsibilities=resps_flat,
        #     requirements=reqs_flat,
        #     TextEditor=TextEditor,  # Pass the class instance
        #     model="openai",
        #     model_id=gpt3,
        # )
        with tqdm(total=len(resps_flat), desc="Modifying Responsibilities") as pbar:
            modified_resps = modify_multi_resps_based_on_reqs(
                responsibilities=resps_flat,
                requirements=reqs_flat,
                TextEditor=TextEditor,  # Pass your TextEditor class here
                model="openai",
                model_id="gpt-3.5-turbo",
                n_jobs=-1,
            )
            pbar.update(1)  # Update progress bar

        # Step 4: Save modified responsibilities
        save_to_json_file(modified_resps, modified_resps_flat_json_file)
