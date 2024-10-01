import os
import logging
import json
import pandas as pd
from tqdm import tqdm
import math
from joblib import Parallel, delayed
from evaluation_optimization.resume_editor import TextEditor
from evaluation_optimization.text_similarity_finder import AsymmetricTextSimilarity
from evaluation_optimization.similarity_metric_eval import categorize_scores
from evaluation_optimization.resume_editor import TextEditor
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


def run_pipeline(
    responsibilities_flat_json_file,
    requirements_flat_json_file,
    modified_resps_flat_json_file,
):
    """Pipeline to modify resume"""

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
