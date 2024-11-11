"""
Filename: resume_editing.py
Lst updated on: 2024 Oct 9
"""

import os
from pathlib import Path
import logging
from joblib import Parallel, delayed
from typing import Any, Tuple, Dict, Union
from tqdm import tqdm
from openai import OpenAI
from evaluation_optimization.resume_editor import TextEditor
from utils.generic_utils import save_to_json_file
from utils.llm_api_utils import get_openai_api_key
from models.resume_job_description_io_models import (
    OptimizedText,
    ResponsibilityMatch,
    ResponsibilityMatches,
    Responsibilites,
    Requirements,
)


# Set up logger
logger = logging.getLogger(__name__)


def modify_resp_based_on_reqs(
    resp_key: str, resp: str, reqs: dict, llm_provider: str, model_id: str
) -> Tuple[str, ResponsibilityMatch]:
    """
    *This is the sequential version of the modify_resps_based_on_reqs function.

    Modify a single responsibility text by aligning it with multiple job requirements.

    This function modifies one responsibility (`resp`) by matching it against multiple
    job requirements (`reqs`). The alignment process involves three stages:
    semantic alignment, entailment alignment, and dependency parsing (DP) alignment.
    The TextEditor class is used for these modifications based on the provided model.

    The process includes:
    1. **Semantic Alignment**: Adjusting the responsibility text to ensure it semantically
       aligns with the job requirement.
    2. **Entailment Alignment**: Ensuring that the responsibility text can be logically
       inferred from the job requirement.
    3. **Dependency Parsing Alignment (DP)**: Refining the final responsibility text
       while maintaining its original structure as much as possible.

    Args:
        resp_key (str): Unique identifier for the responsibility.
        resp (str): The responsibility text to be modified.
        reqs (dict): A dictionary of job requirements, where keys are unique requirement
            identifiers and values are the requirement texts.
        llm_provider (str): The language model provider (e.g., "openai").
        model_id (str): The specific model version to be used (e.g., "gpt-3.5-turbo").

    Returns:
        tuple: A tuple containing:
            - 'resp_key' (str): The same responsibility_key passed to the function.
            - 'validated_modifications' (ResponsibilityMatch): A Pydantic model containing
              a dictionary of modified responsibility texts for each requirement, keyed by
              the requirement identifier, with each entry containing the final
              `optimized_text`.

    Example:
        >>> modify_resp_based_on_reqs(
                resp_key="resp1",
                resp="Managed a team of 5 developers",
                reqs={"req1": "Experience leading software development teams."},
                llm_provider="openai",
                model_id="gpt-3.5-turbo"
            )
    """
    # Instantiate the client within the function for each responsibility
    openai_api_key = get_openai_api_key()  # Fetch the API key
    client = OpenAI(api_key=openai_api_key)  # Instantiate the OpenAI client here

    text_editor = TextEditor(
        llm_provider=llm_provider, model_id=model_id, client=client, max_tokens=1024
    )

    local_modifications = {}

    try:
        for req_key, req in reqs.items():
            logger.info(f"Modifying responsibility: {resp} \nwith requirement: {req}")

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

            # Validate the final optimized text using a pydantic model
            optimized_text = OptimizedText(optimized_text=revised_text_3)

            # Store the modification for this requirement
            local_modifications[req_key] = optimized_text.model_dump()

        # Validate the entire set of modifications for this responsibility
        validated_modifications = ResponsibilityMatch(
            optimized_by_requirements=local_modifications
        )

    except Exception as e:
        logger.error(f"Failed to modify responsibility {resp_key}: {e}")
        # Ensure a fallback for this responsibility in case of an error
        local_modifications[req_key] = OptimizedText(optimized_text=resp)

    return (resp_key, validated_modifications)  # Returns a Pydantic object


def modify_multi_resps_based_on_reqs(
    responsibilities: Union[dict[str, str], Responsibilites],
    requirements: Union[dict[str, str], Requirements],
    llm_provider: str = "openai",
    model_id: str = "gpt-3.5-turbo",
) -> ResponsibilityMatches:
    """
    This is the Sequential version of the modify_multi_resps_based_on_reqs function.

    Modify multiple responsibilities by aligning them with multiple job requirements.

    This function processes multiple responsibilities by aligning each responsibility
    with multiple job requirements. It uses the `TextEditor` class to perform the
    modifications and executes the processing in parallel using 'joblib' to speed up
    the process, especially when dealing with large datasets.

    Each responsibility undergoes a three-step modification process:
    1. Semantic Alignment: Ensures that the responsibility text matches the meaning of the
       job requirement.
    2. Entailment Alignment: Ensures that the responsibility text can be logically inferred
       from the job requirement.
    3. Dependency Parsing Alignment (DP): Ensures that the structure of the responsibility
       text is preserved while aligning it with the job requirement.

    Args:
        -responsibilities (dict): A dictionary of responsibility texts, where keys are
            unique identifiers and values are the responsibility texts.
        -requirements (dict): A dictionary of job requirement texts, where keys are unique
            requirement identifiers and values are the requirement texts.
        -TextEditor (callable): The class responsible for performing the text modifications.
        -llm_provider (str, optional): The name of the model to be used (e.g., "openai").
            Defaults to "openai".
        -model_id (str, optional): The specific model version to be used (e.g., "gpt-3.5-turbo").
            Defaults to "gpt-3.5-turbo".
        -n_jobs (int, optional): The number of parallel jobs to run. Defaults to -1,
            which means using all available processors.

    Returns:
        Pydantic object of a dictionary where keys are responsibility identifiers
        and values are dictionaries of modified responsibility texts, each aligned
        with multiple job requirements.

    Example:
        >>> modify_multi_resps_based_on_reqs(
                responsibilities={"resp1": "Managed a team of 5 developers"},
                requirements={"req1": "Experience leading software development teams."},
                TextEditor=TextEditor,
                model="openai",
                model_id="gpt-3.5-turbo",
                n_jobs=-1
            )
    """
    # Validate the input responsibilities using Pydantic models
    # Ensure responsibilities and requirements are Pydantic models
    if isinstance(responsibilities, dict) and isinstance(requirements, dict):
        validated_responsibilities = Responsibilites(responsibilities=responsibilities)
        validated_requirements = Requirements(requirements=requirements)
    else:
        validated_responsibilities = responsibilities
        validated_requirements = requirements

    # Debugging: Check length of responsibilities and requirements
    logger.info(
        f"Number of responsibilities: {len(validated_responsibilities.responsibilities)}"
    )
    logger.info(f"Number of requirements: {len(validated_requirements.requirements)}")

    # Initialize the result dictionary
    modified_responsibilities = {}

    # Sequential processing of responsibilities
    for resp_key, resp in validated_responsibilities.responsibilities.items():
        try:
            # Call the function to modify a single responsibility based on the requirements
            reqs = validated_requirements.requirements
            result = modify_resp_based_on_reqs(
                resp_key=resp_key,
                resp=resp,
                reqs=validated_requirements.requirements,
                llm_provider=llm_provider,
                model_id=model_id,
            )

            # If the result is a valid tuple (resp_key, modifications), add it to the output dictionary
            if result and isinstance(result, tuple):
                modified_responsibilities[result[0]] = result[1]

        except Exception as e:
            logger.error(f"Error processing responsibility {resp_key}: {e}")

    # Validate the entire output with ResponsibilityMatches model
    validated_output = ResponsibilityMatches(responsibilities=modified_responsibilities)

    # Log the number of requirements for debugging
    logger.info(f"Number of requirements: {len(validated_requirements.requirements)}")

    return validated_output
