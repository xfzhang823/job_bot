"""
Filename: resume_editing.py
Lst updated on: 2024 Oct 9

*Still working on!!!!
"""

import os
from pathlib import Path
import logging
from joblib import Parallel, delayed
from typing import Any
from tqdm import tqdm
from openai import OpenAI
from evaluation_optimization.resume_editor import TextEditor
from utils.generic_utils import save_to_json_file
from llm_providers.llm_api_utils import get_openai_api_key
from models.resume_job_description_io_models import (
    OptimizedText,
    ResponsibilityMatch,
    ResponsibilityMatches,
    Responsibilities,
    Requirements,
)


# Set up logger
logger = logging.getLogger(__name__)


def modify_resp_based_on_reqs(
    resp_key: str, resp: str, reqs: dict, model: str, model_id: str
) -> ResponsibilityMatch:
    """
    *This is the parallel version of the function, using joblib!

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
        model (str): Name of the model to be used (e.g., "openai").
        model_id (str): The specific model version to be used (e.g., "gpt-3.5-turbo").

    Returns:
        tuple: A tuple containing:
            - 'resp_key' (str): The same responsibility_key passed to the function.
            - 'local_modifications' (dict): A dictionary of modified responsibility texts for
              each requirement, keyed by the requirement identifier. The value is a dict
              containing the final `optimized_text`.

    Example:
        >>> modify_resp_based_on_reqs(
                resp_key="resp1",
                resp="Managed a team of 5 developers",
                reqs={"req1": "Experience leading software development teams."},
                model="openai",
                model_id="gpt-3.5-turbo"
            )
    """
    # Instantiate the client inside the worker process
    openai_api_key = get_openai_api_key()
    client = OpenAI(api_key=openai_api_key)

    # Instantiate TextEditor class
    text_editor = TextEditor(
        llm_provider=model, model_id=model_id, client=client, max_tokens=1024
    )

    local_modifications = {}

    # Function to process a single requirement
    def process_requirement(req_key, req, text_editor):
        try:
            for req_key, req in reqs.items():
                logger.info(
                    f"Modifying responsibility: {resp} \nwith requirement: {req}"
                )

            # Step 1: Semantic Alignment
            revised = text_editor.edit_for_semantics(
                candidate_text=resp, reference_text=req, temperature=0.5
            )
            revised_text_1 = revised.data.optimized_text

            # Step 2: Entailment Alignment
            revised = text_editor.edit_for_entailment(
                premise_text=revised_text_1, hypothesis_text=req, temperature=0.6
            )
            revised_text_2 = revised.data.optimized_text

            # Step 3: Dependency Parsing Alignment
            revised = text_editor.edit_for_dp(
                target_text=revised_text_2, source_text=resp, temperature=0.9
            )
            revised_text_3 = revised.data.optimized_text

            # Create and return the OptimizedText object
            return req_key, OptimizedText(optimized_text=revised_text_3)

        except Exception as e:
            logger.error(f"Failed to process req_key={req_key}: {e}")
            # Fallback to original responsibility text in case of an error
            return req_key, OptimizedText(optimized_text=resp)

        # Instantiate the client inside the worker process

    openai_api_key = get_openai_api_key()
    client = OpenAI(api_key=openai_api_key)

    # Instantiate TextEditor class
    text_editor = TextEditor(
        llm_provider=model, model_id=model_id, client=client, max_tokens=1024
    )

    # Process requirements in parallel using joblib
    results = Parallel(n_jobs=-1)(
        delayed(process_requirement)(req_key, req, text_editor)
        for req_key, req in reqs.items()
    )

    # Collect results into a dictionary
    local_modifications = {
        req_key: optimized_text for req_key, optimized_text in results
    }

    # Validate the entire set of modifications for this responsibility
    validated_modifications = ResponsibilityMatch(
        optimized_by_requirements=local_modifications
    )

    return resp_key, validated_modifications

    #         # Create and return the OptimizedText object
    #         return req_key, OptimizedText(optimized_text=revised_text_3)

    #         # Validate the final optimized text using a pydantic model
    #         optimized_text = OptimizedText(optimized_text=revised_text_3)

    #         # Store the modification for this requirement
    #         local_modifications[req_key] = optimized_text.model_dump()

    #     # Validate the entire set of modifications for this repsonsiblity
    #     validated_modifications = ResponsibilityMatch(
    #         optimized_by_requirements=local_modifications
    #     )
    # except Exception as e:
    #     logger.error(f"Failed to modify responsibility {resp_key}: {e}")
    #     # Ensure a fallback for this responsibility in case of an error
    #     local_modifications[req_key] = OptimizedText(optimized_text=resp)

    # return (resp_key, validated_modifications)  # Returns a pyd obj


def modify_multi_resps_based_on_reqs(
    responsibilities: dict[str, str],
    requirements: dict[str, str],
    # TextEditor: callable,  # Add TextEditor as an argument here
    model: str = "openai",
    model_id: str = "gpt-3.5-turbo",
    n_jobs: int = 4,  # Start setting this not too high (avoid hitting rate limit)
) -> ResponsibilityMatches:
    """
    Modify multiple responsibilities by aligning them with multiple job requirements in parallel.

    This function processes multiple responsibilities by aligning each responsibility
    with multiple job requirements. It uses the `TextEditor` class to perform the
    modifications and executes the processing in parallel using `joblib` to speed up
    the process, especially when dealing with large datasets.

    Each responsibility undergoes a three-step modification process:
    1. Semantic Alignment**: Ensures that the responsibility text matches the meaning of the
       job requirement.
    2. Entailment Alignment: Ensures that the responsibility text can be logically inferred
       from the job requirement.
    3. Dependency Parsing Alignment (DP): Ensures that the structure of the responsibility
       text is preserved while aligning it with the job requirement.

    *Becasue we are using concurrency, OpenAI API must be instantiated here
    *(at the higher level).

    Args:
        -responsibilities (dict): A dictionary of responsibility texts, where keys are
            unique identifiers and values are the responsibility texts.
        -requirements (dict): A dictionary of job requirement texts, where keys are unique
            requirement identifiers and values are the requirement texts.
        -TextEditor (callable): The class responsible for performing the text modifications.
        -model (str, optional): The name of the model to be used (e.g., "openai").
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

    # Debugging: Check length of responsibilities and requirements
    logger.info(f"Number of responsibilities: {len(responsibilities)}")
    logger.info(f"Number of requirements: {len(requirements)}")

    # Validate the input resonsibilities using pydantic models
    validated_responsibilities = Responsibilities(responsibilities=responsibilities)
    validated_requirements = Requirements(requirements=requirements)

    # # Initialize the OpenAI client here to pass it into the parallel calls
    # openai_api_key = get_openai_api_key()
    # client = OpenAI(api_key=openai_api_key)  # Instantiate once and pass

    # Initialize the result dictionary
    modified_responsibilities = {}

    # Progress bar setup
    total = len(responsibilities)
    with tqdm(total=total, desc="Modifying Responsibilities") as pbar:
        # Run the processing in parallel using joblib's Parallel
        results = list(
            Parallel(n_jobs=n_jobs)(
                delayed(modify_resp_based_on_reqs)(
                    resp_key,
                    resp,
                    validated_requirements.requirements,
                    model,
                    model_id,
                    # client,
                )
                for resp_key, resp in validated_responsibilities.responsibilities.items()
            )
        )
        # Update the progress bar after all results are gathered
        pbar.update(len(results))

    # Aggregate the results
    for result in results:
        if result is not None and isinstance(result, tuple):
            resp_key, modifications = result
            if modifications is not None:
                modified_responsibilities[resp_key] = modifications

    # Validate the entire output with ResponsibilityMatches model
    # and return as pyd object
    validated_output = ResponsibilityMatches(responsibilities=modified_responsibilities)

    logger.info(f"Number of requirements: {len(requirements)}")

    # returns a pyd obj (Pydantic models are picklable, so they will serialize
    # and deserialize correctly when passed between processes. )
    return validated_output
