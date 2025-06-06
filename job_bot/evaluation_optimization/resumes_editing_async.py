"""
Filename: resume_editing_async.py
Lst updated on: 2024 Oct 21

Not tested/debugged yet
"""

import os
from pathlib import Path
import logging
from typing import Dict, Tuple
from pydantic import ValidationError
from tqdm import tqdm
import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from evaluation_optimization.resume_editor_async import TextEditorAsync

# from utils.generic_utils import save_to_json_file
from llm_providers.llm_api_utils import get_openai_api_key, get_anthropic_api_key
from models.resume_job_description_io_models import (
    OptimizedText,
    ResponsibilityMatch,
    ResponsibilityMatches,
    Responsibilities,
    Requirements,
)
from project_config import OPENAI, ANTHROPIC, GPT_4_TURBO, CLAUDE_HAIKU

# Set up logger
logger = logging.getLogger(__name__)


async def modify_resp_based_on_reqs_async(
    resp_key: str, resp: str, reqs: Dict[str, str], llm_provider: str, model_id: str
) -> Tuple[str, ResponsibilityMatch]:
    """
    *This is the async version of the modify_resps_based_on_reqs function.

    Modify a single responsibility text by aligning it with multiple job requirements.

    This function modifies one responsibility (`resp`) by matching it against multiple
    job requirements (`reqs`). The alignment process involves three stages:
    semantic alignment, entailment alignment, and dependency parsing (DP) alignment.
    The TextEditor class is used for these modifications based on the provided model.

    The process includes:
    1. Semantic Alignment: Adjusting the responsibility text to ensure it semantically
       aligns with the job requirement.
    2. Entailment Alignment: Ensuring that the responsibility text can be logically
       inferred from the job requirement.
    3. Dependency Parsing Alignment (DP): Refining the final responsibility text
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
            - 'local_modifications' (dict): A dictionary of modified responsibility texts
              for each requirement, keyed by the requirement identifier.
              The value is a dict containing the final `optimized_text`.

    Example:
        >>> await modify_resp_based_on_reqs_async(
                resp_key="resp1",
                resp="Managed a team of 5 developers",
                reqs={"req1": "Experience leading software development teams."},
                llm_provider="openai",
                model_id="gpt-4-turbo"
            )

    !Note: Rationale for Temperature Settings in Text Alignment:
        *1. Semantic Alignment (Temperature = 0.3):
        - Low temperature ensures **minimal and controlled changes** to preserve
        the core meaning of the candidate text while aligning it with the job requirement.
        This avoids excessive rewording and keeps the semantic alignment subtle.
        *2. Entailment Alignment (Temperature = 0.4):
        - Slightly higher temperature provides a bit more flexibility to introduce
        moderate changes that ensure the premise (candidate text) is logically supported
        by the hypothesis (job requirement).
        This balances between maintaining structure and strengthening entailment without
        drastically altering the original text.
        *3. Dependency Parsing Alignment (Temperature = 0.7):
        - High temperature allows for **larger structural modifications** to align
        the modified text with the original candidate text's structure and tone.
        This step provides more freedom to reshape the sentence while retaining
        authenticity and consistency with the original style of the resume.

        Overall, the temperature strategy balances precision in alignment
        (low temperature) and structural freedom (high temperature) to ensure
        a result that is both semantically aligned with the job posting and
        consistent with the original resume's one and style.
    """

    # Initialize the client based on llm_provider if needed
    if llm_provider == OPENAI:
        client = None  # * don't instantiate here; use global client in lower-level
        logger.info("OpenAI API initialized.")

    elif llm_provider == ANTHROPIC:
        client = None  # * don't instantiate here; use global client in lower-level
        logger.info("Claude API initialized.")

    elif llm_provider == "llama3":
        client = None  # No client needed for local Llama3
        logger.info("Using local Llama3 model.")

    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    # Initialize the async text editor with the async client
    text_editor = TextEditorAsync(
        llm_provider=llm_provider, model_id=model_id, client=client, max_tokens=1024
    )

    local_modifications = {}

    try:
        for req_key, req in reqs.items():
            logger.info(f"Modifying responsibility: {resp} \nwith requirement: {req}")

            # Step 1: Semantic Alignment
            revised = await text_editor.edit_for_semantics_async(
                candidate_text=resp, reference_text=req, temperature=0.3
            )  # Set temperature low to make the change light
            revised_text_1 = revised.data.optimized_text

            # Step 2: Entailment Alignment
            revised = await text_editor.edit_for_entailment_async(
                premise_text=revised_text_1, hypothesis_text=req, temperature=0.4
            )  # set temp low to make the change moderate
            revised_text_2 = revised.data.optimized_text

            # Step 3: Dependency Parsing Alignment -> Original Text
            revised = await text_editor.edit_for_dp_async(
                target_text=revised_text_2, source_text=resp, temperature=0.8
            )  # set temp high to make large change to retain original text's structure and tone
            revised_text_3 = revised.data.optimized_text

            # Store the optimized text
            optimized_text = OptimizedText(optimized_text=revised_text_3)
            local_modifications[req_key] = optimized_text

        # Wrap the modifications under optimized_by_requirements
        validated_modifications = ResponsibilityMatch(
            optimized_by_requirements=local_modifications
        )

    except Exception as e:
        logger.error(f"Failed to modify responsibility {resp_key}: {e}")
        # Fallback for error cases
        local_modifications[req_key] = OptimizedText(optimized_text=resp)
        validated_modifications = ResponsibilityMatch(
            optimized_by_requirements=local_modifications
        )

    return resp_key, validated_modifications


async def modify_multi_resps_based_on_reqs_async(
    responsibilities: Dict[str, str],
    requirements: Dict[str, str],
    llm_provider: str,
    model_id: str,
    no_of_concurrent_workers: int = 5,
) -> ResponsibilityMatches:
    """
    * Async version of the modify_multi_resps_based_on_reqs function.

    Modify multiple responsibilities by aligning them with multiple job requirements.

    This function processes multiple responsibilities by aligning each responsibility
    with multiple job requirements. It uses the `TextEditor` class to perform the
    modifications and executes the processing in parallel using 'joblib' to speed up
    the process, especially when dealing with large datasets.

    Each responsibility undergoes a three-step modification process:
    1. Semantic Alignment: Ensures that the responsibility text matches the meaning
    of the job requirement.
    2. Entailment Alignment: Ensures that the responsibility text can be logically
    inferred from the job requirement.
    3. Dependency Parsing Alignment (DP): Ensures that the structure of the
    responsibility text is preserved while aligning it with the job requirement.

    Args:
        - responsibilities (dict): A dictionary of responsibility texts, where keys are
        unique identifiers and values are the responsibility texts.
        - requirements (dict): A dictionary of job requirement texts, where keys
        are unique requirement identifiers and values are the requirement texts.
        - llm_provider (str, optional): The name of the model to be used (e.g., "openai").
        - model_id (str, optional): The specific model version to be used
        (e.g., "gpt-3.5-turbo").
        Defaults to "gpt-3.5-turbo".
        -n_jobs (int, optional): The number of parallel jobs to run. Defaults to -1,
            which means using all available processors.

    Returns:
        * ResponsibilityMatches:
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

    # Limit the number of concurrent tasks (in this case, coroutines) that
    # can run simultaneously
    semaphore = asyncio.Semaphore(no_of_concurrent_workers)  # Adjust limit as needed

    async def modify_resp_with_limit(resp_key: str, resp: str):
        logger.info(f"🔄 START processing {resp_key}")  # Log before starting

        async with semaphore:
            result = await modify_resp_based_on_reqs_async(
                resp_key, resp, requirements, llm_provider, model_id
            )
            logger.info(
                f"✅ DONE processing {resp_key}.\nNo of optimized texts: {len(result[1].optimized_by_requirements)}"
            )  # Log after finishing
            return result

    tasks = [
        modify_resp_with_limit(resp_key, resp)
        for resp_key, resp in responsibilities.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions in results if any
    modified_responsibilities = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
        elif (
            isinstance(result, tuple) and len(result) == 2
        ):  # Check if result is a tuple with 2 parameters (key and text)
            # Unpack only if it's a tuple with expected length
            resp_key, modifications = result

            # Get or initialize ResponsibilityMatch (setdefault fetches existing
            # data first and then add new)
            resp_match = modified_responsibilities.setdefault(
                resp_key, ResponsibilityMatch(optimized_by_requirements={})
            )

            # Update existing requirement matches
            resp_match.optimized_by_requirements.update(
                modifications.optimized_by_requirements
            )

        else:
            logger.warning(f"Unexpected result format: {result}")

    logger.info(
        f"Before validated by ResponsibilityMatches: \n{modified_responsibilities}"
    )  # TODO: for debugging; delete afterwards

    # Validate and wrap the final result in ResponsibilityMatches model
    try:
        validated_modified_responsibilities = ResponsibilityMatches(
            responsibilities=modified_responsibilities
        )

        logger.info(
            f"After validated by ResponsibilityMatches: \n{validated_modified_responsibilities}"
        )  # TODO: for debugging; delete afterwards

        # Ensure event loop does not stall due to rapid execution (rate limit issue)
        await asyncio.sleep(0.1)

        return validated_modified_responsibilities

    except ValidationError as e:
        logger.error(f"Validation error when creating ResponsibilityMatches: {e}")
        raise ValueError("Failed to validate modified responsibilities.")
