"""
Filename: resume_editing_async.py
Lst updated on: 2024 Oct 21

Not tested/debugged yet
"""

import os
from pathlib import Path
import logging
from joblib import Parallel, delayed
from typing import Any
from tqdm import tqdm
import asyncio
from openai import OpenAI, AsyncOpenAI
from evaluation_optimization.resume_editor_async import TextEditor_async
from utils.generic_utils import save_to_json_file
from utils.llm_data_utils import get_openai_api_key
from models.resume_job_description_io_models import (
    OptimizedText,
    ResponsibilityMatch,
    ResponsibilityMatches,
    Responsibilites,
    Requirements,
)


# Set up logger
logger = logging.getLogger(__name__)


async def modify_resp_based_on_reqs_async(resp_key, resp, reqs, model, model_id):
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
              for each requirement, keyed by the requirement identifier. The value is a dict
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
    # Initialize the async client here for OpenAI (replace with correct async client instantiation)
    openai_api_key = get_openai_api_key()
    client = AsyncOpenAI(api_key=openai_api_key)  # Use an AsyncOpenAI client
    logger.info("OpenAI API initialized.")

    # Initialize the async text editor with the async client
    text_editor = TextEditor_async(
        model=model, model_id=model_id, client=client, max_tokens=1024
    )

    local_modifications = {}

    try:
        for req_key, req in reqs.items():
            logger.info(f"Modifying responsibility: {resp} \nwith requirement: {req}")
            # Make asynchronous API calls
            revised = await text_editor.edit_for_semantics_async(
                candidate_text=resp, reference_text=req
            )
            revised_text_1 = revised["optimized_text"]

            revised = await text_editor.edit_for_entailment_async(
                premise_text=revised_text_1, hypothesis_text=req
            )
            revised_text_2 = revised["optimized_text"]

            revised = await text_editor.edit_for_dp_async(
                target_text=revised_text_2, source_text=resp
            )
            revised_text_3 = revised["optimized_text"]

            optimized_text = OptimizedText(optimized_text=revised_text_3)
            local_modifications[req_key] = optimized_text.model_dump()

    except Exception as e:
        logger.error(f"Failed to modify responsibility {resp_key}: {e}")
        local_modifications[req_key] = OptimizedText(optimized_text=resp)

    return resp_key, local_modifications


async def modify_multi_resps_based_on_reqs_async(
    responsibilities, requirements, model, model_id
):
    tasks = []
    for resp_key, resp in responsibilities.items():
        tasks.append(
            modify_resp_based_on_reqs_async(
                resp_key, resp, requirements, model, model_id
            )
        )

    results = await asyncio.gather(*tasks)
    modified_responsibilities = {result[0]: result[1] for result in results}
    return modified_responsibilities
