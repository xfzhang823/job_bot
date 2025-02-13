"""
extract_requirements_with_llms.py
Helper functions to extract requirements with LLMs.
"""

import logging
from typing import Any, Dict
from llm_providers.llm_api_utils_async import (
    call_anthropic_api_async,
    call_openai_api_async,
)
from prompts.prompt_templates import EXTRACT_JOB_REQUIREMENTS_PROMPT
from models.llm_response_models import RequirementsResponse
from project_config import GPT_35_TURBO, GPT_4_TURBO, CLAUDE_HAIKU, CLAUDE_SONNET


# Set logger
logger = logging.getLogger(__name__)


async def extract_job_requirements_with_openai_async(
    job_description: str, model_id: str = GPT_35_TURBO
) -> Dict[Any, Any]:
    """
    Extracts key requirements from the job description using GPT.

    Args:
        job_description (str): The full text of the job description.
        model_id (str): The model ID for OpenAI (default is gpt-3.5-turbo).

    Returns:
        dict: Extracted requirements including qualifications, responsibilities,
        and skills.
    """
    if not job_description:
        logger.error("job_description text is empty or invalid.")
        raise ValueError("job_description text cannot be empty.")

    # Set up the prompt
    prompt = EXTRACT_JOB_REQUIREMENTS_PROMPT.format(content=job_description)
    logger.info(f"Prompt to extract job requirements:\n{prompt}")

    # Call the async OpenAI API
    response_model = await call_openai_api_async(
        prompt=prompt,
        model_id=model_id,
        expected_res_type="json",
        json_type="requirements",
        temperature=0.5,  # * Need to set temperature higher for extracting requirements
        max_tokens=2000,
    )
    logger.info(f"Validated LLM Response Model: {response_model}")

    # Validate that the response is in the expected JobSiteResponseModel format
    if not isinstance(response_model, RequirementsResponse):
        logger.error(
            "Received response is not in expected JobSiteResponseModel format."
        )
        raise ValueError(
            "Received response is not in expected JobSiteResponseModel format."
        )

    # Return the model-dumped dictionary (from Pydantic obj to dict)
    return response_model.model_dump()


async def extract_job_requirements_with_anthropic_async(
    job_description: str, model_id: str = GPT_35_TURBO
) -> Dict:
    """
    Extracts key requirements from the job description using Anthropic.

    Args:
        job_description (str): The full text of the job description.
        model_id (str): The model ID for OpenAI (default is gpt-3.5-turbo).

    Returns:
        dict: Extracted requirements including qualifications, responsibilities,
        and skills.
    """
    if not job_description:
        logger.error("job_description text is empty or invalid.")
        raise ValueError("job_description text cannot be empty.")

    # Set up the prompt
    prompt = EXTRACT_JOB_REQUIREMENTS_PROMPT.format(content=job_description)
    logger.info(f"Prompt to extract job requirements:\n{prompt}")

    # Call the async OpenAI API
    response_model = await call_anthropic_api_async(
        prompt=prompt,
        model_id=model_id,
        expected_res_type="json",
        json_type="requirements",
        temperature=0.5,  # * Need to set temperature higher for extracting requirements
        max_tokens=2000,
    )
    logger.info(f"Validated LLM Response Model: {response_model}")

    # Validate that the response is in the expected JobSiteResponseModel format
    if not isinstance(response_model, RequirementsResponse):
        logger.error(
            "Received response is not in expected JobSiteResponseModel format."
        )
        raise ValueError(
            "Received response is not in expected JobSiteResponseModel format."
        )

    # Return the model-dumped dictionary (from Pydantic obj to dict)
    return response_model.model_dump()
