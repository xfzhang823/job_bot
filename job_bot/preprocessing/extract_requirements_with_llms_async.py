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
from job_bot.config.project_config import (
    GPT_35_TURBO,
    GPT_4_1_NANO,
    CLAUDE_HAIKU,
    CLAUDE_SONNET_3_5,
)


# Set logger
logger = logging.getLogger(__name__)


async def extract_job_requirements_with_openai_async(
    job_description: str, model_id: str = GPT_35_TURBO
) -> RequirementsResponse:
    """
    Extracts key requirements from the job description using GPT.

    Args:
        job_description (str): The full text of the job description.
        model_id (str): The model ID for OpenAI (default is gpt-3.5-turbo).

    Returns:
        RequirementsResponse: Pydantic model containing extracted requirements
        including qualifications, responsibilities, and skills.

    * ✅ Example Output:
    -------------------
    {
        "status": "success",
        "message": "Job requirements data processed successfully.",
        "data": {
            "pie_in_the_sky": [...],
            "down_to_earth": [...],
            "bare_minimum": [...],
            "cultural_fit": [...],
            "other": [...]
        }
    }
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
    return response_model


async def extract_job_requirements_with_anthropic_async(
    job_description: str, model_id: str = GPT_35_TURBO
) -> RequirementsResponse:
    """
    Extracts key requirements from the job description using Anthropic.

    Args:
        job_description (str): The full text of the job description.
        model_id (str): The model ID for OpenAI (default is gpt-3.5-turbo).

    Returns:
        RequirementsResponse: Pydantic model containing extracted requirements
        including qualifications, responsibilities, and skills.

    * ✅ Example Output:
    -------------------
    {
        "status": "success",
        "message": "Job requirements data processed successfully.",
        "data": {
            "pie_in_the_sky": [
                "Top-tier strategy consulting experience",
                "Motivated by high impact, high visibility work"
            ],
            "down_to_earth": [
                "Bachelor's degree and MBA required",
                "3-5 years of work experience preferred",
                "Strong analytical and communication skills"
            ],
            "bare_minimum": [
                "Bachelor's degree",
                "Ability to manage multiple priorities"
            ],
            "cultural_fit": [
                "Excited about contributing to a dynamic team"
            ],
            "other": [
                "Insurance industry experience is a plus"
            ]
        }
    }
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

    # Return pydantic model (do not model_dump() to dict)
    return response_model
