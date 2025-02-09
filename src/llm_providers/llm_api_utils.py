"""
Filename: llm_api_utils.py
Author: Xiao-Fei Zhang
Last Updated: 2024-12-12

Utils classes/methods for data extraction, parsing, and manipulation
"""

# Built-in & External libraries
import os
import json
import logging
import re
from io import StringIO
from typing import Optional, Union, cast
from dotenv import load_dotenv
import pandas as pd
from pydantic import ValidationError

# LLM imports
from openai import OpenAI
import ollama
from anthropic import Anthropic

# from own modules
from models.llm_response_models import (
    CodeResponse,
    JSONResponse,
    TabularResponse,
    TextResponse,
    EditingResponse,
    JobSiteResponse,
    RequirementsResponse,
)
from models.resume_job_description_io_models import Requirements
from llm_providers.llm_response_validators import (
    validate_json_type,
    validate_response_type,
)
from project_config import (
    GPT_35_TURBO,
    GPT_4,
    GPT_4_TURBO,
    CLAUDE_HAIKU,
    CLAUDE_SONNET,
    CLAUDE_OPUS,
)

# Setup logger
logger = logging.getLogger(__name__)


# API key util functions
def get_openai_api_key() -> str:
    """Retrieves the OpenAI API key from environment variables."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found. Please set it in the .env file.")
        raise EnvironmentError("OpenAI API key not found.")
    return api_key


def get_claude_api_key() -> str:
    """Retrieves the Claude API key from environment variables."""
    load_dotenv()
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        logger.error("Claude API key not found. Please set it in the .env file.")
        raise EnvironmentError("Claude API key not found.")
    return api_key


# API Calling Functions
# Call LLM API
def call_api(
    client: Optional[Union[OpenAI, Anthropic]],
    model_id: str,
    prompt: str,
    expected_res_type: str,
    json_type: str,
    temperature: float,
    max_tokens: int,
    llm_provider: str,
) -> Union[
    JSONResponse,
    TabularResponse,
    CodeResponse,
    TextResponse,
    EditingResponse,
    JobSiteResponse,
    Requirements,
]:
    """
    Unified function to handle API calls for OpenAI, Claude, and Llama. This method handles
    provider-specific nuances (e.g., multi-block responses for Claude) and validates responses
    against expected types and Pydantic models.

    Args:
        client (Optional[Union[AsyncOpenAI, AsyncAnthropic]]):
            The API client instance for the respective provider.
            If None, a new client is instantiated.
        model_id (str):
            The model ID to use for the API call (e.g., "gpt-4-turbo" for OpenAI).
        prompt (str):
            The input prompt for the LLM.
        expected_res_type (str):
            The expected type of response (e.g., "json", "tabular", "str", "code").
        json_type (str):
            Specifies the type of JSON model for validation (e.g., "job_site", "editing").
        temperature (float):
            Sampling temperature for the LLM.
        max_tokens (int):
            Maximum number of tokens for the response.
        llm_provider (str):
            The name of the LLM provider ("openai", "claude", or "llama3").

    Returns:
        Union[JSONResponse, TabularResponse, CodeResponse, TextResponse, EditingResponseModel,
        JobSiteResponseModel]:
            The validated and structured response.

    Raises:
        ValueError: If the response cannot be validated or parsed.
        TypeError: If the response type does not match the expected format.
        Exception: For other unexpected errors during API interaction.

    Notes:
    - OpenAI returns single-block responses, while Claude may return multi-block responses.
    - Llama3 API is synchronous and is executed using an async executor. that needs special treatment.
    #* Therefore, the API calling for each LLM provider need to remain separate:
    #* Combining them into a single code block will have complications;
    #* keep them separate here for each provider is a more clean and modular.

    Examples:
    >>> await call_api_async(
            client=openai_client,
            model_id="gpt-4-turbo",
            prompt="Translate this text to French",
            expected_res_type="json",
            json_type="editing",
            temperature=0.5,
            max_tokens=100,
            llm_provider="openai"
        )
    """

    try:
        logger.info(f"Making API call with expected response type: {expected_res_type}")
        response_content = ""

        # Step  1: Make the API call and receive the response
        if llm_provider == "openai":
            openai_client = cast(OpenAI, client)  # Cast to OpenAI
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant who adheres to instructions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_content = response.choices[0].message.content

        elif llm_provider == "claude":
            claude_client = cast(Anthropic, client)  # Cast to Anthropic (Claude)
            system_instruction = (
                "You are a helpful assistant who adheres to instructions."
            )
            response = claude_client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": system_instruction + prompt}],
                temperature=temperature,
            )

            # *Need to add an extra step to extract content from response object's TextBlocks
            # *(Unlike GPT and LlaMA, Claude uses multi-blocks in its responses:
            # *The content attribute of Message is a list of TextBlock objects,
            # *whereas others wrap everything into a single block.)
            response_content = (
                response.content[
                    0
                ].text  # pylint: disable=attribute-defined-outside-init
                if hasattr(response.content[0], "text")
                else str(response.content[0])
            )

        elif llm_provider == "llama3":
            # Construct an instance of Options
            options = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "batch_size": 10,
                "retry_enabled": True,
            }
            response = ollama.generate(model=model_id, prompt=prompt, options=options)  # type: ignore
            response_content = response["response"]

        logger.info(f"Raw {llm_provider} Response: {response_content}")

        # Validate response is not empty
        if not response_content:
            raise ValueError(f"Received an empty response from {llm_provider} API.")

        # Validate response type (generic text, JSON, tabular, and code)
        validated_response_model = validate_response_type(
            response_content=response_content, expected_res_type=expected_res_type
        )

        # Further validate JSONResponse -> edit response or job site response models
        if expected_res_type == "json":
            # Ensure validated_response_content is a JSONResponse.
            if isinstance(validated_response_model, JSONResponse):
                # Pass directly to validate_json_type for further validation
                validated_response_model = validate_json_type(
                    response_model=validated_response_model, json_type=json_type
                )
                logger.debug(
                    f"Validated response model type: {type(validated_response_model).__name__}"
                )  # Debugging
            else:
                raise TypeError(
                    "Expected validated response content needs to be a JSONResponse model."
                )

        return validated_response_model

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Validation or parsing error: {e}")
        raise ValueError(f"Invalid format received from {llm_provider} API: {e}")
    except Exception as e:
        logger.error(f"{llm_provider} API call failed: {e}")
        raise


def call_openai_api(
    prompt: str,
    model_id: str = GPT_4_TURBO,
    expected_res_type: str = "str",
    json_type: str = "",
    temperature: float = 0.4,
    max_tokens: int = 1056,
    client: Optional[OpenAI] = None,
) -> Union[
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
    EditingResponse,
    JobSiteResponse,
]:
    """
    Calls OpenAI API and parses response.

    Args:
        - prompt (str): The prompt to send to the API.
        - client (Optional[OpenAI]): An OpenAI client instance.
        - If None, a new client is instantiated.
        - model_id (str): Model ID to use for the API call.
        - expected_res_type (str): The expected type of response from the API ('str', 'json',
        'tabular', or 'code').
        - context_type (str): Context type for JSON responses ("editing" or "job_site").
        - temperature (float): Controls the creativity of the response.
        max_tokens (int): Maximum number of tokens for the response.

    Returns:
        Union[TextResponse, JSONResponse, TabularResponse, CodeResponse, EditingResponseModel,
        JobSiteResponseModel]: The structured response from the API as
        a Pydantic model instance.
    """
    # Use provided client or initialize a new one if not given
    openai_client = client if client else OpenAI(api_key=get_openai_api_key())
    logger.info("OpenAI client ready for API call.")

    # Call call_api function and return
    return call_api(
        openai_client,
        model_id,
        prompt,
        expected_res_type,
        json_type,
        temperature,
        max_tokens,
        "openai",
    )


def call_claude_api(
    prompt: str,
    model_id: str = CLAUDE_SONNET,
    expected_res_type: str = "str",
    json_type: str = "",
    temperature: float = 0.4,
    max_tokens: int = 1056,
    client: Optional[Anthropic] = None,
) -> Union[
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
    EditingResponse,
    JobSiteResponse,
]:
    """
    Calls the Claude API to generate responses based on a given prompt and expected response type.

    Args:
        prompt (str): The prompt to send to the API.
        - model_id (str): Model ID to use for the Claude API call.
        - expected_res_type (str): The expected type of response from the API
        ('str', 'json', 'tabular', or 'code').
        - context_type (str): Context type for JSON responses ("editing" or "job_site").
        - temperature (float): Controls the creativity of the response.
        - max_tokens (int): Maximum number of tokens for the response.
        - client (Optional[Anthropic]): A Claude client instance.
        If None, a new client is instantiated.

    Returns:
        Union[TextResponse, JSONResponse, TabularResponse, CodeResponse, EditingResponseModel, JobSiteResponseModel]:
        The structured response from the API.
    """
    # Use provided client or initialize a new one if not given
    claude_client = client if client else Anthropic(api_key=get_claude_api_key())
    logger.info("Claude client ready for API call.")
    return call_api(
        claude_client,
        model_id,
        prompt,
        expected_res_type,
        json_type,
        temperature,
        max_tokens,
        "claude",
    )


def call_llama3(
    prompt: str,
    model_id: str = "llama3",
    expected_res_type: str = "str",
    json_type: str = "",
    temperature: float = 0.4,
    max_tokens: int = 1056,
) -> Union[
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
    EditingResponse,
    JobSiteResponse,
]:
    """Calls the Llama 3 API and parses response."""
    return call_api(
        client=None,
        model_id=model_id,
        prompt=prompt,
        expected_res_type=expected_res_type,
        json_type=json_type,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_provider="llama3",
    )
