"""Async version of llm_api_utils_async.py

This module provides asynchronous utility functions for interacting with various LLM APIs, 
including OpenAI, Claude (Anthropic), and Llama3. It handles API calls, validates responses, 
and manages provider-specific nuances such as single-block versus multi-block responses.

Key Features:
- Asynchronous support for OpenAI and Claude APIs.
- Compatibility with synchronous Llama3 API via an async executor.
- Validation and structuring of responses into Pydantic models.
- Modular design to accommodate provider-specific response handling.

Modules and Methods:
- `call_openai_api_async`: Asynchronously interacts with the OpenAI API.
- `call_claude_api_async`: Asynchronously interacts with the Claude API.
- `call_llama3_async`: Asynchronously interacts with the Llama3 API using a synchronous executor.
- `call_api_async`: Unified async function for handling API calls with validation.
- `run_in_executor_async`: Executes synchronous functions in an async context.
- Validation utilities (e.g., `validate_response_type`, `validate_json_type`).

Usage:
This module is intended for applications that require efficient and modular integration 
with multiple LLM providers.
"""

# Built-in & External libraries
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Optional, cast
import json
import logging
from pydantic import ValidationError

# LLM imports
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import ollama  # ollama remains synchronous as there’s no async client yet

# From own modules
from models.llm_response_models import (
    CodeResponse,
    JSONResponse,
    TabularResponse,
    TextResponse,
    EditingResponse,
    JobSiteResponse,
)
from llm_providers.llm_api_utils import (
    get_claude_api_key,
    get_openai_api_key,
)
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

logger = logging.getLogger(__name__)


# Helper function to run synchronous API calls in an executor
async def run_in_executor_async(func, *args):
    """
    Runs a synchronous function in a ThreadPoolExecutor for async compatibility.

    Args:
        func (Callable): The synchronous function to execute.
        *args: Arguments to pass to the function.

    Returns:
        Any: The result of the synchronous function.
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, func, *args)


# Unified async API calling function
async def call_api_async(
    client: Optional[Union[AsyncOpenAI, AsyncAnthropic]],
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
]:
    """
    Asynchronous function for handling API calls to OpenAI, Claude, and Llama.

    This method handles provider-specific nuances (e.g., multi-block responses for Claude)
    and validates responses against expected types and Pydantic models.

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
    - OpenAI & Llama3 always returns single-block responses, while Claude may
    return multi-block responses, which needs special treatment.
    - Llama3 API is synchronous and is executed using an async executor.
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

        if llm_provider == "openai":
            openai_client = cast(AsyncOpenAI, client)
            response = await openai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_content = response.choices[0].message.content

        elif llm_provider == "claude":
            claude_client = cast(AsyncAnthropic, client)
            system_instruction = (
                "You are a helpful assistant who adheres to instructions."
            )
            response = await claude_client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": system_instruction + prompt}],
                temperature=temperature,
            )

            # *Add an extra step to extract content from response object's TextBlocks
            # *(Unlike GPT and LlaMA, Claude uses multi-blocks in its responses:
            # *The content attribute of Message is a list of TextBlock objects,
            # *whereas others wrap everything into a single block.)
            response_content = (
                response.content[0].text
                if hasattr(response.content[0], "text")
                else str(response.content[0])
            )

        elif llm_provider == "llama3":
            # Llama3 remains synchronous, so run it in an executor
            options = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "batch_size": 10,
                "retry_enabled": True,
            }
            response = await run_in_executor_async(
                ollama.generate, model_id, prompt, options
            )
            response_content = response["response"]

        logger.info(f"Raw {llm_provider} Response: {response_content}")

        # Validation 1: response content and return structured response
        validated_response_model = validate_response_type(
            response_content, expected_res_type
        )

        logger.info(
            f"validated response content after validate_response_type: \n{validated_response_model}"
        )  # TODO: debugging; delete afterwards

        # Validation 2: Further validate JSONResponse -> edit response or job site response models
        if expected_res_type == "json":
            if isinstance(validated_response_model, JSONResponse):
                # Pass directly to validate_json_type for further validation
                validated_response_model = validate_json_type(
                    response_model=validated_response_model, json_type=json_type
                )
            else:
                raise TypeError(
                    "Expected validated response content needs to be a JSONResponse model."
                )

        logger.info(
            f"validated response content after validate_json_type: \n{validated_response_model}"
        )  # TODO: debugging; delete afterwards

        return validated_response_model

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Validation or parsing error: {e}")
        raise ValueError(f"Invalid format received from {llm_provider} API: {e}")
    except Exception as e:
        logger.error(f"{llm_provider} API call failed: {e}")
        raise


# Async wrapper for OpenAI
async def call_openai_api_async(
    prompt: str,
    model_id: str = GPT_4_TURBO,
    expected_res_type: str = "str",
    json_type: str = "",
    temperature: float = 0.4,
    max_tokens: int = 1056,
    client: Optional[AsyncOpenAI] = None,
) -> Union[
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
    EditingResponse,
    JobSiteResponse,
]:
    """Asynchronously calls OpenAI API and parses the response."""
    openai_client = client or AsyncOpenAI(api_key=get_openai_api_key())
    logger.info("OpenAI client ready for async API call.")
    return await call_api_async(
        openai_client,
        model_id,
        prompt,
        expected_res_type,
        json_type,
        temperature,
        max_tokens,
        "openai",
    )


# Async wrapper for Claude
async def call_anthropic_api_async(
    prompt: str,
    model_id: str = CLAUDE_SONNET,
    expected_res_type: str = "str",
    json_type: str = "",
    temperature: float = 0.4,
    max_tokens: int = 1056,
    client: Optional[AsyncAnthropic] = None,
) -> Union[
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
    EditingResponse,
    JobSiteResponse,
]:
    """Asynchronously calls the Claude API to generate responses based on a given prompt."""
    anthropic_client = client or AsyncAnthropic(api_key=get_claude_api_key())
    logger.info("Claude client ready for async API call.")
    return await call_api_async(
        anthropic_client,
        model_id,
        prompt,
        expected_res_type,
        json_type,
        temperature,
        max_tokens,
        "claude",
    )


# Async wrapper for Llama 3
async def call_llama3_async(
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
    """Asynchronously calls the Llama 3 API and parses the response."""
    return await call_api_async(
        client=None,
        model_id=model_id,
        prompt=prompt,
        expected_res_type=expected_res_type,
        json_type=json_type,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_provider="llama3",
    )
