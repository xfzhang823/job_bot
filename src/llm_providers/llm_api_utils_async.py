"""Async version of llm_api_utils_async.py

This module provides asynchronous utility functions for interacting with various LLM APIs, 
including OpenAI, Anthropic, and Llama3. It handles API calls, validates responses, 
and manages provider-specific nuances such as single-block versus multi-block responses.

Key Features:
- Asynchronous support for OpenAI and Anthropic APIs.
- Compatibility with synchronous Llama3 API via an async executor.
- Validation and structuring of responses into Pydantic models.
- Modular design to accommodate provider-specific response handling.

Modules and Methods:
- `call_openai_api_async`: Asynchronously interacts with the OpenAI API.
- `call_anthropic_api_async`: Asynchronously interacts with the Anthropic API.
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
import time
import httpx

# LLM imports
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic, Anthropic
from anthropic._exceptions import RateLimitError
import ollama  # ollama remains synchronous as there’s no async client yet

# From own modules
from models.llm_response_models import (
    CodeResponse,
    JSONResponse,
    TabularResponse,
    TextResponse,
    EditingResponse,
    JobSiteResponse,
    RequirementsResponse,
)
from llm_providers.llm_api_utils import (
    get_anthropic_api_key,
    get_openai_api_key,
)
from llm_providers.llm_response_validators import (
    validate_json_type,
    validate_response_type,
)
from project_config import (
    OPENAI,
    ANTHROPIC,
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


# Global variables for rate limiting
last_request_time = 0
rate_limit_lock = asyncio.Lock()


async def with_rate_limit_and_retry(api_func):
    """
    Wrapper to apply rate limiting, retries, and exponential backoff for any API call.
    """
    global last_request_time

    min_interval = 0.5  # Ensure at least 500ms between requests
    max_retries = 5  # Allow up to 5 retries
    base_delay = 1  # Initial backoff delay (1 sec, 2 sec, 3 sec...)

    for attempt in range(max_retries):
        async with rate_limit_lock:  # Prevent multiple requests at the same time
            current_time = time.time()
            time_since_last = current_time - last_request_time

            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)

            try:
                result = await api_func()  # Execute API function
                last_request_time = time.time()  # Update only after success
                return result

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    retry_after = int(
                        e.response.headers.get("Retry-After", base_delay + attempt)
                    )
                    logger.warning(
                        f"Rate limit hit, retrying in {retry_after} seconds..."
                    )
                    await asyncio.sleep(min(10, retry_after))  # Limit max wait time
                    continue  # Retry after delay
                raise  # Stop retrying if max retries exceeded


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
    RequirementsResponse,
]:
    """
    Asynchronous function for handling API calls to OpenAI, Anthropic, and Llama.

    This method handles provider-specific nuances (e.g., multi-block responses for Anthropic)
    and validates responses against expected types and Pydantic models.

    Also inclues automatic rate limiting and retry logic to handle API restrictions.

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
            The name of the LLM provider ("openai", "anthropic", or "llama3").

    Returns:
        Union[JSONResponse, TabularResponse, CodeResponse, TextResponse, EditingResponseModel,
        JobSiteResponseModel]:
            The validated and structured response.

    Raises:
        ValueError: If the response cannot be validated or parsed.
        TypeError: If the response type does not match the expected format.
        Exception: For other unexpected errors during API interaction.

    Notes:
    - OpenAI & Llama3 always returns single-block responses, while anthropic may
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

        if llm_provider.lower() == "openai":
            openai_client = cast(AsyncOpenAI, client)

            async def openai_request():
                return await openai_client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            response = await with_rate_limit_and_retry(openai_request)

            # Handle cases where response or choices is None
            if not response or not response.choices:
                raise ValueError("OpenAI API returned an invalid or empty response.")

            response_content = response.choices[0].message.content

        elif llm_provider.lower() == "anthropic":
            anthropic_client = cast(AsyncAnthropic, client)
            system_instruction = (
                "You are a helpful assistant who adheres to instructions."
            )

            async def anthropic_request():
                return await anthropic_client.messages.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": system_instruction + prompt}],
                    temperature=temperature,
                )

            response = await with_rate_limit_and_retry(anthropic_request)

            # *Add an extra step to extract content from response object's TextBlocks
            # *(Unlike GPT and LlaMA, Anthropic uses multi-blocks in its responses:
            # *The content attribute of Message is a list of TextBlock objects,
            # *whereas others wrap everything into a single block.)

            # Validate response structure
            if not response or not response.content:
                raise ValueError("Empty response received from Anthropic API")

            # Safely access the first content block
            first_block = response.content[0]
            response_content = (
                first_block.text if hasattr(first_block, "text") else str(first_block)
            )

            if not response_content:
                raise ValueError("Empty content in response from Anthropic API")

            # response_content = (
            #     response.content[0].text
            #     if hasattr(response.content[0], "text")
            #     else str(response.content[0])
            # )

        elif llm_provider.lower() == "llama3":
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
    RequirementsResponse,
]:
    """Asynchronously calls OpenAI API and parses the response."""
    openai_client = client or AsyncOpenAI(api_key=get_openai_api_key())
    logger.info("OpenAI client ready for async API call.")
    return await call_api_async(
        client=openai_client,
        model_id=model_id,
        prompt=prompt,
        expected_res_type=expected_res_type,
        json_type=json_type,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_provider=OPENAI,
    )


# Async wrapper for Anthropic
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
    RequirementsResponse,
]:
    """Asynchronously calls the Anthropic API to generate responses based on a given prompt."""
    anthropic_client = client or AsyncAnthropic(api_key=get_anthropic_api_key())
    logger.info("Anthropic client ready for async API call.")
    return await call_api_async(
        client=anthropic_client,
        model_id=model_id,
        prompt=prompt,
        expected_res_type=expected_res_type,
        json_type=json_type,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_provider=ANTHROPIC,
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
