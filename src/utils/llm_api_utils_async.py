import os
from dotenv import load_dotenv


import asyncio
import logging
import logging_config
import httpx
import aiohttp
import json
from pydantic import ValidationError
import pandas as pd
from io import StringIO
from typing import Union, Optional
from cast import cast

from anthropic import Anthropic
import openai
from openai import OpenAI, AsyncOpenAI
import ollama

from utils.llm_api_utils import (
    get_openai_api_key,
    get_claude_api_key,
    clean_and_extract_json,
    validate_context_type,
    validate_response_type,
)
from models.llm_response_models import (
    CodeResponse,
    JSONResponse,
    TabularResponse,
    TextResponse,
    EditingResponseModel,
    JobSiteResponseModel,
)

# Setup logger
logger = logging.getLogger(__name__)


async def call_api_async(
    client: Optional[Union[OpenAI, Anthropic]],
    model_id: str,
    prompt: str,
    expected_res_type: str,
    context_type: str,
    temperature: float,
    max_tokens: int,
    llm_provider: str,
) -> Union[
    JSONResponse,
    TabularResponse,
    CodeResponse,
    TextResponse,
    EditingResponseModel,
    JobSiteResponseModel,
]:
    """Unified function to handle async API calls for OpenAI, Claude, and Llama."""
    try:
        logger.info(
            f"Making async API call with expected response type: {expected_res_type}"
        )
        if llm_provider == "openai":
            openai_client = cast(OpenAI, client)  # Cast to OpenAI
            response = await openai_client.chat.completions.create(
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
            response = await claude_client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": system_instruction + prompt}],
                temperature=temperature,
            )
            response_content = response.content[0]

        elif llm_provider == "llama3":
            options = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "batch_size": 10,
                "retry_enabled": True,
            }
            response = await ollama.generate(model=model_id, prompt=prompt, options=options)  # type: ignore
            response_content = response["response"]

        logger.info(f"Raw {llm_provider} Response: {response_content}")

        if not response_content:
            raise ValueError(f"Received an empty response from {llm_provider} API.")

        validated_response_content = validate_response_type(
            response_content, expected_res_type
        )

        # Convert to model based on response type and context
        if expected_res_type == "json" and isinstance(validated_response_content, dict):
            return validate_context_type(validated_response_content, context_type)
        elif expected_res_type == "tabular" and isinstance(
            validated_response_content, pd.DataFrame
        ):
            return TabularResponse(data=validated_response_content)
        elif expected_res_type == "code" and isinstance(
            validated_response_content, str
        ):
            return CodeResponse(code=validated_response_content)
        elif expected_res_type == "str" and isinstance(validated_response_content, str):
            return TextResponse(content=validated_response_content)

        raise ValueError(f"Unsupported response type: {expected_res_type}")

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Validation or parsing error: {e}")
        raise ValueError(f"Invalid format received from {llm_provider} API: {e}")
    except Exception as e:
        logger.error(f"{llm_provider} API call failed: {e}")
        raise


# async openai function
async def call_openai_api_async(
    prompt: str,
    model_id: str = "gpt-4-turbo",
    expected_res_type: str = "str",
    context_type: str = "",
    temperature: float = 0.4,
    max_tokens: int = 1056,
    client: Optional[OpenAI] = None,
) -> Union[
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
    EditingResponseModel,
    JobSiteResponseModel,
]:
    """
    *async version

    Handles async API call to OpenAI to generate responses based on a given prompt
    and expected response type.

    Args:
        - client: OpenAI API client instance (optional)
        - model_id (str): Model ID to use for the OpenAI API call.
        - prompt (str): The prompt to send to the API.
        - expected_res_type (str): The expected type of response from the API
        ('str', 'json', 'tabular', or 'code').
        - context_type (str): Specifies whether to use a job-related JSON model or
        an editing model for JSON ("editing" or "job_site")
        - temperature (str): creativity of the response (0 to 1.0)
        - max_tokens: Maximum tokens to be generated in response.

    Returns:
        Union[str, dict, pd.DataFrame]: Response formatted according to the expected_res_type.
    """
    openai_client = client if client else OpenAI(api_key=get_openai_api_key())
    logger.info("OpenAI client ready for async API call.")
    return await call_api_async(
        openai_client,
        model_id,
        prompt,
        expected_res_type,
        context_type,
        temperature,
        max_tokens,
        "openai",
    )


# async claude function
async def call_claude_api_async(
    prompt: str,
    model_id: str = "claude-3-5-sonnet-20241022",
    expected_res_type: str = "str",
    context_type: str = "",
    temperature: float = 0.4,
    max_tokens: int = 1056,
    client: Optional[Anthropic] = None,
) -> Union[
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
    EditingResponseModel,
    JobSiteResponseModel,
]:
    """Asynchronously calls the Claude API to generate responses based on a given prompt."""
    claude_client = client if client else Anthropic(api_key=get_claude_api_key())
    logger.info("Claude client ready for async API call.")
    return await call_api_async(
        claude_client,
        model_id,
        prompt,
        expected_res_type,
        context_type,
        temperature,
        max_tokens,
        "claude",
    )


# async llama3 function
async def call_llama3_async(
    prompt: str,
    model_id: str = "llama3",
    expected_res_type: str = "str",
    context_type: str = "",
    temperature: float = 0.4,
    max_tokens: int = 1056,
) -> Union[
    JSONResponse,
    TabularResponse,
    TextResponse,
    CodeResponse,
    EditingResponseModel,
    JobSiteResponseModel,
]:
    """
    *Async version of the method - still working on... not done yet!

    Handles calls to LLaMA 3 (on-premise) to generate responses based on
    a given prompt and expected response type.

    Args:
        - model_id (str): Model ID to use for the LLaMA 3 API call.
        - prompt (str): The prompt to send to the API.
        - expected_response_type (str):
            * The expected type of response from the API.
            * Options are 'str' (default), 'json', 'tabular', or 'code'.

    Returns:
        - Union[str, JSONResponse, pd.DataFrame, CodeResponse]:
        Response formatted according to the specified expected_response_type
    """
    return await call_api_async(
        client=None,
        model_id=model_id,
        prompt=prompt,
        expected_res_type=expected_res_type,
        context_type=context_type,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_provider="llama3",
    )
