"""Async version of llm_api_utils_async.py"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from typing import Union, Optional, Any, cast
import json
import logging
from pydantic import ValidationError
from io import StringIO
import pandas as pd

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import ollama  # ollama remains synchronous as there’s no async client yet

from models.llm_response_models import (
    CodeResponse,
    JSONResponse,
    TabularResponse,
    TextResponse,
    EditingResponseModel,
    JobSiteResponseModel,
)
from utils.llm_api_utils import (
    validate_json_type,
    validate_response_type,
    get_claude_api_key,
    get_openai_api_key,
)
from config import (
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
    EditingResponseModel,
    JobSiteResponseModel,
]:
    """Unified asynchronous function to handle API calls for OpenAI, Claude, and Llama."""

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
            response_content = (
                response.content[0].text
                if hasattr(response.content[0], "text")
                else str(response.content[0])
            )
            # logger.info(f"claude api response raw output: {response_content}")

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

        # *Validation 1: response content and return structured response
        validated_response_content = validate_response_type(
            response_content, expected_res_type
        )

        logger.info(
            f"validated response content after validate_response_type: \n{validated_response_content}"
        )  # TODO: debugging; delete afterwards

        # *Validation 2: Validate response content and return structured response
        if expected_res_type == "json":
            if isinstance(validated_response_content, JSONResponse):
                # response_data = validated_response_content.data

                # logger.info(
                #     f"response_data: {response_data}"
                # )  # TODO: debugging; delete afterwards

                # if not isinstance(response_data, (dict, list)):
                #     raise ValueError(
                #         "Expected response data to be a dictionary or list."
                #     )
                validated_response_content = validate_json_type(
                    response_model=validated_response_content, json_type=json_type
                )

        logger.info(
            f"validated response content after validate_json_type: \n{validated_response_content}"
        )  # TODO: debugging; delete afterwards

        return validated_response_content

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
    EditingResponseModel,
    JobSiteResponseModel,
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
async def call_claude_api_async(
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
    EditingResponseModel,
    JobSiteResponseModel,
]:
    """Asynchronously calls the Claude API to generate responses based on a given prompt."""
    claude_client = client or AsyncAnthropic(api_key=get_claude_api_key())
    logger.info("Claude client ready for async API call.")
    return await call_api_async(
        claude_client,
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
    EditingResponseModel,
    JobSiteResponseModel,
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
