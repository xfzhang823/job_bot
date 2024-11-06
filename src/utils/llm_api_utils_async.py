import asyncio
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
from utils.llm_api_utils import validate_json_type, validate_response_type

logger = logging.getLogger(__name__)


async def call_api(
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
    """Asynchronous function to handle API calls for OpenAI, Claude, and Llama."""

    try:
        logger.info(f"Making API call with expected response type: {expected_res_type}")
        response_content = ""

        # Step 1: Make the asynchronous API call and receive the response
        if llm_provider == "openai":
            openai_client = cast(AsyncOpenAI, client)  # Cast to AsyncOpenAI
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
            claude_client = cast(
                AsyncAnthropic, client
            )  # Cast to AsyncAnthropic (Claude)
            system_instruction = (
                "You are a helpful assistant who adheres to instructions."
            )
            response = await claude_client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": system_instruction + prompt}],
                temperature=temperature,
            )

            # Extract content from response object's TextBlocks (specific to Claude)
            response_content = (
                response.content[0].text
                if hasattr(response.content[0], "text")
                else str(response.content[0])
            )

        elif llm_provider == "llama3":
            # Llama3 remains synchronous (unless async support is added to Ollama)
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
        validated_response_content = validate_response_type(
            response_content, expected_res_type
        )

        # Further validate JSON responses
        if expected_res_type == "json":
            # Ensure validated_response_content is a JSONResponse before accessing `.data`
            if isinstance(validated_response_content, JSONResponse):
                # Unpack the `data` field, ensuring it's a dict or list of dicts as expected
                response_data = validated_response_content.data
                if not isinstance(response_data, (dict, list)):
                    raise ValueError(
                        "Expected response data to be a dictionary or a list of dictionaries."
                    )

                validated_response_content = validate_json_type(
                    response_data=response_data, json_type=json_type
                )
            else:
                raise TypeError(
                    "Expected validated response content to be a JSONResponse."
                )

        return validated_response_content

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Validation or parsing error: {e}")
        raise ValueError(f"Invalid format received from {llm_provider} API: {e}")
    except Exception as e:
        logger.error(f"{llm_provider} API call failed: {e}")
        raise
