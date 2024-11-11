import os
import json
import logging
import re
from io import StringIO
from dotenv import load_dotenv
import pandas as pd
from pydantic import ValidationError
from typing import Union, Optional, Mapping, Any, Dict, List, cast
import asyncio
from concurrent.futures import ThreadPoolExecutor

from openai import AsyncOpenAI
import ollama
from anthropic import AsyncAnthropic

from models.llm_response_models import (
    CodeResponse,
    JSONResponse,
    TabularResponse,
    TextResponse,
    EditingResponseModel,
    JobSiteResponseModel,
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


# Utility Functions
async def clean_and_extract_json(
    response_content: str,
) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Extracts, cleans, and parses JSON content from the API response.
    Strips out any non-JSON content like extra text before the JSON block.
    Also removes JavaScript-style comments and trailing commas.

    Args:
        response_content (str): The full response content as a string, potentially containing JSON.

    Returns:
        Optional[Union[Dict[str, Any], List[Any]]]: Parsed JSON data as a dictionary or list,
        or None if parsing fails.
    """
    try:
        match = re.search(r"{.*}", response_content, re.DOTALL)
        if not match:
            logger.error("No valid JSON content found in response.")
            return None

        raw_json_string = match.group(0)
        cleaned_json_string = re.sub(r"\s*//[^\n]*", "", raw_json_string)
        cleaned_json_string = re.sub(r",\s*([\]}])", r"\1", cleaned_json_string)
        return json.loads(cleaned_json_string)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return None


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


# Validation Functions
def validate_response_type(
    response_content: str, expected_res_type: str, json_type: str
) -> Union[
    CodeResponse,
    JSONResponse,
    TabularResponse,
    TextResponse,
    EditingResponseModel,
    JobSiteResponseModel,
]:
    """
    Validates and structures the response content based on the expected response type.

    Args:
        response_content (str): The raw response content from the LLM API.
        expected_res_type (str): The expected type of the response
            (e.g., "json", "tabular", "str", "code").
        json_type (str): The context in which to interpret the JSON response
            (e.g., "job_site" or "editing").

    Returns:
        Union[CodeResponse, JSONResponse, TabularResponse, TextResponse,
              EditingResponseModel, JobSiteResponseModel]: The validated and
              structured response as a Pydantic model instance.

            - CodeResponse: Returned when expected_res_type is "code", wraps code content.
            - JSONResponse, JobSiteResponseModel, or EditingResponseModel:
              Returned when expected_res_type is "json", based on json_type.
            - TabularResponse: Returned when expected_res_type is "tabular", wraps a DataFrame.
            - TextResponse: Returned when expected_res_type is "str", wraps plain text content.
    """

    if expected_res_type == "json":
        cleaned_content = await clean_and_extract_json(response_content)
        if cleaned_content is None:
            raise ValueError("No valid JSON content found.")
        return await validate_json_type(cleaned_content, json_type)

    elif expected_res_type == "tabular":
        try:
            df = pd.read_csv(StringIO(response_content))
            return TabularResponse(data=df)
        except Exception as e:
            logger.error(f"Error parsing tabular data: {e}")
            raise ValueError("Response is not valid tabular data.")

    elif expected_res_type == "str":
        return TextResponse(content=response_content)

    elif expected_res_type == "code":
        return CodeResponse(code=response_content)

    else:
        raise ValueError(f"Unsupported response type: {expected_res_type}")


def validate_json_type(
    response_data: Union[Dict[str, Any], List[Any]], json_type: str
) -> Union[JSONResponse, JobSiteResponseModel, EditingResponseModel]:
    """
    Validates and structures the JSON response based on the json_type.

    Args:
        - response_data (dict): The JSON data from the response, already
        validated as a dictionary.
        - json_type (str): The context in which to interpret the response
        (e.g., "job_site" or "editing").

    Returns:
        Union[JSONResponse, JobSiteResponseModel, EditingResponseModel]:
        The structured and validated response as a pydantic model instance.
    """
    if json_type == "job_site":
        if isinstance(response_data, dict):
            return JobSiteResponseModel(
                url=response_data.get("url"),
                job_title=response_data.get("job_title"),
                company=response_data.get("company"),
                location=response_data.get("location"),
                salary_info=response_data.get("salary_info"),
                posted_date=response_data.get("posted_date"),
                content=response_data.get("content"),
            )
        else:
            raise ValueError("Expected a dictionary for job_site context.")

    elif json_type == "editing":
        if isinstance(response_data, dict):
            return EditingResponseModel(
                optimized_text=response_data.get("optimized_text")
            )
        else:
            raise ValueError("Expected a dictionary for editing context.")

    if isinstance(response_data, list):
        return JSONResponse(data={"data": response_data})
    else:
        return JSONResponse(data=response_data)


# Helper function to run synchronous API calls in an executor
async def run_in_executor_async(func, *args):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, func, *args)


# API Calling Functions
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
    """Unified function to handle API calls for OpenAI, Claude, and Llama."""

    try:
        logger.info(f"Making API call with expected response type: {expected_res_type}")

        if llm_provider == "openai":
            openai_client = cast(AsyncOpenAI, client)
            response = await run_in_executor_async(
                openai_client.chat.completions.create,
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
            response = await run_in_executor_async(
                claude_client.messages.create,
                model=model_id,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": "You are a helpful assistant." + prompt}
                ],
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
            response = await run_in_executor_async(
                ollama.generate, model_id, prompt, options
            )
            response_content = response["response"]

        logger.info(f"Raw {llm_provider} Response: {response_content}")

        return await validate_response_type(
            response_content, expected_res_type, json_type
        )

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Validation or parsing error: {e}")
        raise ValueError(f"Invalid format received from {llm_provider} API: {e}")
    except Exception as e:
        logger.error(f"{llm_provider} API call failed: {e}")
        raise


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
    """
    Calls OpenAI API and parses response.

    Args:
        prompt (str): The prompt to send to the API.
        client (Optional[OpenAI]): An OpenAI client instance. If None, a new client is instantiated.
        model_id (str): Model ID to use for the API call.
        expected_res_type (str): The expected type of response from the API ('str', 'json',
        'tabular', or 'code').
        json_type (str): Context type for JSON responses ("editing" or "job_site").
        temperature (float): Controls the creativity of the response.
        max_tokens (int): Maximum number of tokens for the response.

    Returns:
        Union[TextResponse, JSONResponse, TabularResponse, CodeResponse, EditingResponseModel,
        JobSiteResponseModel]: The structured response from the API.
    """
    openai_client = client if client else AsyncOpenAI(api_key=get_openai_api_key())
    logger.info("OpenAI client ready for API call.")

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
    """
    Calls the Claude API to generate responses based on a given prompt and expected response type.

    Args:
        prompt (str): The prompt to send to the API.
        - model_id (str): Model ID to use for the Claude API call.
        - expected_res_type (str): The expected type of response from the API
        ('str', 'json', 'tabular', or 'code').
        - json_type (str): Context type for JSON responses ("editing" or "job_site").
        - temperature (float): Controls the creativity of the response.
        - max_tokens (int): Maximum number of tokens for the response.
        - client (Optional[Anthropic]): A Claude client instance.
        If None, a new client is instantiated.

    Returns:
        Union[TextResponse, JSONResponse, TabularResponse, CodeResponse, EditingResponseModel, JobSiteResponseModel]:
        The structured response from the API.
    """
    claude_client = client if client else AsyncAnthropic(api_key=get_claude_api_key())
    logger.info("Claude client ready for API call.")
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
    """Calls the Llama 3 API and parses response."""
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
