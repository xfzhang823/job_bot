"""
Filename: llm_api_utils.py
Last Updated: 2024-10-28

Utils classes/methods for data extraction, parsing, and manipulation
"""

# External libraries
import os
import json
import logging
import re
from io import StringIO
from dotenv import load_dotenv
import pandas as pd
from pydantic import ValidationError
from typing import Union, Optional, Mapping, Any, Dict, List, cast

from openai import OpenAI
import ollama
from anthropic import Anthropic

from models.llm_response_models import (
    CodeResponse,
    JSONResponse,
    TabularResponse,
    TextResponse,
    EditingResponseModel,
    JobSiteResponseModel,
)
from models.openai_claude_llama_response_basemodels import (
    OpenAITextResponse,
    OpenAIJSONResponse,
    OpenAITabularResponse,
    OpenAICodeResponse,
    ClaudeTextResponse,
    ClaudeJSONResponse,
    ClaudeTabularResponse,
    ClaudeCodeResponse,
    LlamaTextResponse,
    LlamaJSONResponse,
    LlamaTabularResponse,
    LlamaCodeResponse,
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
# def clean_and_extract_json(response_content: str) -> str:
#     """Ensures a response contains only valid JSON by extracting the JSON block."""
#     json_block = re.search(r"{.*}", response_content, re.DOTALL)
#     if json_block:
#         return json_block.group(0)
#     else:
#         raise ValueError("No valid JSON block found in response.")


def clean_and_extract_json(
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
        # Find the first occurrence of the `{` and last occurrence of `}` to extract the JSON block
        start_idx = response_content.find("{")
        end_idx = response_content.rfind("}") + 1

        # Ensure that valid JSON content is found
        if start_idx == -1 or end_idx == -1:
            logger.error("No valid JSON content found in response.")
            return None

        # Extract the JSON part of the response
        raw_json_string = response_content[start_idx:end_idx]

        # Remove single-line comments (// ...) but don't affect valid JSON
        cleaned_json_string = re.sub(r"\s*//[^\n]*", "", raw_json_string)

        # Remove trailing commas before closing curly braces or square brackets
        cleaned_json_string = re.sub(r",\s*([\]}])", r"\1", cleaned_json_string)

        # Parse JSON string into a Python object
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


# Parsing Functions
def parse_response(
    provider: str, response_type: str, response: Mapping[str, Any]
) -> Union[
    OpenAITextResponse,
    OpenAIJSONResponse,
    OpenAITabularResponse,
    OpenAICodeResponse,
    ClaudeTextResponse,
    ClaudeJSONResponse,
    ClaudeTabularResponse,
    ClaudeCodeResponse,
    LlamaTextResponse,
    LlamaJSONResponse,
    LlamaTabularResponse,
    LlamaCodeResponse,
]:
    """
    Parses raw JSON response from an LLM API based on provider and response type.

    Args:
        provider (str): The LLM provider ("openai", "claude", "llama").
        response_type (str): The type of response expected ("text", "json", "tabular", "code").
        response (Mapping[str, Any]): The raw response dictionary from the LLM API.

    Returns:
        Union[OpenAITextResponse, OpenAIJSONResponse, ...]: Parsed response model specific to the provider and response type.
    """
    parsers = {
        "openai": {
            "text": lambda r: OpenAITextResponse(text=r.choices[0].message.content),
            "json": lambda r: OpenAIJSONResponse(data=r),
            "tabular": lambda r: OpenAITabularResponse(rows=r),
            "code": lambda r: OpenAICodeResponse(
                code=r.get("code"),
                language=r.get("language"),
                explanation=r.get("explanation"),
            ),
        },
        "claude": {
            "text": lambda r: ClaudeTextResponse(content=r["completion"]),
            "json": lambda r: ClaudeJSONResponse(data=r),
            "tabular": lambda r: ClaudeTabularResponse(rows=r),
            "code": lambda r: ClaudeCodeResponse(
                code=r.get("code"),
                language=r.get("language"),
                explanation=r.get("explanation"),
            ),
        },
        "llama": {
            "text": lambda r: LlamaTextResponse(result=r["result"]),
            "json": lambda r: LlamaJSONResponse(data=r),
            "tabular": lambda r: LlamaTabularResponse(rows=r),
            "code": lambda r: LlamaCodeResponse(
                code=r.get("code"),
                language=r.get("language"),
                explanation=r.get("explanation"),
            ),
        },
    }
    try:
        return parsers[provider][response_type](response)
    except KeyError:
        raise ValueError(
            f"Unsupported provider or response type: {provider}, {response_type}"
        )


# Validation Functions
def validate_response_type(
    response_content: Union[str, Any], expected_res_type: str
) -> Union[str, Dict[str, Any], pd.DataFrame]:
    """
    Validates the response type based on the expected_res_type.

    Args:
        response_content (str): The raw response content from the LLM API.
        expected_res_type (str): The expected type of the response (e.g., "str", "json", "tabular", "code").

    Returns:
        Union[str, dict, pd.DataFrame]: The validated response content in the appropriate format.
    """
    if expected_res_type == "json":
        try:
            cleaned_content = clean_and_extract_json(response_content)
            return json.loads(cleaned_content)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid JSON format: {e}")
            raise ValueError("Response is not valid JSON.")
    elif expected_res_type == "tabular":
        try:
            return pd.read_csv(StringIO(response_content))
        except Exception as e:
            logger.error(f"Error parsing tabular data: {e}")
            raise ValueError("Response is not valid tabular data.")
    elif expected_res_type in {"str", "code"}:
        return response_content
    else:
        raise ValueError(f"Unsupported response type: {expected_res_type}")


def validate_context_type(
    response_data: Dict[str, Any], context_type: str
) -> Union[JSONResponse, JobSiteResponseModel, EditingResponseModel]:
    """
    Validates and structures the JSON response based on the context_type.

    Args:
        - response_data (dict): The JSON data from the response, already
        validated as a dictionary.
        - context_type (str): The context in which to interpret the response
        (e.g., "job_site" or "editing").

    Returns:
        Union[JSONResponse, JobSiteResponseModel, EditingResponseModel]:
        The structured and validated response as a pydantic model instance.
    """
    if context_type == "job_site":
        return JobSiteResponseModel(
            url=response_data.get("url"),
            job_title=response_data.get("job_title"),
            company=response_data.get("company"),
            location=response_data.get("location"),
            salary_info=response_data.get("salary_info"),
            posted_date=response_data.get("posted_date"),
            content=response_data.get("content"),
        )
    elif context_type == "editing":
        return EditingResponseModel(optimized_text=response_data.get("optimized_text"))
    else:
        return JSONResponse(data=response_data)  # Default JSON structure


# API Calling Functions
def call_api(
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
    """Unified function to handle API calls for OpenAI, Claude, and Llama."""
    try:
        logger.info(f"Making API call with expected response type: {expected_res_type}")
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
            response_content = response.content[0]

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

        if not response_content:
            raise ValueError(f"Received an empty response from {llm_provider} API.")

        # Step 1: Validate response type
        validated_response_content = validate_response_type(
            response_content, expected_res_type
        )

        # Step 2: Convert to model based on response type and context
        if expected_res_type == "json" and isinstance(validated_response_content, dict):
            # For JSON, use context-specific model
            return validate_context_type(validated_response_content, context_type)
        elif expected_res_type == "tabular" and isinstance(
            validated_response_content, pd.DataFrame
        ):
            return TabularResponse(
                data=validated_response_content
            )  # Wrap in TabularResponse
        elif expected_res_type == "code":
            # Ensure validated_response_content is a str for CodeResponse
            if isinstance(validated_response_content, str):
                return CodeResponse(
                    code=validated_response_content
                )  # Wrap in CodeResponse
            else:
                raise TypeError(
                    f"Expected a str for code response, but got {type(validated_response_content).__name__}"
                )
        elif expected_res_type == "str":
            # Ensure validated_response_content is a str for TextResponse
            if isinstance(validated_response_content, str):
                return TextResponse(
                    content=validated_response_content
                )  # Wrap in TextResponse
            else:
                raise TypeError(
                    f"Expected a str for text response, but got {type(validated_response_content).__name__}"
                )

        # Raise an error if expected_res_type does not match any known types
        raise ValueError(f"Unsupported response type: {expected_res_type}")

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Validation or parsing error: {e}")
        raise ValueError(f"Invalid format received from {llm_provider} API: {e}")
    except Exception as e:
        logger.error(f"{llm_provider} API call failed: {e}")
        raise


def call_openai_api(
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
    Calls OpenAI API and parses response.

    Args:
        prompt (str): The prompt to send to the API.
        client (Optional[OpenAI]): An OpenAI client instance. If None, a new client is instantiated.
        model_id (str): Model ID to use for the API call.
        expected_res_type (str): The expected type of response from the API ('str', 'json',
        'tabular', or 'code').
        context_type (str): Context type for JSON responses ("editing" or "job_site").
        temperature (float): Controls the creativity of the response.
        max_tokens (int): Maximum number of tokens for the response.

    Returns:
        Union[TextResponse, JSONResponse, TabularResponse, CodeResponse, EditingResponseModel,
        JobSiteResponseModel]: The structured response from the API.
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
        context_type,
        temperature,
        max_tokens,
        "openai",
    )


def call_claude_api(
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
        context_type,
        temperature,
        max_tokens,
        "claude",
    )


def call_llama3(
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
    """Calls the Llama 3 API and parses response."""
    return call_api(
        client=None,
        model_id=model_id,
        prompt=prompt,
        expected_res_type=expected_res_type,
        context_type=context_type,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_provider="llama3",
    )
