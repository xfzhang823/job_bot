"""
Filename: llm_api_utils.py
Last Updated: 2024-12-12

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
    OptimizedTextData,
    JobSiteData,
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


# Parsing & validation utils functions
# Response type validation
def validate_response_type(
    response_content: Union[str, Any], expected_res_type: str
) -> Union[
    CodeResponse,
    JSONResponse,
    TabularResponse,
    TextResponse,
]:
    """
    Validates and structures the response content based on
    the expected response type.

    Args:
        response_content (Any): The raw response content from the LLM API.
        expected_res_type (str): The expected type of the response
            (e.g., "json", "tabular", "str", "code").

    Returns:
        Union[CodeResponse, JSONResponse, TabularResponse, TextResponse]:
            The validated and structured response as a Pydantic model instance.
            - CodeResponse: Returned when expected_res_type is "code", wraps code content.
            - JSONResponse, JobSiteResponseModel, or EditingResponseModel:
              Returned when expected_res_type is "json", based on json_type.
            - TabularResponse: Returned when expected_res_type is "tabular", wraps a DataFrame.
            - TextResponse: Returned when expected_res_type is "str", wraps plain text content.
    """

    if expected_res_type == "json":
        # Check if response_content is a string that needs parsing
        if isinstance(response_content, str):
            # Only parse if it's a string
            cleaned_content = clean_and_extract_json(response_content)
            if cleaned_content is None:
                raise ValueError("Failed to extract valid JSON from the response.")
        else:
            # If it's already a dict or list, use it directly
            cleaned_content = response_content

        # Create a JSONResponse instance with the cleaned content
        if isinstance(cleaned_content, (dict, list)):
            return JSONResponse(data=cleaned_content)
        else:
            raise TypeError(
                f"Expected dict or list for JSON response, got {type(cleaned_content)}"
            )

    elif expected_res_type == "tabular":
        try:
            # Parse as DataFrame and wrap in TabularResponse model
            df = pd.read_csv(StringIO(response_content))
            return TabularResponse(data=df)
        except Exception as e:
            logger.error(f"Error parsing tabular data: {e}")
            raise ValueError("Response is not valid tabular data.")

    elif expected_res_type == "str":
        # Wrap text response in TextResponse model
        return TextResponse(content=response_content)

    elif expected_res_type == "code":
        # Wrap code response in CodeResponse model
        return CodeResponse(code=response_content)

    else:
        raise ValueError(f"Unsupported response type: {expected_res_type}")


def clean_and_extract_json(
    response_content: str,
) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Extracts, cleans, and parses JSON content from the API response.
    Strips out any non-JSON content like extra text before the JSON block.
    Also removes JavaScript-style comments and trailing commas.

    Args:
        response_content (str): Raw response content.

    Returns:
        Optional[Union[Dict[str, Any], List[Any]]]: Parsed JSON data as a dictionary or list,
        or None if parsing fails.
    """
    try:
        # Attempt direct parsing
        return json.loads(response_content)
    except json.JSONDecodeError:
        logger.warning("Initial JSON parsing failed. Attempting fallback extraction.")

    # Fallback: Extract first JSON-like structure
    match = re.search(r"{.*?}", response_content, re.DOTALL)
    if not match:
        logger.error("No JSON-like content found.")
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON in fallback: {e}")
        return None


def validate_editing_response(response_model: JSONResponse) -> EditingResponseModel:
    """
    Processes and validates a JSON response for the "editing" type.

    This function ensures that the "data" field in the response is an instance of OptimizedTextData.

    Args:
        response_model (JSONResponse): The generic JSON response model to validate.

    Returns:
        EditingResponseModel: The validated EditingResponseModel instance containing "optimized_text".

    Raises:
        ValueError: If the data is None or not an instance of OptimizedTextData.
    """
    response_data = response_model.data

    if response_data is None:
        raise ValueError("Response data is None and cannot be processed.")

    if not isinstance(response_data, OptimizedTextData):
        raise ValueError(
            f"Unexpected data type for editing response: {type(response_data)}"
        )

    validated_response_model = EditingResponseModel(data=response_data)

    logger.info(f"Validated response model - editing: {validated_response_model}")

    return validated_response_model


def validate_job_site_response(response_model: JSONResponse) -> JobSiteResponseModel:
    """
    Processes and validates a JSON response for the "job_site" type.

    This function ensures that the "data" field in the response is an instance of JobSiteData.

    Args:
        response_model (JSONResponse): The generic JSON response model to validate.

    Returns:
        JobSiteResponseModel: The validated JobSiteResponseModel instance containing job-specific data.

    Raises:
        ValueError: If the "data" field is None or not an instance of JobSiteData.
    """
    response_data_model = response_model.data  # parse -> data sub model

    if response_data_model is None:
        raise ValueError("Response data is None and cannot be processed.")

    if not isinstance(response_data_model, JobSiteData):
        raise ValueError(
            f"Unexpected data type for job site response: {type(response_data_model)}"
        )

    validated_response_model = JobSiteResponseModel(
        status="success",
        message="Job site data processed successfully.",
        data=response_data_model,
    )

    logger.info(f"Validated response model - job site: {validated_response_model}")

    return validated_response_model


def validate_json_type(
    response_model: JSONResponse, json_type: str
) -> Union[JobSiteResponseModel, EditingResponseModel, JSONResponse]:
    """
    Validates JSON data against a specific Pydantic model based on 'json_type'.

    Args:
        response_model (JSONResponse): The generic JSON response to validate.
        json_type (str): The expected JSON type ('job_site', 'editing', or 'generic').

    Returns:
        Union[JobSiteResponseModel, EditingResponseModel, JSONResponse]:
        Validated model instance.

    Raises:
        ValueError: If 'json_type' is unsupported or validation fails.
    """
    # Map json_type to the correct model class
    json_model_mapping = {
        "editing": validate_editing_response,
        "job_site": validate_job_site_response,
        "generic": lambda model: model,  # Return as is
    }

    # Pick the right function
    validator = json_model_mapping.get(json_type)
    if not validator:
        raise ValueError(f"Unsupported json_type: {json_type}")

    return validator(response_model)


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
    EditingResponseModel,
    JobSiteResponseModel,
]:
    """
    Unified function to handle API calls for OpenAI, Claude, and Llama.

    *Note:
    *The API calling methods need to remain separate because:
    *OpenAI returns single-block responses.
    *Anthropic (Claude) uses multi-block responses that needs special treatment.
    Combining them into a single code block will have complications;
    keep them separate here for each provider is a more clean and modular.
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
                response.content[0].text
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
        context_type,
        temperature,
        max_tokens,
        "openai",
    )


def call_claude_api(
    prompt: str,
    model_id: str = CLAUDE_SONNET,
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
        json_type=context_type,
        temperature=temperature,
        max_tokens=max_tokens,
        llm_provider="llama3",
    )
