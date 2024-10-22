"""Utils classes/methods for data extraction, parsing, and manipulation"""

# External libraries
import os
import json
import logging
import re
from io import StringIO
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel, ValidationError
from typing import Union
import openai
from openai import OpenAI
import ollama
from models.llm_response_models import (
    CodeResponse,
    JSONResponse,
    TabularResponse,
    TextResponse,
    EditingResponseModel,
    JobSiteResponseModel,
)
from prompts.prompt_templates import CONVERT_JOB_POSTING_TO_JSON_PROMPT

# Import necessary modules for OpenAI and similarity scoring
# from openai_module import OpenAI, get_openai_api_key  # Ensure correct import paths
logger = logging.getLogger(__name__)


def call_openai_api(
    prompt,
    client=None,
    model_id="gpt-4-turbo",
    expected_res_type="str",
    context_type: str = "",  # Use this to determine which JSON model to use
    temperature=0.4,
    max_tokens=1056,
) -> Union[
    TextResponse,
    JSONResponse,
    TabularResponse,
    CodeResponse,
    EditingResponseModel,
    JobSiteResponseModel,
]:
    """
    Handles API call to OpenAI to generate responses based on a given prompt and expected response type.

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
        - Union[str, JSONResponse, pd.DataFrame, CodeResponse]:
        Response formatted according to the specified expected_response_type ()

        - Union: A utility from Python's typing module
        - str: plain text response
        - JSONResponse: a custom Pydantic model for JSON response (e.g., inherited from pydantic.BaseModel).
        - pd.DataFrame: pandas object; for tabular response
        - CodeResponse: a custom Pydantic model for code response (e.g., inherited from pydantic.BaseModel).
    """
    if not client:
        openai_api_key = get_openai_api_key()
        client = OpenAI(api_key=openai_api_key)
        logger.info("OpenaAI API instantiated.")

    try:
        logger.info(f"Making API call with expected response type: {expected_res_type}")

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who strictly adheres to the provided instructions "
                    "and returns responses in the specified format.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,  # Ensure sufficient tokens for response
        )

        # Extract the content from the response
        response_content = response.choices[0].message.content.strip()
        logger.info(f"Raw LLM Response: {response_content}")

        # Check if the response is empty
        if not response_content:
            logger.error("Received an empty response from OpenAI API.")
            raise ValueError("Received an empty response from OpenAI API.")

        # Handle response based on expected type using Pydantic models
        if expected_res_type == "str":
            # Return plain string wrapped in a Pydantic model
            parsed_response = TextResponse(content=response_content)
            return parsed_response  # return as plain string instead the model

        elif expected_res_type == "json":
            try:
                cleaned_response_content = clean_and_extract_json(response_content)
                if not cleaned_response_content:
                    logger.error("Received an empty response after cleaning.")
                    raise ValueError("Received an empty response after cleaning.")

                response_dict = json.loads(cleaned_response_content)

                # Determine the correct model to use based on context_type
                if context_type == "editing":
                    return EditingResponseModel(
                        optimized_text=response_dict.get("optimized_text")
                    )

                elif context_type == "job_site":
                    return JobSiteResponseModel(
                        url=response_dict.get("url"),
                        job_title=response_dict.get("job_title"),
                        company=response_dict.get("company"),
                        location=response_dict.get("location"),
                        salary_info=response_dict.get("salary_info"),
                        posted_date=response_dict.get("posted_date"),
                        content=response_dict.get("content"),
                    )

                else:
                    # Fallback to a more generic JSON response if no specific context is provided
                    return JSONResponse(data=response_dict)

            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse JSON or validate with Pydantic: {e}")
                raise ValueError("Invalid JSON format received from OpenAI API.")

        elif expected_res_type == "tabular":
            try:
                # Parse tabular response using pandas
                # Assumes the response is in a CSV or Markdown table format
                df = pd.read_csv(StringIO(response_content))
                parsed_response = TabularResponse(data=df)
                return parsed_response
            except Exception as e:
                logger.error(f"Error parsing tabular data: {e}")
                raise ValueError("Invalid tabular format received from OpenAI API.")

        elif expected_res_type == "code":
            # Return the code as a Pydantic model
            parsed_response = CodeResponse(code=response_content)
            return parsed_response

        else:
            # Handle unsupported response types
            logger.error(f"Unsupported expected_response_type: {expected_res_type}")
            raise ValueError(f"Unsupported expected_response_type: {expected_res_type}")
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Validation or parsing error: {e}")
        raise ValueError(f"Invalid format received from OpenAI API: {e}")
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise


def call_llama3(
    prompt: str,
    expected_res_type: str = "str",
    temperature: float = 0.4,
    max_tokens: int = 1056,
):
    """
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
    try:
        # Generate response
        response = ollama.generate(
            model="llama3",
            prompt=prompt,
            options={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "batch_size": 10,
                "retry_enabled": True,
            },
        )

        # Extract the content from the response
        response_content = response["response"]

        # Log the raw response before attempting to parse it
        logger.info(f"Raw response content: {response_content}")

        # Check if the response is empty
        if not response_content:
            logger.error("Received an empty response from LLaMA 3.")
            raise ValueError("Received an empty response from LLaMA 3.")

        # Handle response based on expected type using Pydantic models
        if expected_res_type == "str":
            # Return plain string wrapped in a Pydantic model
            parsed_response = TextResponse(content=response_content)
            return parsed_response.content  # return as plain string instead the model

        elif expected_res_type == "json":
            # Extract JSON content using string manipulation
            start_idx = response_content.find("{")
            end_idx = response_content.rfind("}")
            if start_idx != -1 and end_idx != -1:
                cleaned_response_content = response_content[start_idx : end_idx + 1]
            else:
                logger.error("Failed to extract JSON content.")
                raise ValueError("Failed to extract JSON content.")

            # Validate extracted JSON
            try:
                response_dict = json.loads(cleaned_response_content)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                raise ValueError("Invalid JSON format received from LLaMA 3.")

            # Verify JSON structure using Pydantic model
            try:
                parsed_response = JSONResponse(**response_dict)
            except ValidationError as e:
                logger.error(f"JSON validation error: {e}")
                raise ValueError("Invalid JSON format received from LLaMA 3.")

            return parsed_response
        elif expected_res_type == "tabular":
            # Parse tabular response using pandas
            # Assumes the response is in a CSV or Markdown table format
            df = pd.read_csv(StringIO(response_content))
            parsed_response = TabularResponse(data=df)
            return parsed_response

        elif expected_res_type == "code":
            # Return the code as a Pydantic model
            parsed_response = CodeResponse(code=response_content)
            return parsed_response

        else:
            # Handle unsupported response types
            logger.error(f"Unsupported expected_response_type: {expected_res_type}")
            raise ValueError(f"Unsupported expected_response_type: {expected_res_type}")
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Validation or parsing error: {e}")
        raise ValueError(f"Invalid format received from LLaMA 3: {e}")
    except Exception as e:
        logger.error(f"LLaMA 3 call failed: {e}")
        raise


def clean_and_extract_json(response_content):
    """Ensure that a response does not have text other than json"""
    # Use regex to extract the JSON block from the response
    json_block = re.search(r"{.*}", response_content, re.DOTALL)

    if json_block:
        return json_block.group(0)
    else:
        raise ValueError("No valid JSON block found in response.")


# Function to convert text to JSON with OpenAI
def convert_to_json_wt_gpt(input_text, model_id="gpt-3.5-turbo", primary_key=None):
    """
    Extracts JSON content from an LLM response using OpenAI API.

    Args:
        input_text (str): The cleaned text to convert to JSON.
        model_id (str): The model ID to use for OpenAI (default is gpt-3.5-turbo).
        primary_key (str): The URL of the page that uniquely identifies each job posting.

    Returns:
        dict: The extracted JSON content as a dictionary.
    """

    # Load the API key from the environment
    get_openai_api_key()

    # Define the JSON schema and instructions clearly in the prompt
    prompt = CONVERT_JOB_POSTING_TO_JSON_PROMPT.format(content=input_text)

    # Call the OpenAI API
    response = openai.chat.completions.create(
        model=model_id, messages=[{"role": "user", "content": prompt}], temperature=0.7
    )

    # Extract the content from the response
    response_text = response.choices[0].message.content

    # Debugging: Print the raw response to see what OpenAI returned
    logging.info(f"Raw LLM Response: {response_text}")

    try:
        # Convert the extracted text to JSON
        job_posting_dict = json.loads(response_text)  # Proper JSON parsing

        # Nest job data under the URL as the primary key
        if primary_key:
            # Add the URL to the job posting data as a field for redundancy
            job_posting_dict["url"] = primary_key
            job_posting_dict = {primary_key: job_posting_dict}

        logging.info("JSON content successfully extracted and parsed.")
        return job_posting_dict
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        logging.error(f"JSON decoding failed: {e}")
        raise Exception("JSON decoding failed. Please check the response format.")
    except KeyError as e:
        logging.error(f"Missing key in response: {e}")
        raise Exception(
            "Error extracting JSON content. Please check the response format."
        )
    except ValueError as e:
        logging.error(f"Unexpected ValueError: {e}")
        raise Exception("Error occurred while processing the JSON content.")


# Load the API key from the environment securely
def get_openai_api_key():
    """Returns openai api key as str"""
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv("OPENAI_API_KEY")  # Get the API key from environment variables
    if api_key:
        logging.info("OpenAI API key successfully loaded.")
    else:
        logging.error("OpenAI API key not found. Please set it in the .env file.")
        raise EnvironmentError("OpenAI API key not found.")
    return api_key
