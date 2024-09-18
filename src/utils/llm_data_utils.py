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
from base_models import CodeResponse, JSONResponse, TabularResponse, TextResponse
import openai
import ollama

# Internal modules
from utils.webpage_reader import read_webpages
from prompts.prompt_templates import (
    CLEAN_JOB_PAGE_PROMPT,
    CONVERT_JOB_POSTING_TO_JSON_PROMPT,
)

# Import necessary modules for OpenAI and similarity scoring
# from openai_module import OpenAI, get_openai_api_key  # Ensure correct import paths
logger = logging.getLogger(__name__)


def call_openai_api(
    client, model_id, prompt, expected_res_type="str", temperature=0.4, max_tokens=1056
):
    """
    Handles API call to OpenAI to generate responses based on a given prompt and expected response type.

    Args:
        client: OpenAI API client instance.
        model_id (str): Model ID to use for the OpenAI API call.
        prompt (str): The prompt to send to the API.
        expected_response_type (str): The expected type of response from the API.
                                      Options are 'str' (default), 'json', 'tabular', or 'code'.
        max_tokens: default to 1056

    Returns:
        - Union[str, JSONResponse, pd.DataFrame, CodeResponse]:
        Response formatted according to the specified expected_response_type ()

        Union: A utility from Python's typing module
        str: plain text response
        JSONResponse: a custom Pydantic model for JSON response (e.g., inherited from pydantic.BaseModel).
        pd.DataFrame: pandas object; for tabular response
        CodeResponse: a custom Pydantic model for code response (e.g., inherited from pydantic.BaseModel).
    """
    try:
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
            raise Exception("Received an empty response from OpenAI API.")

        # Handle response based on expected type using Pydantic models
        if expected_res_type == "str":
            # Return plain string wrapped in a Pydantic model
            parsed_response = TextResponse(content=response_content)
            return parsed_response.content  # return as plain string instead the model

        elif expected_res_type == "json":
            # Parse JSON response
            try:
                # Convert JSON-formatted string to Python dictionary
                response_dict = json.loads(response_content)

                # Use Pydantic to validate and parse the dictionary into a model
                parsed_response = JSONResponse(**response_dict)

                return parsed_response
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse JSON or validate with Pydantic: {e}")
                raise ValueError("Invalid JSON format received from OpenAI API.")

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
        raise ValueError(f"Invalid format received from OpenAI API: {e}")
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise


def call_llama3(prompt: str, expected_res_type: str = "str", temperature: float = 0.4):
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
            },
        )

        # Extract the content from the response
        response_content = response["response"]

        # Check if the response is empty
        if not response_content:
            logger.error("Received an empty response from LLaMA 3.")
            raise Exception("Received an empty response from LLaMA 3.")

        # Handle response based on expected type using Pydantic models
        if expected_res_type == "str":
            # Return plain string wrapped in a Pydantic model
            parsed_response = TextResponse(content=response_content)
            return parsed_response.content  # return as plain string instead the model

        elif expected_res_type == "json":
            # Parse JSON response
            try:
                # Convert JSON-formatted string to Python dictionary
                response_dict = json.loads(response_content)

                # Use Pydantic to validate and parse the dictionary into a model
                parsed_response = JSONResponse(**response_dict)

                return parsed_response
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse JSON or validate with Pydantic: {e}")
                raise ValueError("Invalid JSON format received from OpenAI API.")

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
        raise ValueError(f"Invalid format received from OpenAI API: {e}")
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise


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


# Function to read webpage and clean with llama3
def read_and_clean_webpage_wt_llama3(url):
    """
    Reads a webpage, cleans the content using LLaMA for job related content only, and
    returns the cleaned content.

    Args:
        url (str): The URL of the webpage to read.

    Returns:
        str: The cleaned content with the URL included at the beginning.
    """
    page = read_webpages(urls=[url], single_page=True)
    page_content = page[url]
    logging.info("Page read.")

    # Initialize cleaned_chunks with the URL as the first element
    cleaned_chunks = [url]
    paragraphs = re.split(r"\n\s*\n", page_content)  # Split by paragraphs

    for paragraph in paragraphs:
        # If paragraph is too long, split into chunks of approximately 3000 characters
        if len(paragraph) > 3000:
            chunks = [paragraph[i : i + 3000] for i in range(0, len(paragraph), 3000)]
            for chunk in chunks:
                # Find the last sentence or paragraph break in the chunk
                last_break = max(chunk.rfind(". "), chunk.rfind("\n"))
                if last_break != -1:
                    chunk = chunk[
                        : last_break + 1
                    ]  # Split at the last sentence or paragraph break
                    prompt = CLEAN_JOB_PAGE_PROMPT.format(content=chunk)
                response = ollama.generate(model="llama3", prompt=prompt)
                cleaned_chunks.append(response["response"])
        else:
            # Format the prompt with the current paragraph content
            prompt = CLEAN_JOB_PAGE_PROMPT.format(content=paragraph)
            response = ollama.generate(model="llama3", prompt=prompt)
            cleaned_chunks.append(response["response"])

    cleaned_content = "\n".join(cleaned_chunks)
    cleaned_content = re.sub(r"\n\s*\n", "\n", cleaned_content)
    logging.info("Page cleaned.")
    return cleaned_content
