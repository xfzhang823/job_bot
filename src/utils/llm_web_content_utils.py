"""
Helper functions to call on LLMs to process webpage related content.
"""

# External libraries
import os
import json
import logging
import logging_config
import re
from io import StringIO
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel, ValidationError
from typing import Union
import openai
from openai import OpenAI
import ollama
from anthropic import Anthropic

# Internal dependencies
from src.utils.llm_api_utils import get_openai_api_key, get_claude_api_key
from prompts.prompt_templates import CONVERT_JOB_POSTING_TO_JSON_PROMPT


# Set logger
logger = logging.getLogger(__name__)


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
