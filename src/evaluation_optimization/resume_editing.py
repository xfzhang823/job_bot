"""
File: resume_editing.py
Author: Xiao-Fei Zhang
Last updated: 2024 Sep 12
"""

import logging
from dotenv import load_dotenv
import os
import json
import openai
from jsonschema import validate, ValidationError
from src.prompts.prompt_templates import EDIT_RESPONSIBILITY_PROMPT
from src.utils.generic_utils import get_openai_api_key

# Get openai api key
openai.api_key = get_openai_api_key()


def validate_response(response_dict):
    """
    Validate the JSON response against the expected schema.

    Args:
        response_dict (dict): The JSON response to validate.

    Raises:
        Exception: If validation fails.
    """
    # Define the expected JSON schema
    schema = {
        "type": "object",
        "properties": {"optimized_text": {"type": "string"}},
        "required": ["optimized_text"],
    }
    try:
        validate(instance=response_dict, schema=schema)
    except ValidationError as e:
        logging.error(f"JSON schema validation failed: {e}")
        raise Exception("JSON schema validation failed.")


# Function to convert text to JSON with OpenAI
def edit_resp_to_match_req_wt_gpt(resp_id, resp_sent, req_sent, model_id="gpt-4-turbo"):
    """
    Edit a bullet responsibility text from the resume to better match a requirement in
    a job description, leveraging an LLM response using the OpenAI API.

    Args:
        resp_id (str): Identifier for the responsibility bullet text.
        resp_sent (str): The candidate text to be optimized.
        req_sent (str): The reference text from the job description.
        model_id (str): OpenAI model to use (default is 'gpt-4').

        resp is short for responsibility
        req is short for (job) requirement

    Returns:
        dict: A dictionary containing 'resp_id' and 'optimized_text'.
    """

    # Define the JSON schema and instructions clearly in the prompt
    # Format the prompt with the provided texts
    try:
        prompt = EDIT_RESPONSIBILITY_PROMPT.format(
            content_1=resp_sent, content_2=req_sent
        )
    except KeyError as e:
        logging.error(f"Error formatting EDIT_RESPONSIBILITY_PROMPT: {e}")
        raise

    # Call the OpenAI API
    try:
        response = openai.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that strictly adheres to the provided instructions and returns responses only in the specified JSON format.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=200,  # Ensure sufficient tokens for response
        )
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        raise

    # Extract the content from the response
    edited_resp = response.choices[0].message.content

    # Check if the response is empty
    if not edited_resp:
        logging.error("Received an empty response from OpenAI API.")
        raise Exception("Received an empty response from OpenAI API.")

    # Debugging: Print the raw response to see what OpenAI returned
    logging.info(f"Raw LLM Response: {edited_resp}")

    # combine revised text with id, and return the dictionary
    try:
        # Parse the JSON response
        response_dict = json.loads(edited_resp)

        # Validate the JSON structure
        validate_response(response_dict)

        # Include resp_id in the result
        result = {"resp_id": resp_id}
        result.update(response_dict)
        return result

    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {e}")
        raise ValueError("JSON decoding failed. Please check the response format.")
    except ValidationError as e:
        logging.error(f"JSON schema validation failed: {e}")
        raise ValueError("JSON schema validation failed.")


# Function to modify resume w/t ChatGPT (unfinished...)
def modify_resume_section(section_json, requirements, model_id="gpt-3.5-turbo"):
    """
    Modifies a specific section of the resume to better align with job requirements.

    Args:
        section_json (dict): The JSON object containing the resume section details.
        requirements (dict): The extracted requirements from the job description.
        model_id (str): The model ID for OpenAI (default is gpt-3.5-turbo).

    Returns:
        dict: Modified resume section.
    """
    prompt = (
        f"Modify the following resume section in JSON format to better align with the job requirements. "
        f"Make it more concise and impactful while highlighting relevant skills and experiences:\n\n"
        f"Current Section JSON:\n{section_json}\n\n"
        f"Job Requirements JSON:\n{requirements}\n\n"
        "Return the modified section in JSON format."
    )

    response = openai.chat.completions.create(
        model=model_id, messages=[{"role": "user", "content": prompt}], temperature=0.7
    )

    return json.loads(response["choices"][0]["message"]["content"])
