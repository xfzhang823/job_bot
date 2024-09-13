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
from prompts.prompts import EDIT_RESPONSIBILITY_PROMPT

# Load the API key from the environment at the module level
load_dotenv()  # Load environment variables from .env file
openai.api_key = os.getenv("OPENAI_API_KEY")


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

    Returns:
        dict: A dictionary containing 'resp_id' and 'optimized_text'.
    """

    # Define the JSON schema and instructions clearly in the prompt
    prompt = EDIT_RESPONSIBILITY_PROMPT.format(conten_1=resp_sent, content_2=req_sent)

    # Call the OpenAI API
    response = openai.chat.completions.create(
        model=model_id, messages=[{"role": "user", "content": prompt}], temperature=0.6
    )

    # Extract the content from the response
    edited_resp = response.choices[0].message.content

    # Debugging: Print the raw response to see what OpenAI returned
    logging.info(f"Raw LLM Response: {edited_resp}")

    try:
        # Parse the JSON response
        response_dict = json.loads(edited_resp)

        # Include resp_id in the result
        result = {"resp_id": resp_id}
        result.update(response_dict)
        return result

    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {e}")
        raise Exception("JSON decoding failed. Please check the response format.")
