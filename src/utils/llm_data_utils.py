"""Utils classes/methods for data extraction, parsing, and manipulation"""

import os
import json
import logging
import re
from dotenv import load_dotenv
import ollama
import openai
from webpage_reader import read_webpages
from prompts.prompt_templates import (
    CLEAN_JOB_PAGE_PROMPT,
    CONVERT_JOB_POSTING_TO_JSON_PROMPT,
)


# Load the API key from the environment securely
def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logging.info("OpenAI API key successfully loaded.")
    else:
        logging.error("OpenAI API key not found. Please set it in the .env file.")
        raise EnvironmentError("OpenAI API key not found.")
    return api_key


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
