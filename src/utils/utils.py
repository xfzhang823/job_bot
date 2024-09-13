"""Tool functions"""

import os
import json
import logging
import re
from dotenv import load_dotenv
import ollama
import openai
from webpage_reader import read_webpages
from prompts.prompts import (
    CLEAN_JOB_PAGE_PROMPT,
    CONVERT_JOB_POSTING_TO_JSON_PROMPT,
)


def check_json_file(filepath):
    """
    Checks if the file path exists and its extension is .json.

    Args:
        filepath (str): The file path to check.

    Returns:
        bool: True if the file exists and has a .json extension, False otherwise.
    """
    # Check if the file exists
    if os.path.exists(filepath):
        # Check if the file extension is .json
        if filepath.lower().endswith(".json"):
            return True
        else:
            print(f"The file '{filepath}' does not have a .json extension.")
    else:
        print(f"The file '{filepath}' does not exist.")

    return False


# Utils tool to list to dict format
def ensure_dict_format(data, prefix="item"):
    """
    Ensure that the input data is in dictionary format.
    If it is a list, convert it to a dictionary with enumerated keys.

    Args:
        data (dict or list): Input data to ensure it is a dictionary.
        prefix (str): Prefix for keys if the data is a list.

    Returns:
        dict: Data converted to a dictionary format.
    """
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        # Convert list to dictionary with keys like 'item_0', 'item_1', etc.
        return {f"{prefix}_{i}": item for i, item in enumerate(data)}
    else:
        raise TypeError("Input data must be a list or dictionary.")


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
    load_dotenv()  # Load environment variables from .env file
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = openai_api_key

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


def load_or_create_json(filepath, key=None):
    """
    Loads a JSON file if it exists and checks if a specific key (e.g., URL) exists in it.
    If the file does not exist, it initializes an empty dictionary for creating a new entry.

    Args:
        filepath (str): The file path to the JSON file.
        key (str, optional): The specific key to check within the JSON data (e.g., a URL).

    Returns:
        tuple: (data, is_existing)
            - data (dict): The loaded or newly created JSON data.
            - is_existing (bool): True if the key exists in the JSON data, False otherwise.
    """
    # Check if the JSON file exists
    if check_json_file(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Check if the specific key exists in the JSON data
                if key and key in data:
                    logging.info(
                        f"Data for key '{key}' already exists in '{filepath}'."
                    )
                    return data, True  # Key exists, return data and True
                else:
                    return data, False  # Key does not exist, return data and False
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.error(f"Error reading {filepath}: {e}")
            return {}, False  # Return empty dict if there is an error
    else:
        # Initialize an empty dictionary if the file does not exist
        return {}, False  # Return empty dict and False since file does not exist


def pretty_print_json(data):
    """
    Pretty prints a Python dictionary or list in JSON format.

    Args:
        data (dict or list): The JSON data (Python dictionary or list) to be pretty printed.

    Returns:
        None
    """
    # Check if the data is a dictionary or list to format it as JSON
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=4))
    else:
        print("The provided data is not a valid JSON object (dict or list).")


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


def add_to_json_file(new_data, filename, key=None):
    """
    Adds or updates data in a master JSON file.

    Args:
        new_data (dict or list): The data to be added or updated in the JSON file.
        filename (str): The path to the master JSON file.
        key (str, optional): If provided, adds or updates the data under this specific key.
                             If not provided, the new data is merged directly with the existing JSON data.

    Returns:
        None
    """
    try:
        # Load existing data from the file
        with open(filename, "r", encoding="utf-8") as f:
            master_data = json.load(f)

        # If a key is provided, add/update under that key
        if key:
            if not isinstance(master_data, dict):
                raise ValueError(
                    "Master JSON data must be a dictionary when using a key."
                )

            # Ensure the key exists and update accordingly
            if key in master_data:
                # If both are lists, extend the list
                if isinstance(master_data[key], list) and isinstance(new_data, list):
                    master_data[key].extend(new_data)  # Append new list items
                # If both are dicts, update the dictionary
                elif isinstance(master_data[key], dict) and isinstance(new_data, dict):
                    master_data[key].update(new_data)  # Merge new dictionary items
                else:
                    raise ValueError(
                        f"Data types for merging under key '{key}' are incompatible."
                    )
            else:
                # If the key does not exist, add it
                master_data[key] = new_data
        else:
            # If no key is provided, attempt to merge the data directly
            if isinstance(master_data, list) and isinstance(new_data, list):
                master_data.extend(new_data)
            elif isinstance(master_data, dict) and isinstance(new_data, dict):
                master_data.update(new_data)
            else:
                raise ValueError(
                    "Mismatched data types or unsupported structure for merging JSON data."
                )

        # Save the updated master data back to the file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(master_data, f, indent=4, ensure_ascii=False)

        logging.info(f"Data successfully added to {filename}.")

    except FileNotFoundError:
        # If the file doesn't exist, create a new one with the new data
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)
        logging.info(f"File {filename} created and data added.")

    except KeyError as e:
        logging.error(f"KeyError when accessing key '{e}' in {filename}.")
        raise

    except ValueError as e:
        logging.error(f"ValueError: {e}")
        raise

    except Exception as e:
        logging.error(f"Error adding data to {filename}: {e}")
        raise


def read_from_json_file(filename, key=None):
    """
    Loads data from a master JSON file and extracts specific sections or values.

    Args:
        filename (str): The path to the master JSON file.
        key (str, optional): If provided, extracts the data under this specific key.

    Returns:
        dict or list: The extracted data from the JSON file.
        None: If the file does not exist or is empty.

    Raises:
        KeyError: If the specified key is not found in the JSON data.
        FileNotFoundError: If the JSON file is not found.
        JSONDecodeError: If there is an error decoding the JSON data.
    """
    # Check directory exist or not
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        raise FileNotFoundError(
            f"Directory '{directory}' does not exist. Please create the directory first."
        )

    # Check if the file exists; if not, return an empty dictionary to handle the creation
    if not os.path.exists(filename):
        logging.info(f"File {filename} not found. It will be created.")
        return {}

    try:
        with open(filename, "r", encoding="utf-8") as f:
            master_data = json.load(f)

        # Debugging: Log the loaded data structure
        logging.info(f"Loaded data from {filename}")

        # If a key is provided, extract data under that key
        if key:
            if key in master_data:
                logging.info(f"Data for key '{key}' found in {filename}.")
                return master_data[key]
            else:
                raise KeyError(f"Key '{key}' not found in the JSON data.")
        else:
            return master_data  # Return the entire JSON data if no key is provided

    except FileNotFoundError:
        # Return None to signal that the file does not exist
        print(f"File {filename} not found. It will be created.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filename}: {e}")
        logging.error(f"Error decoding JSON from {filename}: {e}")
        raise  # Re-raise the exception to handle it outside the function
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        logging.error(f"Error loading data from {filename}: {e}")
        raise  # Re-raise any other exceptions to handle them outside the function


def save_to_json_file(data, filename):
    """
    Saves a Python dictionary or list to a JSON file.

    Args:
        data (dict or list): The data to be saved in JSON format.
        filename (str): The path to the file where the data will be saved.

    Returns:
        None
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"Data successfully saved to {filename}.")
    except Exception as e:
        logging.info(f"Error saving data to {filename}: {e}")
