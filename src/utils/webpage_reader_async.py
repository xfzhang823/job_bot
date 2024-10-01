"""
Filename: webpage_reader.py
Last updated: 2024 Aug 26

Description: Utilities functions to read and/or summarize webpage(s)

Dependencies: llama_index, ollama
"""

import logging
import logging_config
import re
import asyncio
import aiohttp
import json
from playwright.async_api import async_playwright
from utils.llm_data_utils_async import call_openai_api_async
from prompts.prompt_templates import CONVERT_JOB_POSTING_TO_JSON_PROMPT

# from base_models import JSONResponse

# Set up logging
logger = logging.getLogger(__name__)


def clean_webpage_text(content):
    """
    Clean the extracted text by removing JavaScript, URLs, scripts, and excessive whitespace.

    This function performs the following cleaning steps:
    - Removes JavaScript function calls.
    - Removes URLs (e.g., tracking or other unwanted URLs).
    - Removes script tags and their content.
    - Replaces multiple spaces or newline characters with a single space or newline.
    - Strips leading and trailing whitespace.

    Args:
        content (str): The text content to be cleaned.

    Returns:
        str: The cleaned and processed text.
    """
    # Remove JavaScript function calls (e.g., requireLazy([...]))
    content = re.sub(r"requireLazy\([^)]+\)", "", content)

    # Remove URLs (e.g., http, https)
    content = re.sub(r"https?:\/\/\S+", "", content)

    # Remove <script> tags and their contents
    content = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", content)

    # Remove excessive whitespace (more than one space)
    content = re.sub(r"\s+", " ", content).strip()

    # Replace double newlines with single newlines
    content = content.replace("\n\n", "\n")

    return content


# Function to load webpage content using Playwright
async def load_webpages_with_playwright(urls: list) -> tuple:
    """
    Fetch webpage content using Playwright (for dynamic or complex sites).

    Args:
        urls (list): List of URLs to load.

    Returns:
        tuple:
            - dict: Dictionary of cleaned webpage content (strings).
            - list: List of URLs that failed to load.
    """
    content_dict = {}
    failed_urls = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False
        )  # Set headless=False for debugging if needed
        page = await browser.new_page()

        for url in urls:
            try:
                logger.info(
                    f"Attempting to load content with Playwright for URL: {url}"
                )
                await page.goto(url)

                # Wait for network to be idle
                await page.wait_for_load_state("networkidle")

                # Try using page.evaluate() to directly access the text content via JavaScript
                # This gives you a more "how rearders see it" format - cleaner
                logger.debug(f"Extracted content: {content}")
                content = await page.evaluate("document.body.innerText")

                # The page.text_content("body") extracts everything, including all the html and css stuff
                # for this, evaluate is better
                # content = await page.text_content("body")
                logger.debug(f"Raw content from Playwright for {url}: {content}")
                if content and content.strip():
                    clean_content = clean_webpage_text(content)
                    content_dict[url] = clean_content
                    logger.info(f"Playwright successfully processed content for {url}")
                else:
                    logger.error(f"No content extracted for {url}")
                    failed_urls.append(url)

            except Exception as e:
                logger.error(f"Error occurred while fetching content for {url}: {e}")
                failed_urls.append(url)  # Mark the URL as failed

        await browser.close()

    return content_dict, failed_urls  # Return both the content and the failed URLs


def validate_webpage_content(content: str, keywords=None, min_length=600) -> bool:
    """
    Validate the extracted content to check if it is sufficient or relevant.

    Args:
        content (str): The extracted content.
        keywords (list): List of keywords to check in the content (optional).
        min_length (int): Minimum acceptable length of the content (default: 1000 characters).

    Returns:
        bool: True if content is valid, False otherwise.
    """
    # Check if the content meets the minimum length requirement
    if len(content) < min_length:
        return False

    # If keywords are provided, ensure at least one is present in the content
    if keywords:
        for keyword in keywords:
            if keyword.lower() in content.lower():
                return True
        return False

    # If no keywords are provided, assume content is valid if it meets the length threshold
    return True


async def read_webpages(urls: list) -> dict:
    """
    Extract and clean text content from one or multiple webpages using Playwright.
    """
    documents, failed_urls = await load_webpages_with_playwright(urls)

    if failed_urls:
        logger.warning(f"Failed to load the following URLs: {failed_urls}")

    return documents, failed_urls


async def convert_to_json_wt_gpt_async(
    input_text: str, model_id="gpt-4-turbo", temperature=0.3
) -> dict:
    """
    Parse and convert job posting content to JSON format using OpenAI API asynchronously.

    Args:
        input_text (str): The cleaned text to convert to JSON.
        model_id (str): The model ID to use for OpenAI (default is gpt-4-turbo).

    Returns:
        dict: The extracted JSON content as a dictionary.
    """
    # Input validation
    if not input_text:
        logger.error("Input text is empty or invalid.")
        raise ValueError("Input text cannot be empty.")

    # Set up prompt
    prompt = CONVERT_JOB_POSTING_TO_JSON_PROMPT.format(content=input_text)
    logger.info(f"Prompt to convert job posting to JSON format:\n{prompt}")

    # Call the OpenAI API asynchronously
    async with aiohttp.ClientSession() as session:
        try:
            # Constructing the payload for the API call
            payload = call_openai_api_async(
                prompt=prompt,
                model_id=model_id,
                expected_res_type="json",
                temperature=temperature,
                max_tokens=2000,
            )
            logger.info(f"Raw LLM Response: {payload}")

            # Async API call using aiohttp
            async with session.post(
                "https://api.openai.com/v1/chat/completions", json=payload
            ) as response:
                # Raise exception if the status code is not successful
                if response.status != 200:
                    raise Exception(
                        f"API call failed with status code {response.status}"
                    )

                response_data = await response.json()  # Parse the JSON response
                logger.info(f"Raw LLM Response: {response_data}")

                # Check if the response contains valid JSON
                if not isinstance(response_data, dict):
                    logger.error("Received response is not in expected JSON format.")
                    raise ValueError(
                        "Received response is not in expected JSON format."
                    )

                # You can further validate or process `response_data` as needed
                return response_data  # Or process further if required

        except Exception as e:
            logger.error(f"Error in API call: {e}")
            raise


# WIP: Need to integrate call llama-3 into this function later on!
# llama3 has a context window issue.
async def process_webpages_to_json_async(urls: list) -> list:
    """
    Read webpage content, convert to JSON asynchronously using GPT, and return the result.
    """
    # Convert a single URL string into a list if necessary
    if isinstance(urls, str):
        urls = [urls]

    webpages_content, failed_urls = await read_webpages(urls)
    json_results = []

    for url, content in webpages_content.items():
        try:
            json_result = await convert_to_json_wt_gpt_async(
                content
            )  # Convert it to async if needed
            logger.info(f"Successfully converted content from {url} to JSON.")
            json_results.append({url: json_result})
        except Exception as e:
            logger.error(f"Error processing content for {url}: {e}")
            json_results.append({url: f"Error processing content: {e}"})

    if failed_urls:
        logger.info(f"urls failed to process:\n{failed_urls}")

    return json_results


def save_webpage_content(content_dict, file_path, file_format="txt"):
    """
    Save the output content to a file in either txt or json format.

    Args:
        content_dict (dict): A dictionary with URLs as keys and content as values.
        file_path (str): The file path where the content should be saved.
        file_format (str): The format to save the content as, either 'txt' or 'json'. Defaults to 'txt'.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            if file_format == "json":
                json.dump(content_dict, f, ensure_ascii=False, indent=4)
            else:
                for url, content in content_dict.items():
                    f.write(f"--- Content from URL: {url} ---\n{content}\n\n")
        logger.info(f"Content saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving content: {e}")
        raise


def summarize_webpages(urls):
    """
    Summarize the content of one or multiple webpages using a language model.

    Args:
        urls (list): List of URLs to summarize.

    Returns:
        dict: A dictionary with URLs as keys and the summary as values.
    """
    try:
        documents = load_webpages(urls)
        index = SummaryIndex.from_documents(documents)
        query_engine = index.as_query_engine(llm=llm)
        summary_dict = {}

        for i, url in enumerate(urls):
            response = query_engine.query("Summarize the webpage content.")
            summary_dict[url] = str(response)

        logging.info("Webpage summarized.")
        return summary_dict

    except Exception as e:
        logger.error(f"Error summarizing webpages: {e}")
        raise
