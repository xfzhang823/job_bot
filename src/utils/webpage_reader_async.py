"""
Filename: webpage_reader_async.py
Last updated: 2024 Dec

Description: Utilities functions to read and/or summarize webpage(s).

This module facilitates a two-step process:
1. Extract content from webpages using Playwright.
   - Fetches everything visible on the page, cleans the text, and identifies URLs 
   that fail to load.

2. Transform the extracted content into structured JSON format using an LLM (OpenAI API).
   - Converts raw webpage content into JSON representations suitable for specific use cases, 
   such as job postings.
"""

from pathlib import Path
import logging
import logging_config
import json
from typing import Any, Dict, List, Literal, Tuple, Union
import asyncio
from playwright.async_api import async_playwright
from utils.llm_api_utils_async import call_openai_api_async
from utils.webpage_reader import clean_webpage_text
from prompts.prompt_templates import CONVERT_JOB_POSTING_TO_JSON_PROMPT
from models.llm_response_models import JobSiteResponseModel

# Set up logging
logger = logging.getLogger(__name__)


# Function to load webpage content using Playwright
async def load_webpages_with_playwright_async(
    urls: List[str],
) -> Tuple[Dict[str, str], List[str]]:
    """
    Fetch webpage content using Playwright (for dynamic or complex sites).

    Args:
        urls (List[str]): List of URLs to load.

    Returns:
        Tuple[Dict[str, str], List[str]]:
            A tuple containing:
            - A dictionary where keys are URLs (str) and values are cleaned webpage content (str).
            - A list of URLs (str) that failed to load.

    Raises:
        Exception: Logs errors for individual URLs that fail to load.
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

                # *page.text_content() vs page.evaluate (page.text_content is used in most tutorials)
                # *The page.text_content("body") extracts everything, including all the HTML
                # *and CSS stuff
                # *page.evaluate() directly access the text content via JavaScript.
                # *Here, evaluate is better, b/c it has a more "how readers see it" format,
                # *which is cleaner
                content = await page.evaluate("document.body.innerText")
                logger.debug(f"Extracted content: {content}")

                logger.debug(f"Raw content from Playwright for {url}: {content}\n")
                if content and content.strip():
                    clean_content = clean_webpage_text(content)
                    content_dict[url] = {
                        "url": url,
                        "content": clean_content,
                    }  # include url (needed for later parsing by LLM)

                    logger.info(
                        f"Playwright successfully processed content for {url}\n"
                    )

                else:
                    logger.error(f"No content extracted for {url}\n")
                    failed_urls.append(url)

            except Exception as e:
                logger.error(f"Error occurred while fetching content for {url}: {e}")
                logger.debug(f"Failed URL: {url}")
                failed_urls.append(url)  # Mark the URL as failed

        await browser.close()

    return content_dict, failed_urls  # Return both the content and the failed URLs


async def read_webpages_async(urls: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    Extract and clean text content from one or multiple webpages using Playwright.

    Args:
        urls (List[str]): List of URLs to load and process.

    Returns:
        Tuple[Dict[str, str], List[str]]:
            A tuple containing:
            - A dictionary where keys are URLs (str) and values are cleaned webpage content (str).
            - A list of URLs (str) that failed to load.

    Logs:
        Warnings for any URLs that failed to load.
    """
    documents, failed_urls = await load_webpages_with_playwright_async(urls)

    if failed_urls:
        logger.warning(f"Failed to load the following URLs: {failed_urls}\n")

    return documents, failed_urls


async def convert_to_json_with_gpt_async(
    input_text: str, model_id: str = "gpt-4-turbo", temperature: float = 0.3
) -> Dict[str, Any]:
    """
    Parse and convert job posting content to JSON format using OpenAI API asynchronously.

    Args:
        - input_text (str): The cleaned text to convert to JSON.
        - model_id (str, optional): The model ID to use for OpenAI (default is "gpt-4-turbo").
        - temperature (float, optional): temperature for the model (default is 0.3).

    Returns:
        Dict[str, Any]: A dictionary representation of the extracted JSON content.

    Raises:
        ValueError: If the input text is empty or the response is not in the expected format.

    Logs:
        Information about the prompt and raw response.
        Errors if the response does not match the expected model format.
    """
    if not input_text:
        logger.error("Input text is empty or invalid.")
        raise ValueError("Input text cannot be empty.")

    # Set up the prompt
    prompt = CONVERT_JOB_POSTING_TO_JSON_PROMPT.format(content=input_text)
    logger.info(f"Prompt to convert job posting to JSON format:\n{prompt}")

    # Call the async OpenAI API
    response_model = await call_openai_api_async(
        prompt=prompt,
        model_id=model_id,
        expected_res_type="json",
        json_type="job_site",
        temperature=temperature,
        max_tokens=2000,
    )
    logger.info(f"Validated LLM Response Model: {response_model}")

    # Validate that the response is in the expected JobSiteResponseModel format
    if not isinstance(response_model, JobSiteResponseModel):
        logger.error(
            "Received response is not in expected JobSiteResponseModel format."
        )
        raise ValueError(
            "Received response is not in expected JobSiteResponseModel format."
        )

    # Return the model-dumped dictionary (from Pydantic obj to dict)
    return response_model.model_dump()


# TODO: Need to integrate call llama-3 into this function later on...
# TODO: llama3 has a context window issue (requires chunking first...)
async def process_webpages_to_json_async(urls: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Read webpage content, convert to JSON asynchronously using GPT, and return the result.

    Args:
        urls (List[str]): List of URLs to process.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary where keys are URLs (str) and values are:
            - JSON representation of the processed content (Dict[str, Any]) if successful.
            - Error messages (str) if the processing fails for a URL.

    Logs:
        Information about successfully processed URLs.
        Errors for URLs that fail during processing.
        Lists of URLs that failed to load.
    """
    if isinstance(urls, str):
        urls = [urls]

    webpages_content, failed_urls = await read_webpages_async(urls)
    json_results = {}

    for url, content in webpages_content.items():
        try:
            json_result = await convert_to_json_with_gpt_async(input_text=content)
            logger.info(f"Successfully converted content from {url} to JSON.")
            json_results[url] = json_result
        except Exception as e:
            logger.error(f"Error processing content for {url}: {e}")
            json_results[url] = {"error": str(e)}

    if failed_urls:
        logger.info(f"URLs failed to process:\n{failed_urls}")

    return json_results


def save_webpage_content(
    content_dict: Union[Dict[str, str], Dict[str, Dict[str, Any]]],
    file_path: Union[Path, str],
    file_format: Literal["json", "txt"],
) -> None:
    """
    Save the output content to a file in either txt or json format.

    Args:
        - content_dict (Union[Dict[str, str], Dict[str, Dict[str, Any]]]):
        A dictionary with URLs as keys and content as values.
        - file_path (Union[Path, str]): The file path where the content should be saved.
        - file_format (Literal["json", "txt"]): The format to save the content as,
        either "json" or "txt".

    Returns:
        None

    Raises:
        Exception: If there is an error saving the content to the specified file path.

    Logs:
        Information about successful save operations.
        Errors if the save operation fails.
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
