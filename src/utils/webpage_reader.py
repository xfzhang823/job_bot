"""
Filename: webpage_reader.py
Last updated: 2024 Aug 26

Description: Utilities functions to read and/or summarize webpage(s)

Dependencies: llama_index, ollama
"""

import logging
import logging_config
import sys
import json
import requests
import re
from bs4 import BeautifulSoup
import asyncio
from playwright.async_api import async_playwright
import trafilatura
from tqdm import tqdm
from IPython.display import display, Markdown
import ollama

# from llama_index.readers.web import TrafilaturaWebReader
from llama_index.llms.ollama import Ollama
from llama_index.core import SummaryIndex
from prompts.prompt_templates import (
    CLEAN_JOB_PAGE_PROMPT,
    CONVERT_JOB_POSTING_TO_JSON_PROMPT,
)
from utils.llm_data_utils import call_openai_api
from base_models import JSONResponse

# Set up logging
logger = logging.getLogger(__name__)


# Set llm to ollama/llama3
llm = Ollama(model="llama3", request_timeout=120.0)

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
async def load_webpages_with_playwright(urls):
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


# Function to load a webpage using requests and clean with BeautifulSoup
def load_webpages_with_requests(urls):
    """
    Fetch webpage content using requests and clean it with BeautifulSoup.

    Args:
        urls (list): List of URLs to load.

    Returns:
        tuple:
            - dict: Dictionary of cleaned webpage content (strings).
            - list: List of URLs that failed to load.
    """
    content_dict = {}
    failed_urls = []

    for url in urls:
        try:
            # Send HTTP request to the URL
            response = requests.get(
                url, timeout=30
            )  # Add a timeout to handle long requests

            # Check if the request was successful
            if response.status_code == 200:
                # Parse HTML content using BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get the text content
                text = soup.get_text()

                # Break into lines and remove leading and trailing space
                lines = (line.strip() for line in text.splitlines())

                # Break multi-headlines into a line each
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )

                # Drop blank lines
                content = "\n".join(chunk for chunk in chunks if chunk)

                # Clean content
                clean_content = clean_webpage_text(content)
                content_dict[url] = clean_content

            else:
                logger.info(
                    f"Failed to retrieve webpage. Status code: {response.status_code}"
                )
                failed_urls.append(url)  # Add to failed URLs list

        except Exception as e:
            logger.error(f"An error occurred while processing {url}: {e}")
            failed_urls.append(url)  # Add to failed URLs list in case of error

    return content_dict, failed_urls  # Return both the content and the failed URLs


def validate_webpage_content(content, keywords=None, min_length=600):
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


# Function to load a webpage using Trafilatura (Using native trafilatura)
def load_webpages_with_trafilatura(urls, validation_keywords=None):
    """
    Attempt to load data from URLs using native Trafilatura.

    Args:
        urls (list): List of URLs to load.
        validation_keywords (list): Optional keywords to validate the content.

    Returns:
        tuple:
            - dict: Dictionary with URLs as keys and cleaned text content as values.
            - list: List of URLs that failed (due to download issues or validation failure).
    """
    # Holder for the dict to hold URL(s) and web content
    content_dict = {}

    # Holder for failed urls (to hand off to other methods)
    failed_urls = []

    for url in urls:
        try:
            # Fetch and extract content using native Trafilatura
            downloaded = trafilatura.fetch_url(url)
            logger.debug(f"Raw content from Trafilatura for {url}: {extracted_content}")
            if downloaded:
                # Extract readable content
                extracted_content = trafilatura.extract(downloaded)

                if extracted_content:
                    # Clean the extracted content
                    clean_content = clean_webpage_text(extracted_content)
                    content_dict[url] = clean_content

                    # Validate the content
                    if validate_webpage_content(
                        clean_content, keywords=validation_keywords
                    ):
                        content_dict[url] = clean_content
                    else:
                        logging.warning(f"Validation failed for content at {url}")
                        content_dict[url] = None
                        failed_urls.append(url)  # Mark URL as failed due to validation
                else:
                    logging.warning(f"Trafilatura failed to extract content from {url}")
                    content_dict[url] = None
                    failed_urls.append(url)  # Mark URL as failed
            else:
                logging.warning(f"Failed to download content from {url}")
                content_dict[url] = None
                failed_urls.append(url)  # Mark URL as failed

        except Exception as e:
            logging.error(f"Error occurred while fetching content for {url}: {e}")
            content_dict[url] = None
            failed_urls.append(url)  # Mark URL as failed due to an exception

    return content_dict, failed_urls

    # Function to load a webpage using Trafilatura (llama_index's traf...reader)
    # Not working properly b/c the pydantic in the code is throwing off the error handling
    # (not returning the failed urls so that I can hand off to playright and requests)
    # def load_webpages_with_trafilatura(urls, validation_keywords=None):
    """
    Attempt to load data from URLs using TrafilaturaWebReader
    (it's not a pure native lib - it's a llama_index wrapper on top of the Trafilatura library)

    Args:
        urls (list): List of URLs to load.

    Returns:
        dict: Dictionary with URLs as keys and cleaned text content as values.
    """
    reader = TrafilaturaWebReader()

    # Holder for the dict to hold URL(s) and web content
    content_dict = {}

    try:
        documents = reader.load_data(urls=urls)

        # Check if any documents were loaded
        if not documents:
            logger.warning("No documents found for the provided URLs.")
            return {
                url: None for url in urls
            }  # Return None for all URLs if nothing was loaded

        # Iterate over the loaded documents
        for i, doc in enumerate(documents):
            # Check if the document has a 'text' attribute (as in Trafilatura Document object)
            if hasattr(doc, "text"):
                # If it's a Trafilatura Document, use its text attribute
                page_text = clean_text(doc.text)
            elif isinstance(doc, str):
                # Handle cases where the document is a string (from raw requests)
                page_text = clean_text(doc)
            else:
                # If neither case applies, log an error and move to the next document
                logger.error(f"Unrecognized document type for URL: {urls[i]}")
                page_text = ""

            # Validate content
            if validate_webpage_content(page_text, keywords=None):
                # Add the cleaned text content to the result dictionary
                content_dict[urls[i]] = page_text
            else:
                logger.warning(f"Trafilatura content validation failed for {urls[i]}")
                content_dict[urls[i]] = None  # Mark for Playwright fallback

        logger.info("Webpage(s) read successfully.")
        return content_dict

    except Exception as e:
        logger.error(f"Error occurred while loading webpages: {str(e)}")
        return {}


# Main function to load webpages (calls Trafilatura first, then falls back to requests)
def load_webpages(urls):
    """
    Load data from a list of specified URLs using Trafilatura.
    If Trafilatura fails, fallback to Playwright.
    If Playwright also fails, fallback to Requests.

    Args:
        urls (list): List of URLs to load.
        validation_keywords (list): List of keywords to validate the content (optional).

    Returns:
        dict: Dictionary of webpage content (cleaned strings).
    """
    documents = {}

    # Step 1: Try loading with Trafilatura
    logger.info("Attempting to load content using Trafilatura...")

    # Call trafilatura to gather web content
    # `documents` is a dictionary, and `failed_urls` is a list
    documents, failed_urls = load_webpages_with_trafilatura(
        urls, validation_keywords=None
    )
    logger.info(f"URLs that failed Trafilatura: {failed_urls}")

    # Step 2: If Trafilatura fails, fallback to Playwright
    # Collect URLs that failed validation (Trafilatura results are None)
    if failed_urls:
        logger.info(f"Falling back to Playwright for {len(failed_urls)} URLs...")
        try:
            # Run Playwright and get the content and failed URLs
            playwright_content, playwright_failed_urls = asyncio.run(
                load_webpages_with_playwright(failed_urls)
            )
            logger.info(f"Playwright processed {len(playwright_content)} URLs")

            # Update the documents with Playwright results
            documents.update(playwright_content)

            # Collect URLs that still failed after Playwright
            failed_urls = playwright_failed_urls
            logger.info(f"URLs that failed Playwright: {failed_urls}")

        except Exception as e:
            logger.error(f"Error in Playwright processing: {e}")

    # Step 3: If Playwright fails, fallback to Requests
    if failed_urls:
        logger.info(f"Falling back to Requests for {len(failed_urls)} URLs...")
        try:
            requests_content, requests_failed_urls = load_webpages_with_requests(
                failed_urls
            )
            logger.info(f"Requests processed {len(requests_content)} URLs")

            # Update the documents with Requests results
            documents.update(requests_content)

            # Collect URLs that still failed after Requests (content is None)
            failed_urls = requests_failed_urls
            logger.info(f"URLs that failed Requests: {failed_urls}")

        except Exception as e:
            logger.error(f"Error in Requests processing: {e}")

    # Raise an error if any URLs failed after all attempts
    if failed_urls:
        raise ValueError(
            f"Failed to load content from the following URLs: {failed_urls}"
        )
    return documents


def read_webpages(urls: list):
    """
    Extract and clean text content from one or multiple webpages.

    Args:
        urls (list): List of URLs or a single URL as a string to read.

    Returns:
        dict: A dictionary with URLs as keys and the concatenated text content as values.

    Example output:
    {'https://example.com/page1': 'Job Posting 1\nThis is a description of the first job.\nResponsibilities include...'}
    """
    # If a single URL is passed as a string, convert it to a list
    if isinstance(urls, str):
        urls = [urls]

    documents = load_webpages(urls)

    content_dict = {}

    for i, doc in enumerate(documents):
        # Check if `doc` is a Trafilatura Document object or a string (from requests)
        if hasattr(doc, "text"):
            # If it's a Trafilatura Document, use doc.text
            page_text = clean_webpage_text(doc.text)
        elif isinstance(doc, str):
            # If it's a raw string from requests, clean it directly
            page_text = clean_webpage_text(doc)
        else:
            # If the document is neither, log an error and continue
            logging.error(f"Unrecognized document type for URL: {urls[i]}")
            page_text = ""

        # Store the cleaned text in the dictionary
        content_dict[urls[i]] = page_text

    logger.info("Webpage(s) read.")
    return content_dict


def convert_to_json_wt_gpt(input_text, model_id="gpt-4-turbo", temperature=0.3):
    """
    Parse and convert job posting content to JSON format using OpenAI API.

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

    # Call the OpenAI API
    response_pyd_obj = call_openai_api(
        prompt=prompt,
        model_id=model_id,
        expected_res_type="json",
        temperature=temperature,
        max_tokens=2000,
    )
    logger.info(f"Raw LLM Response: {response_pyd_obj}")

    if not isinstance(response_pyd_obj, JSONResponse):
        logger.error("Received response is not in expected JSONResponse format.")
        raise ValueError("Received response is not in expected JSONResponse format.")

    return response_pyd_obj.model_dump()


# WIP: Need to integrate call llama-3 into this function later on!
# llama3 has a context window issue.
def process_webpages_to_json(urls):
    """
    Read webpage content, convert to JSON, and return the result.
    Handles both single and multiple webpages.

    Args:
        urls (list or str): A single URL or a list of URLs to process.

    Returns:
        dict or list: A single JSON object (for one webpage) or a list of JSON objects (for multiple webpages).
    """
    # Convert a single URL string into a list if necessary
    if isinstance(urls, str):
        urls = [urls]

    # Read webpage(s) content
    logger.info(f"Reading webpage(s) from: {urls}")
    webpages_content = read_webpages(urls)

    json_results = []

    # Process each webpage content
    for url, content in webpages_content.items():
        try:
            json_result = convert_to_json_wt_gpt(content)
            logger.info(f"Successfully converted content from {url} to JSON.")
            # Store the result as a dictionary with the URL as a key
            json_results.append({url: json_result})
        except Exception as e:
            logger.error(f"Error processing content for {url}: {e}")
            json_results.append({url: f"Error processing content: {e}"})

    # If there's only one result, return it directly
    if len(json_results) == 1:
        return json_results[0]

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
                    logger.info(f"prompt: \n{prompt}")
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


def main():
    urls = [
        "https://www.autosar.org/working-groups/adaptive-platform",
        "https://www.autosar.org/standards/classic-platform",
    ]

    # tqdm progress bar
    with tqdm(total=2, desc="Overall Progress") as pbar:
        try:
            text_content = read_webpages(urls, single_page=False)
            pbar.update(1)

            summary_content = summarize_webpages(urls)
            pbar.update(1)

            # Print the content and summary to the console
            print("\n" + "=" * 40 + "\nOriginal Content:\n" + "=" * 40)
            for url, content in text_content.items():
                print(f"--- Content from URL: {url} ---\n{content}\n")

            print("\n" + "=" * 40 + "\nSummary:\n" + "=" * 40)
            for url, summary in summary_content.items():
                print(f"--- Summary for URL: {url} ---\n{summary}\n")

        except Exception as e:
            logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
