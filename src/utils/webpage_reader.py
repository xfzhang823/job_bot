"""
Filename: webpage_reader.py
Last updated: 2024 Aug 26

Description: Utilities functions to read and/or summarize webpage(s)

Dependencies: llama_index, ollama
"""

import logging
import sys
import json
from tqdm import tqdm
from IPython.display import display, Markdown
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.llms.ollama import Ollama
from llama_index.core import SummaryIndex

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

# Set llm to ollama/llama3
llm = Ollama(model="llama3", request_timeout=120.0)


def load_webpages(urls):
    """
    Load data from a list of specified URLs using TrafilaturaWebReader.

    Args:
        urls (list): List of URLs to load.

    Returns:
        list: List of documents containing webpage content.
    """
    try:
        reader = TrafilaturaWebReader()
        documents = reader.load_data(urls=urls)
        if not documents:
            raise ValueError(f"No content retrieved from the URLs: {urls}")
        logging.info("Webpages loaded.")
        return documents
    except Exception as e:
        logger.error(f"Error loading webpages: {e}")
        raise


def clean_text(text):
    """
    Clean the extracted text by replacing double newlines with single.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    return text.replace("\n\n", "\n")


def read_webpages(urls, single_page=False):
    """
    Extract and clean text content from one or multiple webpages.

    Args:
        urls (list): List of URLs to read.
        single_page (bool): If True, only the first page will be read. Defaults to False.

    Returns:
        dict: A dictionary with URLs as keys and the concatenated text content as values.
    """
    documents = load_webpages(urls)
    if single_page:
        documents = [documents[0]]  # Read only the first document

    content_dict = {}
    for i, doc in enumerate(documents):
        page_text = clean_text(doc.text)
        content_dict[urls[i]] = page_text
    logging.info("Webpage read.")
    return content_dict


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


def save_output(content_dict, file_path, file_format="txt"):
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

            # Save outputs
            save_output(text_content, "webpage_content.txt")
            save_output(summary_content, "webpage_summary.json", file_format="json")

        except Exception as e:
            logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
