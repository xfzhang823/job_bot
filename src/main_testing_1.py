""" Temp for testing """

import os
import logging
import json
from IPython.display import display
from playwright.sync_api import sync_playwright
from utils.webpage_reader import load_webpages, clean_webpage_text, read_webpages


# Set up logging
logger = logging.getLogger(__name__)


def json_to_md(json_file, md_file):
    # Format JSON with proper newlines
    with open(json_file, "r") as f:
        data = json.load(f)

        json_output = json.dumps(data, indent=4)

    # Save to a Markdown-like text file
    with open(md_file, "w") as f:
        f.write(json_output)


def main():
    json_file = r"C:\github\job_bot\input_output\preprocessing\jobpostings.json"
    md_file = r"C:\github\job_bot\input_output\preprocessing\jobpostings.md"

    json_file = (
        r"C:\github\job_bot\input_output\preprocessing\extracted_job_requirements.json"
    )
    md_file = (
        r"C:\github\job_bot\input_output\preprocessing\extracted_job_requirements.md"
    )

    json_to_md(json_file, md_file)


if __name__ == "__main__":
    main()
