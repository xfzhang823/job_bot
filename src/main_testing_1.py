""" Temp for testing """

import os
import logging
import json
from IPython.display import display
from playwright.sync_api import sync_playwright
from utils.webpage_reader import load_webpages, clean_webpage_text, read_webpages
from utils.generic_utils import read_from_json_file
from config import job_descriptions_json_file, METRICS_OUTPUTS_CSV_FILES_DIR
from pipelines.resume_eval_pipeline import fetch_urls_for_eval

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

    description = read_from_json_file(job_descriptions_json_file)
    # Extract URLs from the keys of the job descriptions JSON
    urls = [
        key for key in description.keys() if key
    ]  # Keep valid, non-empty keys (URLs)
    print(type(urls))
    output_dir = METRICS_OUTPUTS_CSV_FILES_DIR / "iteration_0"
    print(output_dir)
    new_urls = fetch_urls_for_eval(urls, output_dir)
    print(new_urls)

    try:
        # Assuming this is where you're fetching URLs
        urls = get_urls_for_evaluation()  # Replace with your actual function name
        print(f"Type of urls: {type(urls)}")
        print(f"Content of urls: {urls[:5]}")  # Print first 5 URLs for debugging
        
        # Your existing code to check metrics output files
        existing_urls, new_urls = check_metrics_output_file_list(urls, output_dir)
    except Exception as e:
        logger.error(f"An error occurred while fetching URLs for evaluation: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error args: {e.args}")
        # Print the full traceback for more detailed error information
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
if __name__ == "__main__":
    main()
