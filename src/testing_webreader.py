from pathlib import Path
import asyncio
from pathlib import Path
import logging
import logging_config

from utils.webpage_reader_async import (
    process_webpages_to_json_async,
    save_webpage_content,
)

# Set up logging
logger = logging.getLogger(__name__)


def main():
    urls = [
        "https://careers.ibm.com/job/21200627/research-scientist-ai-climate-sustainability-ibm-research-yorktown-heights-ny/?codes=WEB_Search_NA",  # Replace this with a live URL for testing
    ]

    async def test_webpage_reader():
        # Step 1: Process webpages to JSON
        try:
            logger.info("Starting the webpage processing test...")
            results = await process_webpages_to_json_async(urls)
            logger.info("Processing complete. Results:")
            print(results)

            # Step 2: Save results to a JSON file
            output_file = Path(r"C:\github\job_bot\sandbox\testing_file.json")
            save_webpage_content(results, output_file, file_format="json")
            logger.info(f"Results saved to {output_file}")

        except Exception as e:
            logger.error(f"An error occurred during testing: {e}")

    # Run the async test function
    asyncio.run(test_webpage_reader())


if __name__ == "__main__":
    main()
