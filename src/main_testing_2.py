""" Temp for testing """

from playwright.async_api import async_playwright
import logging
import logging_config
import re
import asyncio
import nest_asyncio

logger = logging.getLogger(__name__)

# Allow nested event loops (needed for Jupyter/IPython environments)
nest_asyncio.apply()


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


async def load_webpages_with_playwright(urls):
    content_dict = {}
    failed_urls = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False
        )  # Headless False for debugging
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
                content = await page.evaluate("document.body.innerText")
                logger.debug(f"Extracted content: {content}")

                if content and content.strip():
                    clean_content = clean_webpage_text(content)
                    content_dict[url] = clean_content
                    logger.info(f"Successfully processed content for {url}")
                else:
                    raise ValueError("No content extracted.")

            except Exception as e:
                logger.error(f"Error occurred while fetching content for {url}: {e}")
                failed_urls.append(url)

        await browser.close()

    return content_dict, failed_urls


async def main():
    urls = [
        "https://www.metacareers.com/jobs/522232286825036/?rx_campaign=Linkedin1&rx_ch=connector&rx_group=126320&rx_job=a1KDp00000E28eGMAR&rx_medium=post&rx_r=none&rx_source=Linkedin&rx_ts=20240927T121201Z&rx_vp=slots&utm_campaign=Job%2Bboard&utm_medium=jobs&utm_source=LIpaid&rx_viewer=e3efacca649311ef917d17a1705b89ba0dc4e1e7a57f4231bbce94a604c83931",
        "https://jobs.careers.microsoft.com/us/en/job/1771714/Head-of-Partner-Intelligence-and-Strategy?jobsource=linkedin",
    ]
    content_dict, failed_urls = await load_webpages_with_playwright(urls)
    print(content_dict, failed_urls)


# For environments like Jupyter, use get_event_loop instead of asyncio.run
if __name__ == "__main__":
    try:
        asyncio.get_running_loop().run_until_complete(main())
    except RuntimeError:  # In case no loop is running (standard Python env)
        asyncio.run(main())
