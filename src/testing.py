from playwright.async_api import async_playwright
import asyncio


async def scrape_with_custom_headers(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Set headers to mimic a real browser
        await page.set_extra_http_headers(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
            }
        )

        try:
            await page.goto(url, timeout=10000)
            await page.wait_for_load_state("networkidle")
            content = await page.evaluate("document.body.innerText")
            print(
                f"Scraped content from {url}:\n", content[:500]
            )  # Print first 500 chars
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
        finally:
            await browser.close()


# Test
asyncio.run(
    scrape_with_custom_headers("https://www.mongodb.com/careers/jobs/6466537")
)  # Replace with your URL
