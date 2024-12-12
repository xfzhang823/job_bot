from utils.webpage_reader_async import load_webpages_with_playwright_async
import asyncio

urls = ["https://careers.microsoft.com/v2/global/en/culture"]

content, _ = asyncio.run(load_webpages_with_playwright_async(urls=urls))

print(content)
