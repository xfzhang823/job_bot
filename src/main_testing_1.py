""" Temp for testing """

import os
import logging
import json
from IPython.display import display
from playwright.sync_api import sync_playwright
from utils.webpage_reader import load_webpages, clean_text, read_webpages


# Set up logging
logger = logging.getLogger(__name__)


def fetch_synamic_webpage_content(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page.goto(url)

        content = page.text_content('body')

        browser.close()
        return content


def main():

    url = "https://www.metacareers.com/jobs/522232286825036/?rx_campaign=Linkedin1&rx_ch=connector&rx_group=126320&rx_job=a1KDp00000E28eGMAR&rx_medium=post&rx_r=none&rx_source=Linkedin&rx_ts=20240927T121201Z&rx_vp=slots&utm_campaign=Job%2Bboard&utm_medium=jobs&utm_source=LIpaid&rx_viewer=e3efacca649311ef917d17a1705b89ba0dc4e1e7a57f4231bbce94a604c83931"
    content = fetch_synamic_webpage_content(url)

    if content:
        print(content)
    else:
        print("No content extracted.")


if __name__ == "__main__":
    main()
