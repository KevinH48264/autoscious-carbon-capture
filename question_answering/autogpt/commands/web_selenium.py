"""Selenium web scraping module."""
from __future__ import annotations

import logging
from pathlib import Path
from sys import platform
from typing import Optional, Type

from bs4 import BeautifulSoup
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeDriverService
from selenium.webdriver.chrome.webdriver import WebDriver as ChromeDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeDriverService
from selenium.webdriver.edge.webdriver import WebDriver as EdgeDriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as GeckoDriverService
from selenium.webdriver.firefox.webdriver import WebDriver as FirefoxDriver
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.safari.webdriver import WebDriver as SafariDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager as EdgeDriverManager
from selenium.common.exceptions import TimeoutException

import os
import signal
import subprocess

# from autogpt.agents.agent import Agent
from autogpt.command_decorator import command
from autogpt.logs import logger
# from autogpt.memory.vector import MemoryItem, get_memory
from autogpt.processing.html import extract_hyperlinks, format_hyperlinks
from autogpt.url_utils.validators import validate_url

BrowserOptions = ChromeOptions | EdgeOptions | FirefoxOptions | SafariOptions

FILE_DIR = Path(__file__).parent.parent

@command(
    "browse_website",
    "Browses a Website",
    {
        "url": {"type": "string", "description": "The URL to visit", "required": True},
        "question": {
            "type": "string",
            "description": "What you want to find on the website",
            "required": True,
        },
    },
)
@validate_url
def browse_website(url: str, question: str, agent: Agent) -> str:
    print("browsing website!!", url, question, agent)
    """Browse a website and return the answer and links to the user

    Args:
        url (str): The url of the website to browse
        question (str): The question asked by the user

    Returns:
        Tuple[str, WebDriver]: The answer and links to the user and the webdriver
    """
    try:
        driver, text = scrape_text_with_selenium(url, agent)
    except WebDriverException as e:
        # These errors are often quite long and include lots of context.
        # Just grab the first line.
        msg = e.msg.split("\n")[0]
        return f"Error: {msg}"

    add_header(driver)
    summary = summarize_memorize_webpage(url, text, question, agent, driver)
    links = scrape_links_with_selenium(driver, url)

    # Limit links to 5
    if len(links) > 5:
        links = links[:5]
    close_browser(driver)
    return f"Answer gathered from website: {summary}\n\nLinks: {links}"


def scrape_text_with_selenium(url: str, agent: Agent) -> tuple[WebDriver, str]:
    print("agent.config.selenium_web_browser", agent.config.selenium_web_browser)

    """Scrape text from a website using selenium

    Args:
        url (str): The url of the website to scrape

    Returns:
        Tuple[WebDriver, str]: The webdriver and the text scraped from the website
    """
    logging.getLogger("selenium").setLevel(logging.CRITICAL)

    options_available: dict[str, Type[BrowserOptions]] = {
        "chrome": ChromeOptions,
        "edge": EdgeOptions,
        "firefox": FirefoxOptions,
        "safari": SafariOptions,
    }

    options: BrowserOptions = options_available[agent.config.selenium_web_browser]()
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36"
    )

    if agent.config.selenium_web_browser == "firefox":
        if agent.config.selenium_headless:
            options.headless = True
            options.add_argument("--disable-gpu")
        driver = FirefoxDriver(
            service=GeckoDriverService(GeckoDriverManager().install()), options=options
        )
    elif agent.config.selenium_web_browser == "edge":
        driver = EdgeDriver(
            service=EdgeDriverService(EdgeDriverManager().install()), options=options
        )
    elif agent.config.selenium_web_browser == "safari":
        # Requires a bit more setup on the users end
        # See https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari
        driver = SafariDriver(options=options)
    else:
        if platform == "linux" or platform == "linux2":
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--remote-debugging-port=9222")

        options.add_argument("--no-sandbox")
        if agent.config.selenium_headless:
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")

        chromium_driver_path = Path("/usr/bin/chromedriver")

        driver = ChromeDriver(
            service=ChromeDriverService(str(chromium_driver_path))
            if chromium_driver_path.exists()
            else ChromeDriverService(ChromeDriverManager().install()),
            options=options,
        )
    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # Get the HTML content directly from the browser's DOM
    page_source = driver.execute_script("return document.body.outerHTML;")
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return driver, text

def scrape_text_with_selenium_no_agent(url: str, driver: WebDriver) -> str:
    print("Going through url: ", url)

    """Scrape text from a website using selenium

    Args:
        url (str): The url of the website to scrape
        driver (WebDriver, optional): The webdriver to use for scraping. If None, a new webdriver will be created.

    Returns:
        str: The text scraped from the website
    """
    # Timeouts are really buggy with passing in and out driver so I'm going going to reuse drivers.
    # if driver is None:
    print("select chrome options!")
    options: BrowserOptions = ChromeOptions()
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36"
    )

    # Hard coding Chrome for now
    print("hard coding chrome")
    if platform == "linux" or platform == "linux2":
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--remote-debugging-port=9222")

    options.add_argument("--no-sandbox")
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--incognito")
    
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    chromium_driver_path = Path("/usr/bin/chromedriver")

    print("setting up chrome driver")
    driver = ChromeDriver(
        service=ChromeDriverService(str(chromium_driver_path))
        if chromium_driver_path.exists()
        else ChromeDriverService(ChromeDriverManager().install()),
        options=options,
    )

    print("Driver is getting url")

    # Set the timeout to 10 seconds, doesn't work on higher numbers for some reason, probably because certificate errors keep showing up
    driver.set_page_load_timeout(10)
    driver.implicitly_wait(10)
    print("set timeout!")

    try:
        driver.get(url)
        print('Page loaded within 10 seconds')
    except TimeoutException:
        print('Page did not load within 10 seconds')
        return driver, "No information found"
    except Exception as e:
        print('An unexpected error occurred:', e)
        return driver, "No information found"
    except: 
        print("there was an error")
    print("Driver got url")

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    print("Driver has found page source")
    # Get the HTML content directly from the browser's DOM
    page_source = driver.execute_script("return document.body.outerHTML;")
    print("Handing off to Beautiful Soup!")
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(['style', 'script', 'head', 'title', 'meta', '[document]']):
        script.extract()
    print("done extractin")

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    print("Text: ", text[:500])

    driver.quit()

    return driver, text

# Selenium currently takes a long time to load the page so I'll either add timeout or just currently use BeautifulSoup and pdf parser
def scrape_text_with_bs_no_agent(url: str) -> tuple[WebDriver, str]:
    print("Going through url: ", url)

    """Scrape text from a website using selenium

    Args:
        url (str): The url of the website to scrape

    Returns:
        Tuple[WebDriver, str]: The webdriver and the text scraped from the website
    """
    # logging.getLogger("selenium").setLevel(logging.CRITICAL)

    # options_available: dict[str, Type[BrowserOptions]] = {
    #     "chrome": ChromeOptions,
    #     "edge": EdgeOptions,
    #     "firefox": FirefoxOptions,
    #     "safari": SafariOptions,
    # }

    options: BrowserOptions = ChromeOptions()
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36"
    )

    # Hard coding Chrome for now
    if platform == "linux" or platform == "linux2":
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--remote-debugging-port=9222")

    options.add_argument("--no-sandbox")
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")

    chromium_driver_path = Path("/usr/bin/chromedriver")

    driver = ChromeDriver(
        service=ChromeDriverService(str(chromium_driver_path))
        if chromium_driver_path.exists()
        else ChromeDriverService(ChromeDriverManager().install()),
        options=options,
    )
    print("Driver is getting url")
    driver.get(url)
    print("Driver got url")

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    print("Driver has found page source")
    # Get the HTML content directly from the browser's DOM
    page_source = driver.execute_script("return document.body.outerHTML;")
    print("Handing off to Beautiful Soup!")
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return driver, text

def scrape_links_with_selenium(driver: WebDriver, url: str) -> list[str]:
    """Scrape links from a website using selenium

    Args:
        driver (WebDriver): The webdriver to use to scrape the links

    Returns:
        List[str]: The links scraped from the website
    """
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup, url)

    return format_hyperlinks(hyperlinks)


def close_browser(driver: WebDriver) -> None:
    """Close the browser

    Args:
        driver (WebDriver): The webdriver to close

    Returns:
        None
    """
    driver.quit()


def add_header(driver: WebDriver) -> None:
    """Add a header to the website

    Args:
        driver (WebDriver): The webdriver to use to add the header

    Returns:
        None
    """
    try:
        with open(f"{FILE_DIR}/js/overlay.js", "r") as overlay_file:
            overlay_script = overlay_file.read()
        driver.execute_script(overlay_script)
    except Exception as e:
        print(f"Error executing overlay.js: {e}")


def summarize_memorize_webpage(
    url: str,
    text: str,
    question: str,
    agent: Agent,
    driver: Optional[WebDriver] = None,
) -> str:
    """Summarize text using the OpenAI API

    Args:
        url (str): The url of the text
        text (str): The text to summarize
        question (str): The question to ask the model
        driver (WebDriver): The webdriver to use to scroll the page

    Returns:
        str: The summary of the text
    """
    if not text:
        return "Error: No text to summarize"

    text_length = len(text)
    logger.info(f"Text length: {text_length} characters")

    memory = get_memory(agent.config)

    new_memory = MemoryItem.from_webpage(text, url, agent.config, question=question)
    memory.add(new_memory)
    return new_memory.summary
