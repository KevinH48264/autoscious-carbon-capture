import re
import json
import time
from duckduckgo_search import DDGS
from itertools import islice
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as GeckoDriverService
from selenium.webdriver.firefox.webdriver import WebDriver as FirefoxDriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.chrome.options import Options as ChromeOptions
from pathlib import Path
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeDriverService
from selenium.webdriver.chrome.webdriver import WebDriver as ChromeDriver
from selenium.webdriver.common.by import By

DUCKDUCKGO_MAX_ATTEMPTS = 3

def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9]', '_', filename)

# Need to determine how useful the text is likely to be for answering the key questions
def get_predicted_usefulness_of_text_prompt(context, decomposition, sample_text_chunks):
    return f'''
Task: 
Based on the research question decomposition key questions and the sample text chunks of the source text, the goal is to identify how useful reading the full source text would be to extract direct quoted facts and determine the best answer to any of the key questions. 

Context:
{context}

Research question decomposition:
{decomposition}

Sample text chunks from the source text:
{sample_text_chunks}

Deliverables:
For each key question, assign a predicted usefulness score of the full source text using a 5-point Likert scale, with 1 being very unlikely to be usefulness to 5 being very likely useful and containing facts that answer the key question.

The output should be of the following JSON format
{{
    <insert key question index>: <insert predicted usefulness>,
   etc.
}}


Respond only with the output, with no explanation or conversation.
'''

def web_search_ddg(query: str, num_results: int = 8) -> str:
    """Return the results of a Google search

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        str: The results of the search.
    """
    search_results = []
    attempts = 0

    while attempts < DUCKDUCKGO_MAX_ATTEMPTS:
        if not query:
            return json.dumps(search_results)

        results = DDGS().text(query)
        search_results = list(islice(results, num_results))

        if search_results:
            break

        time.sleep(1)
        attempts += 1

    results = json.dumps(search_results, ensure_ascii=False, indent=4)
    return safe_google_results(results)

def safe_google_results(results: str | list) -> str:
    """
        Return the results of a Google search in a safe format.

    Args:
        results (str | list): The search results.

    Returns:
        str: The results of the search.
    """
    if isinstance(results, list):
        safe_message = json.dumps(
            [result.encode("utf-8", "ignore").decode("utf-8") for result in results]
        )
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")
    return safe_message

def scrape_text_with_selenium_no_agent(url: str, driver, search_engine = 'firefox') -> str:
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

    print("select firefox options!")
    if search_engine == 'firefox':
        options = FirefoxOptions()
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36"
        )

        options.headless = True
        options.add_argument("--disable-gpu")
        driver = FirefoxDriver(
            service=GeckoDriverService(GeckoDriverManager().install()), options=options
        )
    else:
        # Turns out Chrome actually hangs on some websites but Firefox might now
        # Hard coding Chrome for now
        options = ChromeOptions()
        print("hard coding chrome")

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

    # Set the timeout to 15 seconds, doesn't work on higher numbers for some reason, probably because certificate errors keep showing up
    driver.set_page_load_timeout(15)
    driver.implicitly_wait(15)
    print("set timeout!")

    try:
        driver.get(url)
        print('Page loaded within 15 seconds')
    except TimeoutException:
        print('Page did not load within 15 seconds')
        return "No information found"
    except Exception as e:
        print('An unexpected error occurred:', e)
        return "No information found"
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

    for script in soup(['style', 'script', 'head', 'title', 'meta', '[document]', 'header', 'footer', 'iframe']):
        script.extract()
    print("done extractin")

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    print("Text: ", text[:500])

    driver.quit()

    return text