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
from rank_bm25 import BM25Okapi
import re
import requests
from PyPDF2 import PdfReader 
from io import BytesIO
import time
from collections import defaultdict
from dotenv import load_dotenv
from googleapiclient.discovery import build
import os
from llm import chat_openai

# Load google search credentials
load_dotenv()

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

def get_initial_search_queries_prompt(key_question, search_engine):
  return f'''
Key question:
{key_question}

Task:
For the key question, write a clear and comprehensive but short (1 query) list of search queries optimized for best search engine results, so that you can confidently and quickly surface the most relevant information to determine the best answer to the question. Extract a string of search keywords query from the key question.

The output should be in JSON format: 
```json
{{
  "1": "<insert query>",
  "keywords_query": "<insert keywords>"
}}

Respond only with the output, with no explanation or conversation.
'''

def get_filtering_web_results_ratings(key_question, web_search_res):
    return f'''
Key question:
{key_question}

Task:
Based on the key question and each search result's title and body content, reason and assign a predicted usefulness score of the search result's content and potential useful references to answering the key question using a 5-point Likert scale, with 1 being very not useful, 2 being not useful, 3 being somewhat useful, 4 being useful, 5 being very useful.

Search results:
{web_search_res}

The output should be in JSON format: 
```json
{{
  'href': 'relevance score',
  etc.
}}
```

Respond only with the output, with no explanation or conversation.
'''

# Need to determine how useful the text is likely to be for answering the key questions
def get_predicted_usefulness_of_text_prompt(key_question, sample_text_chunks):
    return f'''
Key question:
{key_question}

Task: 
Based on the key question and the sample text chunks of the source text, the goal is to identify how useful reading the full source text would be to extract direct quoted facts or references to determine the best answer to the key question. 

Deliverable:
Assign a predicted usefulness score of the full source text using a 5-point Likert scale, with 1 being very unlikely to be usefulness, 2 being unlikely to be useful, 3 being somewhat likely to be useful, 4 being likely to be useful, and 5 being very likely useful and containing facts or references that answer the key question.

Sample text chunks from the source text:
{sample_text_chunks}

The output should be of the following JSON format
{{
    ""predicted_usefulness: <insert predicted usefulness rating>,
   etc.
}}


Respond only with the output, with no explanation or conversation.
'''

def get_most_relevant_chunks_with_bm25(key_question, text, CHUNK_SIZE, num_chunk_samples):
    # 1. Split text into chunks
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    # 2. Tokenize the chunks
    tokenized_chunks = [re.findall(r"\w+", chunk) for chunk in chunks]

    # 3. Initialize BM25
    bm25 = BM25Okapi(tokenized_chunks)

    # 4. Query BM25 with the key question
    tokenized_question = re.findall(r"\w+", key_question)
    scores = bm25.get_scores(tokenized_question)

    # 5. Sort chunks by BM25 scores
    sorted_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)]

    # 6. Return top num_chunk_samples chunks
    return sorted_chunks[:num_chunk_samples]

def find_title(url, web_search_info):
    for item in web_search_info:
        if item["href"] == url:
            return item["title"]
    return None

def check_pdf(url, web_search_info):
    for item in web_search_info:
        if "pdf" in item.keys() and item["pdf"]:
            return True
    print("Not pdf")
    return False

def try_getting_pdf(url):
    try:
        response = requests.get(url, verify=True)
        f = BytesIO(response.content)
        pdf = PdfReader(f)
        return True
    except:
        print("Could not get pdf")
        return False

# Get the PDF content
def try_getting_pdf_content(url):
    try:
        response = requests.get(url, verify=True)
        f = BytesIO(response.content)
        pdf = PdfReader(f)
        content = ""

        for i in range(len(pdf.pages)):
            page = pdf.pages[i]
            text = page.extract_text()
            content += text
        return content
    except:
        print("Error getting PDF content")
        return ""

def google_search_raw(search_term, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=os.getenv('DEV_KEY'))
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()

    search_results = res.get("items", [])
    time.sleep(1)

    # Create a list of only the URLs from the search results
    search_results_links = [item["link"] for item in search_results]
    return search_results

def search_google(search_query):
    num_google_searches = 8
    results = google_search_raw(search_query, os.getenv('MY_CSE_ID'), num=num_google_searches, lr="lang_en", cr="countryUS")
    return results

def get_sample_chunks(text, CHUNK_SIZE, num_chunk_samples):
    step_size = len(text) // num_chunk_samples

    chunks = []
    for i in range(0, len(text), step_size):
        chunk = text[i:i+CHUNK_SIZE]
        chunks.append(chunk)

        # Break after getting the required number of chunks
        if len(chunks) >= num_chunk_samples:
            break

    return chunks

# Can be broken to prompts.py and util.py
def extract_facts_from_website_text(search_query_file_safe, key_question, website_title, website_text, website_url):
    seed_initial_question_decomposition_prompt = f'''
Key question: 
{key_question}

Task:
What's the most useful answer based on the source text? Give me as specific and correct of an answer as possible. Then, quote the section of the source text that supports your answer. 

The output should be in JSON format: 
```json
{{
  "best_answer": "<insert most useful answer>",
  "quote": "<insert quote>",
}}
```

Source text: 
{website_text}

Respond only with the output, with no explanation or conversation.
'''
    # Ask GPT the prompt
    print("seed_initial_question_decomposition_prompt", seed_initial_question_decomposition_prompt)
    res = chat_openai(seed_initial_question_decomposition_prompt, model="gpt-3.5-turbo")
    print("Extracted quotes: ", res[0])

    # Save the quote to the corresponding key question index file
    res_json = json.loads(res[0])
    # for key, value in res_json.items():
    answer = res_json.get("best_answer", "")
    quote = res_json.get("quote", "")

    # Only log if there is a quote
    if quote and type(quote) == str:
        file_name = f'autoscious_logs/{search_query_file_safe}/facts/facts.txt'

        with open(file_name, 'a', encoding='utf-8') as f:
            f.write(answer + os.linesep)

        # Save the best answer and quote into a JSON for reference retrieval later
        json_file_name = f'autoscious_logs/{search_query_file_safe}/facts/facts.json'
        if os.path.exists(json_file_name):
            with open(json_file_name, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
        else:
            data = {}

        # Update the dictionary and save it back
        data[answer] = quote.replace('/"', '"') + f"[{website_url}]"
        
        with open(json_file_name, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

    return

def chunk_text(text: str, key_question: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """
    Splits a text into overlapping chunks and ranks them using BM25 based on relevance to a key question.
    
    Args:
    - text (str): The source text.
    - key_question (str): The key question to rank the chunks by.
    - chunk_size (int): The size of each chunk.
    - num_chunk_samples (int): The number of top-ranked chunks to return.
    - overlap (int): The size of the overlap between chunks. Default is 0.
    
    Returns:
    - list[str]: The top-ranked chunks based on BM25 scores.
    """

    # 1. Split text into overlapping chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

    # 2. Tokenize the chunks
    tokenized_chunks = [re.findall(r"\w+", chunk) for chunk in chunks]

    # 3. Initialize BM25
    bm25 = BM25Okapi(tokenized_chunks)

    # 4. Query BM25 with the key question
    tokenized_question = re.findall(r"\w+", key_question)
    scores = bm25.get_scores(tokenized_question)

    # 5. Sort chunks by BM25 scores
    sorted_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)]

    # 6. Return top num_chunk_samples chunks
    return sorted_chunks

def get_rerank_facts_list_prompt(search_query, facts_list):
  return f'''
Key question: 
{search_query}

Task:
Rerank the following facts based on how useful they are and well they answer the key question. The more useful, specific, and correct, the better. The best answer should be at the top, and the worst answer should be at the bottom of the list. Use direct quotes and do not change the wording of the facts. Leave out facts that are not relevant to the key question.

The output should be a JSON list of facts:
```json
['<fact>', etc.]
```

Current facts list:
{facts_list}

Respond only with the output, with no explanation or conversation. I expect a first-rate answer.
'''