import os
import signal
import subprocess
from autogpt.commands.web_selenium import scrape_text_with_selenium_no_agent
from autogpt.commands.web_search import safe_google_results, web_search_ddg
import json
import os
import re
from util import sanitize_filename
from llm import chat_openai

# Get the list of key questions
def get_key_questions_list_string(search_query_file_safe):
    file_path = f'autoscious_logs/{search_query_file_safe}/decompositions/improved_decomposition.json'

    with open(file_path, 'r') as f:
            question_decomposition = json.load(f)
    key_questions = question_decomposition['key_drivers']['1']['hypotheses']['1']['key_questions']
    print("\nkey_questions\n", key_questions)

    numbered_key_questions_string = ""
    for key, value in key_questions.items():
        first_two_words = ' '.join(value.split())
        numbered_key_questions_string += f'{int(key) - 1}. {first_two_words}\n'

    print("\nnumbered_key_questions_string\n", numbered_key_questions_string)
    return key_questions, numbered_key_questions_string

def extract_facts_from_website_text(search_query_file_safe, key_questions_list_string, website_title, website_text, website_url):
    seed_initial_question_decomposition_prompt = f'''
Key questions (index : question): 
{key_questions_list_string}

Task: 
Extract and output as many accurate direct quotes from the text that are relevant to answering the key questions and its most relevant key question index. Format as a JSON.
```json
{{
  "1": {{
    "quote": "",
    "index": 0
  }},
  etc.
}}
```

Text: {website_text}

Respond only with the output, with no explanation or conversation.
'''
    # Ask GPT the prompt
    print("seed_initial_question_decomposition_prompt", seed_initial_question_decomposition_prompt)
    res = chat_openai(seed_initial_question_decomposition_prompt, model="gpt-3.5-turbo")
    print("Extracted quotes: ", res[0])

    # Save the quote to the corresponding key question index file
    res_json = json.loads(res[0])
    for key, value in res_json.items():
        index = value['index']
        quote = value['quote']

        file_name = f'autoscious_logs/{search_query_file_safe}/facts/kq{index}/facts.txt'

        # Add facts to facts.txt
        with open(file_name, 'a', encoding='utf-8') as f:
            f.write(quote.replace('/"', '"') + f"[{website_url}]" + os.linesep)
    return

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
        print("len(chunk): ", text[i : i + chunk_size])
    return chunks

def kill_chromedriver():
    # Find and kill chromedriver processes
    if os.name == 'nt':  # For Windows
        try:
            subprocess.check_call("taskkill /im chromedriver.exe /f", shell=True)
        except subprocess.CalledProcessError:
            print('No existing chromedriver processes were found.')
    else:  # For Unix-like OS
        try:
            subprocess.check_call(['pkill', 'chromedriver'])
        except subprocess.CalledProcessError:
            print('No existing chromedriver processes were found.')

chunk_size = 4000 # How many characters to read per chunk in website text
overlap = 25 # How much overlap between chunks
MAX_TOKENS = 10000 # Roughly 3K tokens, $0.10 per MAX_TOKENs, max tokens to read per key question
search_query = "How efficiently do the ECR enzymes work, especially in Kitsatospor setae bacteria?"

# 1) Get key questions
search_query_file_safe = sanitize_filename(search_query)
key_questions_dict, key_questions_list_string = get_key_questions_list_string(search_query_file_safe)
print("key_questions_dict", key_questions_dict)

# 2) create facts folder
if not os.path.exists(f'autoscious_logs/{sanitize_filename(search_query)}/facts'):
    os.makedirs(f'autoscious_logs/{sanitize_filename(search_query)}/facts')
for kq_idx in key_questions_dict:
    folder_path = f'autoscious_logs/{sanitize_filename(search_query)}/facts/kq{int(kq_idx)-1}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(f'{folder_path}/facts.txt', 'w') as f:
        pass

# 3) Go through each key question and search for facts
for kq_idx in key_questions_dict:
    curr_tokens = 0
    print(f"Key question searching for: {kq_idx}, {key_questions_dict[kq_idx]}")
    web_search_res = json.loads(web_search_ddg(key_questions_dict[kq_idx]))
    print("Finished searching the web for results", web_search_res)

    # Save web search files
    with open(f'autoscious_logs/{sanitize_filename(search_query)}/facts/kq{int(kq_idx)-1}/web_search_res.json', 'w') as f:
        json.dump(web_search_res, f, indent=2)

    # Use web search results to start searching for facts across all questions, but using this key question's web search results for MAX_TOKENS long
    print("Starting to search for results")
    kill_chromedriver()
    driver = None
    for web_source in web_search_res:
        print("Current tokens for key question: ", curr_tokens)
        if curr_tokens > MAX_TOKENS:
            print("Max tokens reached!")
            break

        url = web_source['href']
        print("Searching web result: ", web_source['title'], url)
        file_title = sanitize_filename(web_source["title"])
        file_path = f'autoscious_logs/{sanitize_filename(search_query)}/facts/kq0/{file_title}_complete_text.txt'

        # If we haven't already scraped the text
        print("file_path: ", file_path)
        if not os.path.exists(file_path):
            print("Scraping text...")

            # Scrape url
            driver, text = scrape_text_with_selenium_no_agent(url, driver)

            # Extract relevant facts to key questions from web search results
            # TODO: Chunk text if text is too long
            chunks = chunk_text(text, chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                print(f"Chunk: {i} / {len(chunks)}")
                extract_facts_from_website_text(search_query_file_safe, key_questions_list_string, file_title, chunk, url)

                curr_tokens += len(chunk)
                if curr_tokens > MAX_TOKENS:
                    print("Max tokens reached in chunks!")
                    break
                break # Just read the first MAX_TOKENS of each source first

            # Save complete text to mark already analyzed
            for other_kq_idx in key_questions_dict:
                if other_kq_idx != kq_idx:
                    # Record that this website title has been used by creating a file with the same name
                    with open(f'autoscious_logs/{sanitize_filename(search_query)}/facts/kq{int(other_kq_idx)-1}/{file_title}_complete_text.txt', 'w', encoding='utf-8') as f:
                        f.write("This website title was already used to extract facts from before.")
            # Record main text in current kq_idx
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
        else:
            print("Already scraped!")

        if curr_tokens > MAX_TOKENS:
            print("Max tokens reached for this key question!")
            break

    if driver:
        driver.quit()
print("SEARCH COMPLETE!")