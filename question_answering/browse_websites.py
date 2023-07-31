from autogpt.commands.web_selenium import browse_website, scrape_text_with_selenium_no_agent
from autogpt.app.main import run_auto_gpt
from autogpt.app.cli import main
import json

with open(r'C:\Users\1kevi\Desktop\projects\Research\autoscious-carbon-capture\question_answering\autoscious_logs\web_search\What_is_the_level_of_ECR_enzyme_activity_in_Kitsatospor_setae_bacteria_.json', 'r') as f:
    web_search_res = json.loads(json.loads(f.read()))

import re

def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9]', '_', filename)

key_question = "What is the level of ECR enzyme activity in Kitsatospor setae bacteria?"
key_question_file_safe = sanitize_filename(key_question)

# browse_website()

for res in web_search_res:
    print("Web search result: ", res)

    # Scrape text
    driver, text = scrape_text_with_selenium_no_agent(res['href'])
    print("driver: ", driver, "text", text)

    # Save complete text
    with open(f'autoscious_logs/web_search/What_is_the_level_of_ECR_enzyme_activity_in_Kitsatospor_setae_bacteria_/{sanitize_filename(res["title"])}_complete_text.txt', 'w', encoding='utf-8') as f:
        f.write(text)

    # TODO: Extract relevant quoted text to key questions from web search results, chrome selenium



    driver.quit()


    # break