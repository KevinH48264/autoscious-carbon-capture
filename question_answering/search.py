from autogpt.commands.web_selenium import browse_website, scrape_text_with_selenium_no_agent
from autogpt.app.main import run_auto_gpt
from autogpt.app.cli import main
import json
from util import sanitize_filename
import os
from prompts import get_predicted_usefulness_of_text_prompt
from collections import defaultdict
from llm import chat_openai

search_query = "How efficiently do the ECR enzymes work, especially in Kitsatospor setae bacteria?"

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

def find_title(url, web_search_inf):
    for item in web_search_inf:
        if item["href"] == url:
            return item["title"]
    return None

# COMPLETE code for predicting usefulness of very relevant (5) and relevant (4) results.
from autogpt.commands.web_selenium import scrape_text_with_selenium_no_agent

CHUNK_SIZE = 1000
SAMPLING_FACTOR = 0.1 # Also cap it so it falls under the max token limit
MAX_TOKENS = 2500 * 4 # 1 token = 4 chars, 2500 + 500 (prompt) tokens is high for GPT3.5
MAX_CHUNKS = int(MAX_TOKENS / CHUNK_SIZE)
context = "Enoyl-CoA carboxylase/reductase enzymes (ECRs)"

# Load in decomposition
with open(f'autoscious_logs/{sanitize_filename(search_query)}/decompositions/improved_decomposition.json', 'r') as f:
    decomposition = json.load(f)

# Load in rated web_results
question_idx = 0
with open(f'autoscious_logs/{sanitize_filename(search_query)}/sources/kq0/rated_web_results_{question_idx}.json', 'r') as f:
    rated_web_results = json.load(f)
with open(f'autoscious_logs/{sanitize_filename(search_query)}/sources/kq0/initial_queries_{question_idx}_web_search_res.json', 'r') as f:
    web_search_info = json.load(f)

# Need to extract only URLs with 5s and 4s to go through
ratings_url_dict = defaultdict(list)
for url, ratings in rated_web_results.items():
    if ratings != -1:  # only process if the ratings is not -1
        for rating in ratings:
            ratings_url_dict[str(rating[1])].append(url)  # append the URL to the correct category
    else:
        ratings_url_dict[str(ratings)].append(url)
print("ratings_url_dict", ratings_url_dict)

# Start with iterating through 4s and 5s of ratings_url_dict
for rating, urls in ratings_url_dict.items():
    if rating == '5':
        folder_path = f'autoscious_logs/{sanitize_filename(search_query)}/sources/kq0/predicted_usefulness_{rating}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for rating_source_idx, url in enumerate(urls):
            print("rating_source_idx", rating_source_idx, "Skimming url:", url)
            text = scrape_text_with_selenium_no_agent(url, None)

            # Only evaluate websites you're able to scrape
            if text != "No information found":
                total_chunks = len(text) / CHUNK_SIZE
                num_chunk_samples = min(int(total_chunks * SAMPLING_FACTOR), MAX_CHUNKS)
                sample_chunks = get_sample_chunks(text, CHUNK_SIZE, num_chunk_samples)
                print("len(sample_chunks)", len(sample_chunks))

                # Get predicted usefulness based on sample chunks
                predicted_usefulness_results = json.loads(chat_openai(get_predicted_usefulness_of_text_prompt(context, decomposition, sample_chunks), model="gpt-3.5-turbo")[0])

                # save filtered search results
                with open(f'autoscious_logs/{sanitize_filename(search_query)}/sources/kq0/predicted_usefulness_{rating}/{rating_source_idx}.json', 'w') as f:
                    json.dump(predicted_usefulness_results, f, indent=2)
                
                # Check if any scores were (4 or) 5, because then we should save the full text
                if 5 in predicted_usefulness_results.values() or '5' in predicted_usefulness_results.values():
                    title = find_title(url, web_search_info)
                    with open(f'autoscious_logs/{sanitize_filename(search_query)}/sources/full_text/{sanitize_filename(title)}.txt', 'w', encoding='utf-8') as f:
                        f.write(title + '\n')
                        f.write(url + '\n')
                        f.write(text)