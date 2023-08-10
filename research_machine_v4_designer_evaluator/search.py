# Input = a search query
# Output = a set of facts and conclusion

import json
from util import sanitize_filename, get_predicted_usefulness_of_text_prompt, web_search_ddg, get_initial_search_queries_prompt, get_filtering_web_results_ratings, scrape_text_with_selenium_no_agent, get_predicted_usefulness_of_text_prompt, get_sample_chunks, search_google, google_search_raw, try_getting_pdf_content, try_getting_pdf, check_pdf, find_title, get_most_relevant_chunks_with_bm25, extract_facts_from_website_text, chunk_text, get_rerank_facts_list_prompt
import os
from collections import defaultdict
from llm import chat_openai
from datetime import datetime
from scholarly import ProxyGenerator
from scholarly import scholarly
import json

def search_web_for_facts(search_query):
    # Starting web search for facts with this search query
    print(f"****Starting web search for facts with this search query: {search_query}****")

    search_query_file_safe = sanitize_filename(search_query[:50])
    search_engine = "academic"

    folder_path = f'autoscious_logs/{search_query_file_safe}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = f'autoscious_logs/{search_query_file_safe}/sources'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # technical debt of just setting variables to search_query
    key_question_initial_search_queries = {"1": search_query}
    keywords_query = search_query
    with open(f'autoscious_logs/{search_query_file_safe}/sources/initial_search_queries.json', 'w') as f:
        json.dump(key_question_initial_search_queries, f, indent=2)
    with open(f'autoscious_logs/{search_query_file_safe}/sources/keywords_query.txt', 'w') as f:
        json.dump(keywords_query, f)

    # Web search given search keywords
    print("****Web search given search keywords****")

    pg = ProxyGenerator()
    pg.FreeProxies()
    scholarly.use_proxy(pg)

    MAX_RETRIES = 3

    # for decomposition_idx, key_question_decomposition in enumerate(key_question_decomposition_list):
    with open(f'autoscious_logs/{search_query_file_safe}/sources/initial_search_queries.json', 'r') as f:
        key_question_initial_search_queries = json.load(f)

    for idx, query in key_question_initial_search_queries.items():
        print("query: ", query)

        # Skip already searched queries
        folder_path = f'autoscious_logs/{search_query_file_safe}/sources/initial_search_results_query_{idx}.json'
        if os.path.exists(folder_path):
            print("already searched query!")
            continue

        web_search_res = []
        if search_engine == "academic":
            print("trying academic search")
            try:
                scholar_res_gen = scholarly.search_pubs(query)

                for res in scholar_res_gen:
                    item = {}
                    item['title'] = res['bib']['title']
                    if try_getting_pdf(res['eprint_url']):
                        item['href'] = res['eprint_url']
                        item['pdf'] = True
                    else:
                        item['href'] = res['pub_url']
                        item['pdf'] = False
                    item['body'] = res['bib']['abstract']
                    web_search_res += [item]
                    print("Adding academic paper: ", item)
            except: 
                print("Exception, trying normal search")
        if web_search_res == []:
            # DDG
            print("trying normal search")
            web_search_res = json.loads(web_search_ddg(query))
            if len(web_search_res) == 0:
                print("trying google search!")
                # Google
                web_search_res_raw = search_google(query) # google uses 'link' instead of 'href'
                web_search_res = [{
                    'title': web_search_res_raw[i]['title'], 
                    'href': web_search_res_raw[i]['link'], 
                    'body': web_search_res_raw[i]['snippet'],
                    'pdf': False
                    } for i in range(len(web_search_res_raw))
                ]

        # save web search results
        with open(f'autoscious_logs/{search_query_file_safe}/sources/initial_search_results_query_{idx}.json', 'w') as f:
            json.dump(web_search_res, f, indent=2)

    # Reading type 1: filtering unlikely relevant sources based on title and body
    print("****Reading type 1: filtering unlikely relevant sources based on title and body****")
    with open(f'autoscious_logs/{search_query_file_safe}/sources/initial_search_queries.json', 'r') as f:
        key_question_initial_search_queries = json.load(f)

    for query_idx, query in key_question_initial_search_queries.items():
        # Skip already filtered queries
        folder_path = f'autoscious_logs/{search_query_file_safe}/sources/rated_web_results_query_{int(query_idx)}.json'
        if os.path.exists(folder_path):
            print("already filtered query!")
            continue

        # load web search results
        with open(f'autoscious_logs/{search_query_file_safe}/sources/initial_search_results_query_{query_idx}.json', 'r') as f:
            web_search_res = json.loads(f.read())
        
        filtered_web_results = {}
        if web_search_res != []:
            # filter web results based on title and body
            filtered_web_results = json.loads(chat_openai(get_filtering_web_results_ratings(search_query, web_search_res), model="gpt-3.5-turbo")[0])

        ratings_url_dict = defaultdict(list)
        for url, rating in filtered_web_results.items():
            ratings_url_dict[str(rating)].append(url)

        # save filtered search results
        with open(f'autoscious_logs/{search_query_file_safe}/sources/rated_web_results_query_{int(query_idx)}.json', 'w') as f:
            json.dump(ratings_url_dict, f, indent=2)

    # Reading type 2 & 3: filtering via skimming
    print("****Reading type 2 & 3: filtering via skimming****")
    CHUNK_SIZE = 1000
    SAMPLING_FACTOR = 0.1 # Also cap it so it falls under the max token limit
    MAX_TOKENS = 2500 * 4 # 1 token = 4 chars, 2500 + 500 (prompt) tokens is high for GPT3.5
    MAX_CHUNKS = int(MAX_TOKENS / CHUNK_SIZE)

    folder_path = f'autoscious_logs/{search_query_file_safe}/sources/full_text'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Skimming through each highly relevant paper from skimming
    with open(f'autoscious_logs/{search_query_file_safe}/sources/initial_search_queries.json', 'r') as f:
        key_question_initial_search_queries = json.load(f)

    for query_idx, query in key_question_initial_search_queries.items():
        # open filtered search results
        with open(f'autoscious_logs/{search_query_file_safe}/sources/rated_web_results_query_{int(query_idx)}.json', 'r') as f:
            ratings_url_dict = json.loads(f.read())

        # open web search info to extract metadata
        with open(f'autoscious_logs/{search_query_file_safe}/sources/initial_search_results_query_{int(query_idx)}.json', 'r') as f:
            web_search_info = json.load(f)
        
        for rating, urls in ratings_url_dict.items():
            if rating == '5' or rating == '4' or rating == '3': # Scraping all useful websites to skim through
                # Start with iterating through 4s and 5s of ratings_url_dict
                folder_path = f'autoscious_logs/{search_query_file_safe}/sources/predicted_usefulness_{rating}'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                for rating_source_idx, url in enumerate(urls):
                    print("query ", query_idx, "rating_source_idx", rating_source_idx, "Skimming url:", url)

                    # Ensure the url hasn't already been visited
                    title = find_title(url, web_search_info)
                    if title and not os.path.exists(f'autoscious_logs/{search_query_file_safe}/sources/full_text/{sanitize_filename(title)}.txt') and not os.path.exists(f'{folder_path}/query_{query_idx}_url_index_{rating_source_idx}.json'):

                        # Check if it's a pdf or not
                        if try_getting_pdf(url):
                            print("PDF found!")
                            text = try_getting_pdf_content(url)
                        else:
                            text = scrape_text_with_selenium_no_agent(url, None, search_engine='chrome')

                        # Only evaluate websites you're able to scrape
                        if text and text != "No information found":
                            # Use full text if it's short
                            if len(text) > 10000:
                                total_chunks = len(text) / CHUNK_SIZE
                                num_chunk_samples = min(int(total_chunks * SAMPLING_FACTOR), MAX_CHUNKS)
                                # sample_chunks = get_sample_chunks(text, CHUNK_SIZE, num_chunk_samples)
                                sample_chunks = get_most_relevant_chunks_with_bm25(keywords_query, text, CHUNK_SIZE, num_chunk_samples) # Using BM25 to search for keywords instead of general query
                                print("len(sample_chunks)", len(sample_chunks))
                            else:
                                sample_chunks = [text]
                            print("len(text)", len(text), "sample chunks: ", sample_chunks)

                            # Get predicted usefulness based on sample chunks
                            predicted_usefulness_results = json.loads(chat_openai(get_predicted_usefulness_of_text_prompt(search_query, sample_chunks), model="gpt-3.5-turbo")[0])

                            # save filtered search results
                            with open(f'{folder_path}/query_{query_idx}_url_index_{rating_source_idx}.json', 'w') as f:
                                predicted_usefulness_results['title'] = title
                                predicted_usefulness_results['url'] = url
                                json.dump(predicted_usefulness_results, f, indent=2)
                            
                            # Check if any scores were (4 or) 5, because then we should save the full text
                            pred_usefulness = predicted_usefulness_results.values()

                            # TODO: perhaps make this more dynamic
                            if 5 in pred_usefulness or '5' in pred_usefulness or 4 in pred_usefulness or '4' in pred_usefulness:
                            # DEBUG: Just looking at the scraping results
                                with open(f'autoscious_logs/{search_query_file_safe}/sources/full_text/{sanitize_filename(title)}.txt', 'w', encoding='utf-8') as f:
                                    f.write(title + '\n')
                                    f.write(url + '\n')
                                    f.write(text)
                    else:
                        print("URL or text already visited!")
    
    # Extract facts
    print("****Extract facts****")
    chunk_size = 4000 # How many characters to read per chunk in website text
    overlap = 25 # How much overlap between chunks
    MAX_TOKENS = 40000 # 10K -- Roughly 3K tokens, $0.10 per MAX_TOKENs, max tokens to read per key question -- 40K tokens is $0.40

    # 2) create facts folder and subfolders and only run if the facts folder doesn't exist
    facts_folder_path = f'autoscious_logs/{search_query_file_safe}/facts'
    if not os.path.exists(facts_folder_path):
        os.makedirs(facts_folder_path)

        with open(f'autoscious_logs/{search_query_file_safe}/sources/keywords_query.txt', 'r') as f:
            keywords_query = f.read().strip('"')

        # This method can partially extract the answer, but not the exact table passage which is critical NOR does it prioritize the list of facts. 
        # NOTE: Looks at top 10000 chars from top BM25 ranked chunks!
        # Go through each full text rated highly to extract facts from
        full_text_folder_path = f'autoscious_logs/{search_query_file_safe}/sources/full_text'

        # Loop through every file in the directory, just goes in order
        for filename in os.listdir(full_text_folder_path):
            curr_tokens = 0
            if filename.endswith('.txt'):
                file_path = os.path.join(full_text_folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    title = file.readline().strip()
                    url = file.readline().strip()
                    text = file.read()
                
                # Break the text into chunk to extract information from
                chunks = chunk_text(text, keywords_query, chunk_size, overlap)
                for i, chunk in enumerate(chunks):
                    print(f"Chunk: {i} / {len(chunks)}")
                    extract_facts_from_website_text(search_query_file_safe, search_query, title, chunk, url)

                    curr_tokens += len(chunk)
                    if curr_tokens > MAX_TOKENS:
                        print("Max tokens reached in chunks!")
                        break
    
    # Rerank facts.txt file based on relevancy
    print("****Rerank facts.txt file based on relevancy****")
    file_name = f'autoscious_logs/{search_query_file_safe}/facts/facts.txt'

    with open(file_name, 'r', encoding='utf-8') as f:
        facts_list = f.read()

    reranked_facts_list = json.loads(chat_openai(get_rerank_facts_list_prompt(search_query, facts_list), model="gpt-3.5-turbo")[0])
    print(reranked_facts_list)

    file_name = f'autoscious_logs/{search_query_file_safe}/facts/facts_reranked.txt'

    with open(file_name, 'w', encoding='utf-8') as f:
        for answer in reranked_facts_list:
            f.write(answer + os.linesep)

    # What if we just return the reranked facts instead of the evaluation?
    # Improvement: we can also include the code for the evaluation too if the agent doesn't work well with the facts list
    # Improvemnet: Add years, source urls that the agent can use
    return reranked_facts_list