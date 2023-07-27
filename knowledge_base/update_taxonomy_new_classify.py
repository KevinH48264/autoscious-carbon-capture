'''
This file does the bulk of topic modeling. It updates the taxonomy, extracts keywords, and classifies papers. All log outputs are directed to log.txt for debugging.
'''
# Note tokens debug statements can be found in terminal. Ctrl+F "completion"

# Updating taxonomy and extracting and classifying keywords from papers not yet keyword extracted
import pandas as pd
import time
from datetime import datetime
from update_taxonomy_util import extract_valid_json_string, extract_taxonomy_and_classification, extract_taxonomy_mapping, update_classification_ids, load_latest_taxonomy_papers, save_taxonomy_papers_note
from prompts import retrieve_update_taxonomy_extract_keywords_prompt, retrieve_taxonomy_mapping_prompt
import json
import os
from llm import chat_openai
import sys

def clear_log():
    open('log.txt', 'w').close()

def print_and_flush(*args, **kwargs):
    print(*args)  # print to terminal
    with open('log.txt', 'a', encoding='utf-8') as f:
        print(*args, file=f)  # print to log.txt
        f.flush()

def process_papers():
    clear_log()

    numbered_taxonomy, df = load_latest_taxonomy_papers()

    # IMPORTANT: Uncomment this only if you want to clear all classification_ids and restart.
    # df['classification_ids'] = pd.Series(dtype='object')

    print_and_flush("# classification not None: ", len(df[df['classification_ids'].notna()]), "# None: ", len(df[df['classification_ids'].isna()]))

    # Typically 16000 is good for 8K max tokens
    # TOTAL_PROMPT_TOKENS = 5000
        # "prompt_tokens": 1700,
        # "completion_tokens": 1180,
        # "total_tokens": 2880
    # TOTAL_PROMPT_TOKENS = 1000
        # "prompt_tokens": 2782,
        # "completion_tokens": 1315,
        # "total_tokens": 4097
    # TOTAL_PROMPT_TOKENS = 7500 is best (20 prompt papers, 20 paper responses to hit max tokens)
    TOTAL_PROMPT_TOKENS = 7500
    CHARS_PER_TEXT = 250
    NUM_BATCHES = int(TOTAL_PROMPT_TOKENS / CHARS_PER_TEXT) # should be more than enough

    now = datetime.now()
    date_str = now.strftime('%y-%m-%d')
    time_str = now.strftime('%H-%M-%S')
    if not os.path.exists(f'clusters/{date_str}'):
        os.makedirs(f'clusters/{date_str}')
    if not os.path.exists(f'papers/{date_str}'):
        os.makedirs(f'papers/{date_str}')

    df.reset_index(drop=True, inplace=True)
    for iter_idx in range(0, NUM_BATCHES):
        print_and_flush(f"--- ITERATION {iter_idx} / {NUM_BATCHES} ---")
        subset = df.loc[df['classification_ids'].isna(), 'paperId':'text']
        if subset.empty:
            print_and_flush("subset was all classified!")
            return
        min_idx = int(subset.index.min())
        print_and_flush("Checking rows starting from", min_idx, ", num paper tokens to use: ", TOTAL_PROMPT_TOKENS - len(numbered_taxonomy))
        print_and_flush("df: \n", df['classification_ids'][max(min_idx-50, 0):min_idx + 50], "numbered_taxonomy: \n", numbered_taxonomy)
        
        # Create dictionary mapping index to paperId and add as many papers up to TOTAL_PROMPT_TOKENS
        index_to_paperId = {i: row['paperId'] for i, (_, row) in enumerate(subset.iterrows())}
        papers = {}
        total_length = 0
        for i, (_, row) in enumerate(subset.iterrows()):
            text = row['text'][:CHARS_PER_TEXT]
            if total_length + len(text) > TOTAL_PROMPT_TOKENS - len(numbered_taxonomy):
                break 
            papers[i] = text
            total_length += len(text)
        papers_processed = ""
        for index in papers.keys():
            papers_processed += f"{index} : {papers[index]}\n"

        # Call OpenAI API to update taxonomy and classify papers
        update_taxonomy_classify_papers_prompt = retrieve_update_taxonomy_extract_keywords_prompt(numbered_taxonomy, papers_processed)
        res = chat_openai(update_taxonomy_classify_papers_prompt)
        print_and_flush("update_taxonomy_classify_papers_prompt", update_taxonomy_classify_papers_prompt)
        updated_taxonomy, paper_classification = extract_taxonomy_and_classification(res[0])
        print_and_flush("\nupdated taxonomy: ", updated_taxonomy, "\npaper classification: ", paper_classification)

        # Ensure that you update all previously classified papers' classification ids with the new taxonomy
        taxonomy_mapping_prompt = retrieve_taxonomy_mapping_prompt(numbered_taxonomy, updated_taxonomy)
        res = chat_openai(taxonomy_mapping_prompt)  # call to OpenAI API
        print_and_flush("Map taxonomies result: ", res[0])
        changed_category_ids = extract_taxonomy_mapping(res[0])
        print_and_flush("changed category ids: ", changed_category_ids)
    
        # update classification_ids from paper_classification using index_to_paperId
        for idx, class_ids in paper_classification.items():
            paper_id = index_to_paperId[int(idx)]
            df.loc[df['paperId'] == paper_id, 'classification_ids'] = df.loc[df['paperId'] == paper_id, 'classification_ids'].apply(lambda x: class_ids)
            
        # check and update for any changed paper classification ids because of updated taxonomy
        df['classification_ids'] = df['classification_ids'].apply(update_classification_ids, args=(changed_category_ids,))

        # save the taxonomy and df to a txt and csv file
        n = len(papers.keys())
        save_taxonomy_papers_note(updated_taxonomy, df, f"{min_idx}_{n}_update_taxonomy_new_classify")
        
        # TODO: perhaps need to double check this is actually being updated?
        print_and_flush("numbered == updated taxonomy", numbered_taxonomy == updated_taxonomy)
        numbered_taxonomy = updated_taxonomy
        print_and_flush("numbered == updated taxonomy", numbered_taxonomy == updated_taxonomy)

    return

process_papers()