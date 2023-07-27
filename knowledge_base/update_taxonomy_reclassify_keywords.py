# Updating taxonomy and extracting and classifying keywords from papers not yet keyword extracted
# TODO: add titles as context to keywords for better classification!

import pandas as pd
import time
from update_taxonomy_util import extract_valid_json_string, extract_taxonomy_and_classification, extract_taxonomy_mapping, update_classification_ids, save_taxonomy_papers_note, load_latest_taxonomy_papers
from prompts import retrieve_update_taxonomy_extract_keywords_prompt, retrieve_taxonomy_mapping_prompt, retrieve_classify_keywords_prompt
import json
from llm import chat_openai
import sys
import os
from datetime import datetime

def clear_log():
    open('log.txt', 'w').close()

def print_and_flush(*args, **kwargs):
    print(*args)  # print to terminal
    with open('log.txt', 'a', encoding='utf-8') as f:
        print(*args, file=f)  # print to log.txt
        f.flush()

def process_keywords():
    clear_log()

    numbered_taxonomy, df = load_latest_taxonomy_papers()

    print_and_flush("# classification not None: ", len(df[df['classification_ids'].notna()]), "# None: ", len(df[df['classification_ids'].isna()]))

    try:
        # Typically 16000 is good for 8K max tokens
        TOTAL_PROMPT_TOKENS = 5000
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

        for iter_idx in range(0, int(NUM_BATCHES)):
            print_and_flush(f"--- ITERATION {iter_idx} / {NUM_BATCHES} ---")

            # only select rows that don't have keyword classification ids yet but have classification_ids (keywords have been extracted)
            subset_cols = df[['paperId', 'classification_ids']]
            subset = subset_cols[subset_cols['classification_ids'].apply(lambda x: type(x) == list)]
            if subset.empty:
                print("subset was all classified!")
                return

            min_idx = subset.index.min()
            print_and_flush("Checking rows starting from", subset.index.min(), " num paper tokens to use: ", TOTAL_PROMPT_TOKENS - len(numbered_taxonomy))
            print_and_flush("df", df['classification_ids'][min_idx-50:min_idx + 50], "numbered_taxonomy", numbered_taxonomy)
            
            # Create dictionary mapping index to paperId and add as many paper keywords up to TOTAL_PROMPT_TOKENS
            index_to_paperId = {i: row['paperId'] for i, (_, row) in enumerate(subset.iterrows())}
            papers = {}
            total_length = 0
            for i, (_, row) in enumerate(subset.iterrows()):
                classification_ids = row['classification_ids']
                keywords = str([item[0] for item in classification_ids])
                # 10K for GPT3.5, 15K for GPT4
                if total_length + len(keywords) > TOTAL_PROMPT_TOKENS - len(numbered_taxonomy): 
                    break 
                papers[i] = keywords
                total_length += len(keywords)
            papers_processed = ""
            for index in papers.keys():
                papers_processed += f"{index} : {papers[index]}\n"

            # Call OpenAI API to update taxonomy and classify papers
            update_taxonomy_prompt = retrieve_classify_keywords_prompt(numbered_taxonomy, papers_processed)
            res = chat_openai(update_taxonomy_prompt)
            print_and_flush("update_taxonomy_relassify_keywords_prompt", update_taxonomy_prompt, "\n chat update taxonomy prompt res: ", res)
            updated_taxonomy, paper_classification = extract_taxonomy_and_classification(res[0])
            print_and_flush("\nupdated taxonomy: ", updated_taxonomy, "\npaper classification: ", paper_classification)

            # Ensure that you update all previously classified papers' classification ids with the new taxonomy
            taxonomy_mapping_prompt = retrieve_taxonomy_mapping_prompt(numbered_taxonomy, updated_taxonomy)
            res = chat_openai(taxonomy_mapping_prompt)
            print_and_flush("Map taxonomies result: ", res[0])
            changed_category_ids = extract_taxonomy_mapping(res[0])
            print_and_flush("changed category ids: ", changed_category_ids)

            # update keyword_classification_ids using index_to_paperId
            for idx, class_ids in paper_classification.items():
                paper_id = index_to_paperId[int(idx)]  # map index back to paperId
                df.loc[df['paperId'] == paper_id, 'classification_ids'] = df.loc[df['paperId'] == paper_id, 'classification_ids'].apply(lambda x: class_ids)
                
            # check and update for any changed paper classification ids because of updated 
            print_and_flush("applying update classifiation ids")
            df['classification_ids'] = df['classification_ids'].apply(update_classification_ids, args=(changed_category_ids,))

            # save the taxonomy and df to a txt and csv file
            save_taxonomy_papers_note(updated_taxonomy, df, f"{df.shape[0]}_{min_idx}_{len(papers.keys())}_reclassify_keywords")
            
            numbered_taxonomy = updated_taxonomy

    except Exception as e:
        print("An error occurred: ", e)

    return

process_keywords()