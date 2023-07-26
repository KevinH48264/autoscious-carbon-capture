# Updating taxonomy and extracting and classifying keywords from papers not yet keyword extracted
import pandas as pd
import time
from datetime import datetime
from update_taxonomy_util import extract_valid_json_string, extract_taxonomy_and_classification, extract_taxonomy_mapping, update_classification_ids
from prompts import retrieve_update_taxonomy_extract_keywords_prompt, retrieve_taxonomy_mapping_prompt
import json
import os
from llm import chat_openai

# TODO: copy path of latest papers df and taxonomy and edit TOTAL_PROMPT_TOKEN appropriately
with open(r'C:\Users\1kevi\Desktop\projects\Research\autoscious-carbon-capture\knowledge_base\papers\23-07-25\22-15-02_11935_2_3.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

with open(r'C:\Users\1kevi\Desktop\projects\Research\autoscious-carbon-capture\knowledge_base\clusters\23-07-25\22-15-02_11935_2_3.txt', 'r') as f:
    numbered_taxonomy = f.read()

print("# classification not None: ", len(df[df['classification_ids'].notna()]), "# None: ", len(df[df['classification_ids'].isna()]))

def process_papers(df, numbered_taxonomy):
    # Typically 16000 is good for 8K max tokens
    TOTAL_PROMPT_TOKENS = 5000
    CHARS_PER_TEXT = 250
    NUM_BATCHES = TOTAL_PROMPT_TOKENS / CHARS_PER_TEXT # should be more than enough

    now = datetime.now()
    date_str = now.strftime('%y-%m-%d')
    time_str = now.strftime('%H-%M-%S')
    if not os.path.exists(f'clusters/{date_str}'):
        os.makedirs(f'clusters/{date_str}')
    if not os.path.exists(f'papers/{date_str}'):
        os.makedirs(f'papers/{date_str}')

    for i in range(0, int(NUM_BATCHES)):
        print(f"--- ITERATION {i} ---")
        subset = df.loc[df['classification_ids'].isna(), 'paperId':'text']
        min_idx = subset.index.min()
        if subset.empty:
            print("subset was all classified!")
            return
        print("Checking rows starting from", subset.index.min(), " num paper tokens to use: ", TOTAL_PROMPT_TOKENS - len(numbered_taxonomy))
        print("df", df['classification_ids'][min_idx-50:min_idx + 50], "numbered_taxonomy", numbered_taxonomy)
        
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
        updated_taxonomy, paper_classification = extract_taxonomy_and_classification(res[0])
        print("updated taxonomy: ", updated_taxonomy, "paper classification: ", paper_classification)

        # Ensure that you update all previously classified papers' classification ids with the new taxonomy
        taxonomy_mapping_prompt = retrieve_taxonomy_mapping_prompt(numbered_taxonomy, updated_taxonomy)
        res = chat_openai(taxonomy_mapping_prompt)  # call to OpenAI API
        print("Map taxonomies result: ", res[0])
        changed_category_ids = extract_taxonomy_mapping(res[0])
        print("changed category ids: ", changed_category_ids)
    
        # update classification_ids from paper_classification using index_to_paperId
        for idx, class_ids in paper_classification.items():
            paper_id = index_to_paperId[int(idx)]
            df.loc[df['paperId'] == paper_id, 'classification_ids'] = df.loc[df['paperId'] == paper_id, 'classification_ids'].apply(lambda x: class_ids)
            
        # check and update for any changed paper classification ids because of updated taxonomy
        df['classification_ids'] = df['classification_ids'].apply(update_classification_ids, args=(changed_category_ids,))

        # save the taxonomy and df to a txt and csv file
        n = len(papers.keys())

        with open(f'clusters/{date_str}/{time_str}_{df.shape[0]}_{min_idx}_{n}.txt', 'w') as f:
            f.write(updated_taxonomy)
        df.to_json(f'papers/{date_str}/{time_str}_{df.shape[0]}_{min_idx}_{n}.json', orient='records')
        df[['title', 'classification_ids']].to_json(f'papers/{date_str}/{time_str}_{df.shape[0]}_{min_idx}_{n}_manual_analysis.json', orient='records', indent=2)
        
        numbered_taxonomy = updated_taxonomy

    return df, numbered_taxonomy

df, numbered_taxonomy = process_papers(df, numbered_taxonomy)