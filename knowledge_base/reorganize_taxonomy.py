from prompts import retrieve_organize_taxonomy, retrieve_taxonomy_mapping_prompt
import json
import time
from datetime import datetime
import os
import pandas as pd
from llm import chat_openai

# TODO: copy path of latest papers df and taxonomy
with open(r'C:\Users\1kevi\Desktop\projects\Research\autoscious-carbon-capture\knowledge_base\papers\23-07-25\23-12-00_11935_reorganize_taxonomy.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

with open(r'C:\Users\1kevi\Desktop\projects\Research\autoscious-carbon-capture\knowledge_base\clusters\23-07-25\23-12-00_11935_reorganize_taxonomy.txt', 'r') as f:
    numbered_taxonomy = f.read()

def reorganize_taxonomy(df, numbered_taxonomy):
    now = datetime.now()
    date_str = now.strftime('%y-%m-%d')
    time_str = now.strftime('%H-%M-%S')
    if not os.path.exists(f'clusters/{date_str}'):
        os.makedirs(f'clusters/{date_str}')

    try:
        update_taxonomy_prompt = retrieve_organize_taxonomy(numbered_taxonomy)
        print("update_taxonomy_prompt", update_taxonomy_prompt)

        res = chat_openai(update_taxonomy_prompt)  # call to OpenAI API
        print("Reorganized taxonomy result: ", res[0])
        
        # parse the res[0]
        updated_taxonomy = ""
        for line in res[0].splitlines():
            if len(line.strip()) > 2 and line.strip()[1] == ".":
                updated_taxonomy += line.strip() + "\n"
        print("updated taxonomy: ", updated_taxonomy)

        # Ensure that you update all previously classified papers' classification ids with the new taxonomy
        print("MAPPING TAXONOMIES")
        taxonomy_mapping_prompt = retrieve_taxonomy_mapping_prompt(numbered_taxonomy, updated_taxonomy)
        res = chat_openai(taxonomy_mapping_prompt)  # call to OpenAI API
        
        print("Map taxonomies result: ", res[0])
        changed_category_ids = extract_taxonomy_mapping(res[0])
        print("changed category ids: ", changed_category_ids)
            
        # check and update for any changed paper classification ids
        df['keyword_classification_ids'] = df['keyword_classification_ids'].apply(update_classification_ids, args=(changed_category_ids,))

        # save the taxonomy and df to a txt and csv file
        with open(f'clusters/{date_str}/{time_str}_{df.shape[0]}_reorganize_taxonomy.txt', 'w') as f:
            f.write(updated_taxonomy)
        df.to_json(f'papers/{date_str}/{time_str}_{df.shape[0]}_reorganize_taxonomy.json', orient='records')
        df[['title', 'classification_ids']].to_json(f'papers/{date_str}/{time_str}_{df.shape[0]}_reorganize_taxonomy.json', orient='records', indent=2)

        # save to main
        with open(f'clusters/latest_taxonomy.txt', 'w') as f:
            f.write(updated_taxonomy)
        df.to_json(f'papers/latest_papers.json', orient='records')
    except Exception as e:
        print("An error occurred: ", e)

    return df, numbered_taxonomy

reorganize_taxonomy(df, numbered_taxonomy)