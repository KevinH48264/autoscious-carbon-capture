import pandas as pd
import json
import re
import ast
import numpy as np
from datetime import datetime
import os

def load_latest_taxonomy_papers(search_term_filter="", custom_df_path="", custom_taxonomy_path=""):
    # Uncomment for custom loading
    # with open(r'C:\Users\1kevi\Desktop\projects\Research\autoscious-carbon-capture\knowledge_base\papers\23-07-26\22-35-55_11935_1155_23_update_taxonomy_new_classify.json', 'r') as f:
    #     data = json.load(f)
    # df = pd.DataFrame(data)

    # with open(r'C:\Users\1kevi\Desktop\projects\Research\autoscious-carbon-capture\knowledge_base\clusters\23-07-26\22-35-55_11935_1155_23_update_taxonomy_new_classify.txt', 'r') as f:
    #     numbered_taxonomy = f.read()

    # load the df
    with open(custom_df_path or 'papers/latest_papers.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    with open(custom_taxonomy_path or 'clusters/latest_taxonomy.txt', 'r') as f:
        numbered_taxonomy = f.read()

    # accomodate for search_term_filters
    if search_term_filter:
        # Step 1: Filter out NaN rows
        filtered_df = df[df['search_score'].notna()]

        filtered_search_term_df = filtered_df[filtered_df['search_score'].apply(lambda x: any(search_term_filter in sublist for sublist in x))].copy()

        print("Loading a filtered df and numbered taxonomy for this search", search_term_filter)
        return numbered_taxonomy, filtered_search_term_df
    
    print("Loading a complete df and numbered taxonomy")
    return numbered_taxonomy, df

def save_taxonomy_papers_note(numbered_taxonomy=None, df=None, note="", search_term_filter=""):
    now = datetime.now()
    date_str = now.strftime('%y-%m-%d')
    time_str = now.strftime('%H-%M-%S')
    if not os.path.exists(f'clusters/{date_str}'):
        os.makedirs(f'clusters/{date_str}')
    if not os.path.exists(f'papers/{date_str}'):
        os.makedirs(f'papers/{date_str}')

    # save taxonomy
    if numbered_taxonomy:
        # save to checkpoints
        with open(f'clusters/{date_str}/{time_str}_{df.shape[0]}_{note}.txt', 'w') as f:
            f.write(numbered_taxonomy)
        
        # save to main
        with open(f'clusters/latest_taxonomy.txt', 'w') as f:
            f.write(numbered_taxonomy)

    # save papers df
    if df is not None and not df.empty:
        if search_term_filter:
            # load the df
            with open('papers/latest_papers.json', 'r') as f:
                data = json.load(f)
            existing_complete_df = pd.DataFrame(data)

            # Update the existing dataframe
            existing_complete_df.update(df)

            # Reset the index so it's not just aligned on id but the unique index for other code that requires min index for debugging
            existing_complete_df.reset_index(drop=True, inplace=True)

            df = existing_complete_df

            print("Merged filtered df with latest papers df")

        # save to checkpoints
        df.to_json(f'papers/{date_str}/{time_str}_{df.shape[0]}_{note}.json', orient='records')
        df[['title', 'classification_ids']].to_json(f'papers/{date_str}/{time_str}_{df.shape[0]}_{note}_manual_inspection.json', orient='records', indent=2)

        # save to main
        df.to_json(f'papers/latest_papers.json', orient='records')

    print("Saved df and taxonomy to checkpoint and updated latest files!")

def extract_valid_json_string(json_str):
    print("EXTRACTING VALID JSONS FROM ", json_str)
    closing_brace_indices = [i for i, char in enumerate(json_str) if char == "}"]
    for index in reversed(closing_brace_indices):
        test_str = json_str[:index+1] + "]"
        try:
            json.loads(test_str)
            return test_str
        except json.JSONDecodeError:
            continue
    return None

def extract_taxonomy_and_classification(chat_output):
    # print("THIS IS CHAT OUTPUT IN EXTRACT TAXONOMY AND CLASSIFICATION: ", chat_output)

    # Extracting taxonomy
    taxonomy_start = chat_output.find('UPDATED TAXONOMY:') + len('UPDATED TAXONOMY:')
    taxonomy_end = chat_output.find('PAPER CLASSIFICATION:')
    updated_taxonomy = chat_output[taxonomy_start:taxonomy_end].strip()

    # Extracting paper classifications
    end_index = chat_output.rfind(']')
    classification_str = chat_output[taxonomy_end+len('PAPER CLASSIFICATION:'):end_index+1].strip()

    # Iterate through each line until one ending with ']]' is found
    valid_classification_str = ""
    classification_dict = {}
    for line in classification_str.splitlines():
        print("Line: ", line)
        if line.strip().endswith(']],') or line.strip().endswith(']]'):
            key, value = line.split(":", 1)
            end_line_index = value.rfind(']')
            list_value = ast.literal_eval(value[:end_line_index + 1].strip())
            classification_dict[key.strip().strip('"').strip("'")] = list_value
        # if line.strip().endswith(']],'):
        #     key, value = line.split(":", 1)
        #     end_line_index = value.rfind(']')
        #     classification_dict[key.strip().strip('"')] = value[:end_line_index + 1].strip()
        # elif line.strip().endswith(']]'):
        #     key, value = line.split(":", 1)
        #     end_line_index = value.rfind(']')
        #     classification_dict[key.strip().strip('"')] = value[:end_line_index+1].strip()
    
    print("classification_dict", classification_dict)
    return updated_taxonomy, classification_dict

def extract_taxonomy_mapping(chat_output):
    print("THIS IS CHAT OUTPUT EXTRACT TAXONOMY MAPPING: ", chat_output)

    # Extracting changed category IDs
    changed_category_start = chat_output.find('[')
    changed_category_end = chat_output.rfind(']')
    print("changed_category_start", changed_category_start, "changed_category_end", changed_category_end)
    changed_category_ids_str = chat_output[changed_category_start:changed_category_end+1].strip()
    print("changed_category_ids_str", changed_category_ids_str)

    if changed_category_ids_str and (changed_category_ids_str[0] == '[' and changed_category_ids_str[-1] == ']'):
        changed_category_ids = json.loads(changed_category_ids_str)
        changed_category_ids_dict = {list(d.keys())[0]: list(d.values())[0] for d in changed_category_ids}
    else:
        changed_category_ids_dict = {}

    print("\nchanged changed_category_ids_dict: ", changed_category_ids_dict)
    return changed_category_ids_dict

def update_classification_ids(changed_category_ids, filtered_df, search_term_filter=""):
    numbered_taxonomy, df = load_latest_taxonomy_papers()

    df.update(filtered_df)

    def update_ids(row):
        if isinstance(row, list):
            res = []
            for item in row:
                if len(item) > 1:
                    if item[1]:
                        category_id = item[1].split(' ', 1)[0]
                        if category_id in changed_category_ids:
                            res.append([item[0], changed_category_ids[category_id]])
                        else:
                            print("WARNING: ", category_id, " not found in changed_category_ids_dict, setting item[1] to None")
                            res.append([item[0], None])
            return res
        else:
            return row

    df['classification_ids'] = df['classification_ids'].apply(update_ids)

    save_taxonomy_papers_note(numbered_taxonomy, df, "updated classification ids")

    # Return the filtered df if there was a search term
    _, new_df = load_latest_taxonomy_papers(search_term_filter)
    return new_df