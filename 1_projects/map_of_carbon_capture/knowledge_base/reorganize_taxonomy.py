from prompts import retrieve_organize_taxonomy, retrieve_taxonomy_mapping_prompt
import json
import time
from datetime import datetime
import os
import pandas as pd
from llm import chat_openai
from update_taxonomy_util import extract_valid_json_string, extract_taxonomy_and_classification, extract_taxonomy_mapping, update_classification_ids, load_latest_taxonomy_papers, save_taxonomy_papers_note

def reorganize_taxonomy():
    numbered_taxonomy, df = load_latest_taxonomy_papers()

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
        df = update_classification_ids(changed_category_ids, df)

        # save the taxonomy and df to a txt and csv file
        save_taxonomy_papers_note(updated_taxonomy, df, "reorganize_taxonomy")

    except Exception as e:
        print("An error occurred: ", e)

    return

reorganize_taxonomy()