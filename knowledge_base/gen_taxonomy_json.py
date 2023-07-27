# [GPT Topic Labeling - 10]

# TODO: Keep on modularizing from here, and then test the entire workflow on more data.

# TAXONOMY / CLUSTERS
import ast
from update_taxonomy_util import save_taxonomy_papers_note, load_latest_taxonomy_papers

# TAXONOMY / CLUSTERS
from collections import defaultdict
import ast
import json
import os
from datetime import datetime
import pandas as pd

def generate_taxonomy_nested():
    taxonomy_str, df = load_latest_taxonomy_papers()

    # Create a dictionary where each key is a classification_id and
    # each value is a list of dictionaries containing paper data.
    class_children = defaultdict(list)

    # Iterate over the DataFrame once
    for i, row in df.iterrows():
        # Skip rows with no classification_ids
        if not row['classification_ids']:
            continue

        # Ensure classifications_ids is a list
        classification_ids = row['classification_ids']
        if type(classification_ids) == str:
            classification_ids_list = ast.literal_eval(classification_ids)
        else:
            classification_ids_list = classification_ids

        for keyword_idx, id_info in enumerate(classification_ids_list):
            # Skip malformed id_info
            if len(id_info) != 3:
                continue

            # Extract data from id_info
            keywords = id_info[0]
            paper_classification_id = id_info[1]
            confidence_score = id_info[2]

            # Add the paper data to the corresponding classification_id in class_children
            paper_data = {
                "name": str(row['paperId']) + "-" + str(keyword_idx), 
                "value": [{
                    "paperId": row['paperId'] if pd.notna(row['paperId']) else None, 
                    "title": row['title'] if pd.notna(row['title']) else None, 
                    "abstract": row['abstract'] if pd.notna(row['abstract']) else None,
                    "authors": [[item if pd.notna(item) else None for item in sublist] for sublist in row['authors']] if row['authors'] is not None else None,
                    "citationCount": row['citationCount'] if pd.notna(row['citationCount']) else None,
                    "doi": row['doi'] if pd.notna(row['doi']) else None,
                    "isOpenAccess": row['isOpenAccess'] if pd.notna(row['isOpenAccess']) else None,
                    "language": row['language'] if pd.notna(row['language']) else None,
                    "publicationDate": row['publication_date'] if pd.notna(row['publication_date']) else None,
                    "relevance_score": row["relevance_score"] if pd.notna(row["relevance_score"]) else None,
                    "url": row["url"] if pd.notna(row["url"]) else None,
                    "year": row["year"] if pd.notna(row["year"]) else None,
                    "tsne_x": row["x"] if pd.notna(row["x"]) else None,
                    "tsne_y": row["y"] if pd.notna(row["y"]) else None,
                    "keywords": keywords if keywords else None, 
                    "score": confidence_score if confidence_score else None
                }]
            }
            class_children[paper_classification_id].append(paper_data)

    # Now you can use your existing code to generate the taxonomy tree,
    # but replace the paper data generation part with a lookup in class_children.
    # Here is a rough example:

    print("class_children", class_children.keys())

    stack = []
    taxonomy_json = []
    lines = taxonomy_str.split('\n')
    id_counter = 0  # Keep track of unique id

    for line in lines:
        if line:
            category_id, category_name = line.strip().split(' ', 1)
            category_id = category_id.rstrip('.')
            layer = category_id.count('.')

            category_obj = {
                'id': id_counter,
                'classification_id': category_id,
                'name': category_name,
                'layer': layer,
                'children': class_children[category_id]
            }
            id_counter += 1

            if not stack:
                taxonomy_json.append(category_obj)
            else:
                while stack and stack[-1]['layer'] >= layer:
                    stack.pop()
                if not stack:
                    taxonomy_json.append(category_obj)
                else:
                    stack[-1]['children'].append(category_obj)
            stack.append(category_obj)

    # Write the taxonomy JSON to a file
    preprocessed_taxonomy_json = [{
        "name": "Carbon capture",
        "children": taxonomy_json
    }]

    now = datetime.now()
    date_str = now.strftime('%y-%m-%d')
    time_str = now.strftime('%H-%M-%S')

    with open(f'clusters/{date_str}/{time_str}_taxonomy_gen_taxonomy_json.json', 'w') as f:
        json.dump(preprocessed_taxonomy_json, f, indent=4)

    # save to main
    with open(f'clusters/latest_taxonomy.json', 'w') as f:
        json.dump(preprocessed_taxonomy_json, f, indent=4)

    # save checkpoint
    save_taxonomy_papers_note(taxonomy_str, df, "gen_taxonomy_json")

    return    

generate_taxonomy_nested()