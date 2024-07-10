'''
This file adds confidence score to the keyword and classification name based on cosine similarity from the Allen AI SPECTER model. 
'''

import json
import pandas as pd
import re
import ast
import torch
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from update_taxonomy_util import load_latest_taxonomy_papers, save_taxonomy_papers_note
import os

# TODO (optional to load in custom data): copy path of latest papers df and taxonomy
# with open(r'C:\Users\1kevi\Desktop\projects\Research\autoscious-carbon-capture\knowledge_base\papers\23-07-26\19-55-08_11935_11935_0_36_reclassify_keywords.json', 'r') as f:
#     data = json.load(f)
# df = pd.DataFrame(data)

# with open(r'C:\Users\1kevi\Desktop\projects\Research\autoscious-carbon-capture\knowledge_base\clusters\23-07-26\19-55-08_11935_11935_0_36_reclassify_keywords.txt', 'r') as f:
#     numbered_taxonomy = f.read()

numbered_taxonomy, df = load_latest_taxonomy_papers()

# Load pretrained model/tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter")

# Add keyword classification confidence scores - 1
# create class_id_to_name dictionary
class_id_to_name = {}
for line in numbered_taxonomy.split("\n"):
    split_line = line.strip().split(maxsplit=1)
    if len(split_line) == 2:
        if split_line[0][-1] == ".": 
            class_id_to_name[split_line[0][:-1]] = split_line[1]
        else:
            class_id_to_name[split_line[0]] = split_line[1]

print("class_id_to_name:", class_id_to_name)

def get_cosine_similarity(text1, text2):
    # Tokenize texts
    inputs1 = tokenizer(text1, padding=True, truncation=True, max_length=512, return_tensors='pt')
    inputs2 = tokenizer(text2, padding=True, truncation=True, max_length=512, return_tensors='pt')

    # Generate embeddings
    with torch.no_grad():
        embedding1 = model(**inputs1)[0].mean(dim=1).squeeze()
        embedding2 = model(**inputs2)[0].mean(dim=1).squeeze()

    # Compute cosine similarity
    score = 1 - cosine(embedding1.numpy(), embedding2.numpy())
    return score

# Ensure we're only taking rows where there are keywords not yet scored. 
# TODO: Improvement, can add other conditions too (if confidence score is 0 or error handling if format is off)
def filter_rows(lst):
    # Ensure that there's a list
    if not lst:
        return False

    # Take any row where a sublist has length 2 (has yet to be scored)
    for sub in lst:
        if len(sub) == 2:
            return True
        
    return False

# Add keyword classification confidence scores - 3
import itertools

mask = df['classification_ids'].apply(filter_rows)
df_filtered = df.loc[mask].copy()

for count, (index, row) in enumerate(df_filtered.iterrows()):
    print(f"Scoring row {count} / {df_filtered.shape[0]}")
    classification_ids = row['classification_ids']

    updated_classification_ids = []
    for item in classification_ids:
        if item and len(item) == 2:
            keywords = item[0]
            class_id = item[1]
            if class_id in class_id_to_name.keys():
                classification = class_id_to_name[class_id]
                
                # Get cosine similarity score using HuggingFace Semantic Scholar Spectre API embeddings
                score = round(get_cosine_similarity(keywords, classification), 2)
                updated_classification_ids.append([keywords, item[1], score])
            else:
                print(f"ROW {index} CLASS ID: {class_id} WAS NOT FOUND IN CLASS_ID_TO_NAME")
                updated_classification_ids.append([keywords, item[1], -1]) # Just mark it as not found with -1 flag
    
    df_filtered.at[index, 'classification_ids'] = updated_classification_ids
print("finished adding confidence scores!")

# Merge the filtered DataFrame back into the original one
df.update(df_filtered)

# save the taxonomy and df to a txt and csv file
save_taxonomy_papers_note(numbered_taxonomy, df, "add_keyword_class_scores")