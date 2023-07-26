import json
import pandas as pd
import re
import ast
import torch
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import os

# TODO: copy path of latest papers df and taxonomy
with open(r'C:\Users\1kevi\Desktop\projects\Research\autoscious-carbon-capture\knowledge_base\papers\latest_papers.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

with open(r'C:\Users\1kevi\Desktop\projects\Research\autoscious-carbon-capture\knowledge_base\clusters\latest_taxonomy.txt', 'r') as f:
    numbered_taxonomy = f.read()

# Load pretrained model/tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter")

now = datetime.now()
date_str = now.strftime('%y-%m-%d')
time_str = now.strftime('%H-%M-%S')
if not os.path.exists(f'clusters/{date_str}'):
    os.makedirs(f'clusters/{date_str}')
if not os.path.exists(f'papers/{date_str}'):
    os.makedirs(f'papers/{date_str}')

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

# Add keyword classification confidence scores - 3
for i, (_, row) in enumerate(df.iterrows()):
    if row['classification_ids'] and row['classification_ids'] != 'nan' and (type(row['classification_ids']) == list or type(row['classification_ids']) == str):
        # Using ast.literal_eval to convert the string to a list of lists
        if type(row['classification_ids']) != list:
            classification_ids = ast.literal_eval(row['classification_ids'])
        else:
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
                    updated_classification_ids.append([keywords, item[1], str(score)])
                else:
                    print("ROW ", i, " CLASS ID: ", class_id, " WAS NOT FOUND IN CLASS_ID_TO_NAME")
            
    df.loc[i, 'classification_ids'] = str(updated_classification_ids)

# save the taxonomy and df to a txt and csv file
with open(f'clusters/{date_str}/{time_str}_{df.shape[0]}_add_keyword_class_scores.txt', 'w') as f:
    f.write(numbered_taxonomy)
df.to_json(f'papers/{date_str}/{time_str}_{df.shape[0]}_add_keyword_class_scores.json', orient='records')
df[['title', 'classification_ids']].to_json(f'papers/{date_str}/{time_str}_{df.shape[0]}_add_keyword_class_scores_manual_inspection.json', orient='records', indent=2)

# save to main
with open(f'clusters/latest_taxonomy.txt', 'w') as f:
    f.write(numbered_taxonomy)
df.to_json(f'papers/latest_papers.json', orient='records')