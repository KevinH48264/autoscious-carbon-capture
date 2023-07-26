'''This creates text column for entries without text and embeddings for rows without embeddings yet. CTRL+C whenever you want to stop generating embeddings. Saves happen every 500 papers.'''

# From [Pre-processing - 1]

import pandas as pd
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import os

MODEL = "allenai/specter"
MODEL_NAME_SAVE_SAFE = MODEL.replace("/", "_")  # replace the slash with underscore


# Function to get SPECTER embedding
def get_specter_embedding(text, tokenizer, model):
    # Tokenize text
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')

    # Generate embedding
    with torch.no_grad():
        embedding = model(**inputs)[0].mean(dim=1).squeeze()

    # Convert tensor to numpy array
    embedding_np = embedding.numpy()

    return str(embedding_np.tolist())

def get_embeddings():
    # Load in existing knowledge base from papers or custom papers JSON
    papers_json_path = 'papers/latest_papers.json'

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)

    # load the data from your JSON file
    with open(papers_json_path, 'r') as f:
        data = json.load(f)

    # convert the data into a pandas DataFrame
    df = pd.DataFrame(data)

    print("Before calculating text, # text not None: ", len(df[df['text'].notna()]), "# None: ", len(df[df['text'].isna()]))

    # Define the conditions for the 'text' column
    condition = df['text'].isna()

    # Create a subset dataframe where 'text' is NaN
    subset_df = df[condition]

    # Define what the 'text' column should be if the condition is met
    value_when_true = 'Title: ' + subset_df['title'].astype(str) + '. Abstract: ' + subset_df['abstract'].astype(str)

    # Define what the 'text' column should be if the abstract is na
    value_when_abstract_na = 'Title: ' + subset_df['title'].astype(str) + '.'

    # Apply the conditions to the DataFrame
    df.loc[condition, 'text'] = np.where(subset_df['abstract'].isna(), value_when_abstract_na, value_when_true)

    print("After calculating text, # text not None: ", len(df[df['text'].notna()]), "# None: ", len(df[df['text'].isna()]))


    print("Before calculating embedding, # embedding not None: ", len(df[df['embedding'].notna()]), "# None: ", len(df[df['embedding'].isna()]))

    # Get indices where 'embedding' is None
    embedding_isna_indices = df[df['embedding'].isna()].index
    max_index = embedding_isna_indices.max()

    # Compute SPECTER embeddings for these rows and store in 'embedding' column
    now = datetime.now()
    date_str = now.strftime('%y-%m-%d')
    time_str = now.strftime('%H-%M-%S')
    folder_path = f'papers/{date_str}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, idx in enumerate(embedding_isna_indices):
        print(f"i {i} idx {idx} / {max_index}")
        df.loc[idx, 'embedding'] = get_specter_embedding(df.loc[idx, 'text'], tokenizer, model)

        # save to json in papers and date, labeled by time every 500 iterations
        if (i + 1) % 500 == 0:
            df.to_json(f'{folder_path}/{time_str}_{df.shape[0]}_get_embeddings_{i}_{MODEL_NAME_SAVE_SAFE}.json', indent=2)

            # save to main
            df.to_json(papers_json_path)

    print("# embedding not None: ", len(df[df['embedding'].notna()]), "# None: ", len(df[df['embedding'].isna()]))

get_embeddings()