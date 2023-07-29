'''This creates text column for entries without text and embeddings for rows without embeddings yet. CTRL+C whenever you want to stop generating embeddings. Saves happen every 500 papers.'''

# From [Pre-processing - 1]

import pandas as pd
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import os
from update_taxonomy_util import load_latest_taxonomy_papers, save_taxonomy_papers_note

MODEL = "allenai/specter"
MODEL_NAME_SAVE_SAFE = MODEL.replace("/", "_")  # replace the slash with underscore
search_term_filter = "Enoyl-CoA carboxylase/reductase enzymes" # "" to just process all rows


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
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)

    # convert the data into a pandas DataFrame
    numbered_taxonomy, df = load_latest_taxonomy_papers(search_term_filter)

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
    for i, idx in enumerate(embedding_isna_indices):
        print(f"i {i} / {len(embedding_isna_indices)}, latest papers row index {idx}")
        df.loc[idx, 'embedding'] = get_specter_embedding(df.loc[idx, 'text'], tokenizer, model)

        # save to json in papers and date, labeled by time every 500 iterations
        if (i + 1) % 500 == 0:
            save_taxonomy_papers_note(numbered_taxonomy, df, f"get_embeddings_{i}_{MODEL_NAME_SAVE_SAFE}", search_term_filter)

    print("# embedding not None: ", len(df[df['embedding'].notna()]), "# None: ", len(df[df['embedding'].isna()]))

    save_taxonomy_papers_note(numbered_taxonomy, df, f"get_embeddings_complete_{MODEL_NAME_SAVE_SAFE}", search_term_filter)

get_embeddings()
