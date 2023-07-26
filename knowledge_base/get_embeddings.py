# From [Pre-processing - 1]

import pandas as pd
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# NOTE: Only need to edit this with latest papers JSON from knowledge_base/papers folder and model for pre-processing steps 1 and 2
# Load in latest JSON files
papers_json_path = r'C:\Users\1kevi\Desktop\projects\Research\autoscious-carbon-capture\knowledge_base\papers\23-07-25_11935_database_update.json'

tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter")

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

# Function to get SPECTER embedding
def get_specter_embedding(text):
    # Tokenize text
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')

    # Generate embedding
    with torch.no_grad():
        embedding = model(**inputs)[0].mean(dim=1).squeeze()

    # Convert tensor to numpy array
    embedding_np = embedding.numpy()

    return str(embedding_np.tolist())

# Get indices where 'embedding' is None
embedding_isna_indices = df[df['embedding'].isna()].index
max_index = embedding_isna_indices.max()

# Compute SPECTER embeddings for these rows and store in 'embedding' column
now = datetime.now()
date_str = now.strftime('%y-%m-%d')
time_str = now.strftime('%H-%M-%S')
folder_path = f'papers/{date_str}'
n = len(papers.keys())
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for i, idx in enumerate(embedding_isna_indices):
    print(f"idx {idx} / {max_index}")
    df.loc[idx, 'embedding'] = get_specter_embedding(df.loc[idx, 'text'])

    # save to json in papers and date, labeled by time every 1000 iterations
    if (i + 1) % 1000 == 0:
        df.to_json(f'{folder_path}/{time_str}_embeddings_{i}.json')

print("# embedding not None: ", len(df[df['embedding'].notna()]), "# None: ", len(df[df['embedding'].isna()]))