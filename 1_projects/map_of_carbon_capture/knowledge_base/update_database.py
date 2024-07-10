import pandas as pd
import json
from datetime import datetime
import os

# Load in existing knowledge base from papers
papers_json_path = 'papers/latest_papers.json'

# Load in potentially new data from data_collection folder
new_data_json_path = '../data_collection/openalex/latest_unique_papers_downloaded.json'

# load the data from your JSON file
with open(papers_json_path, 'r') as f:
    data = json.load(f)

# convert the data into a pandas DataFrame
existing_df = pd.DataFrame(data)

# load the data from your JSON file
with open(new_data_json_path, 'r') as f:
    new_data = json.load(f)

# convert the data into a pandas DataFrame
new_df = pd.DataFrame(new_data)

'''Combine dfs'''
# Set 'id' as the index in both dataframes if not done already
existing_df.set_index('id', inplace=True)
new_df.set_index('id', inplace=True)

# Combine the two dataframes, updating the values in existing_df with the non-NA ones from new_df and adding new rows from new_df
combined_df = new_df.combine_first(existing_df)

# Reset the index
combined_df.reset_index(inplace=True)

'''Save data'''
now = datetime.now()
date_str = now.strftime('%y-%m-%d')
time_str = now.strftime('%H-%M-%S')
if not os.path.exists(f'papers/{date_str}'):
    os.makedirs(f'papers/{date_str}')

combined_df.to_json(f'papers/{date_str}/{time_str}_{combined_df.shape[0]}_database_update.json', orient='records')

combined_df.to_json(f'papers/{date_str}/{time_str}_{combined_df.shape[0]}_database_update_readable.json', orient='records', indent=2)

print("Successfully combined dfs!")