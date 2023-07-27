'''
This script 1) retrieves the top num_papers papers from the OpenAlex database for a given search term search_term, 2) ensure that extracted info about ids are not in the latest knowledge base existing_complete_papers_json_path, and 3) exports all new paper ids to openalex/date/time_size_search_term_num_papers.json for checkpointing and openalex/latest_unique_papers_downloaded.json for combining
'''

import pyalex
from pyalex import Works
import os
from datetime import datetime
import json
import pandas as pd

# Arguments
pyalex.config.email = "1kevin.huang@gmail.com"
search_term = "Enoyl-CoA carboxylase/reductase enzymes"
SEARCH_TERM_SAVE_SAFE = search_term.replace("/", "_")  # replace the slash with underscore
num_papers = 200
num_papers_per_page = 200 # max = 200
existing_complete_papers_json_path = '../knowledge_base/papers/latest_papers.json'
disregard_existing_papers = False # IMPORTANT: only if you want to replace existing papers in the knowledge base

# Search by relevance for search term
results_pager = Works().search(search_term) \
    .paginate(per_page=num_papers_per_page)

top_results = []
for i, page in enumerate(results_pager):
    top_results += page
    print("retrieving # results: ", len(page), "total results retrieved: ", len(top_results))

    if len(top_results) >= num_papers:
        break

# Get today's date
now = datetime.now()
date_str = now.strftime('%y-%m-%d')
time_str = now.strftime('%H-%M-%S')
folder_path = "openalex/" + date_str
raw_file_name = f'raw_output_relevance_{len(top_results)}.json'

# Check if the folder path exists, create it if it doesn't
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Open the file in append mode # saving space by commenting this out.
# with open(os.path.join(folder_path, raw_file_name), 'w') as file:
#     json.dump(top_results, file, indent=2)


'''
Extracting the desired columns, 8 min for 2000 results
'''
# load the data from your JSON file
with open(existing_complete_papers_json_path, 'r') as f:
    data = json.load(f)
existing_df = pd.DataFrame(data)
existing_paper_ids_set = set(existing_df['id'])
print("Loading in existing knowledge base of size: ", len(existing_paper_ids_set))

# Create an empty list to store the extracted results
extracted_data = []

# Iterate over the results and extract the desired columns
print("Iterating over results!")
for i, result in enumerate(top_results):
    print("Checking results from paper index: ", i, "paper id: ", result["id"])
    # Only extract results not already in existing paper id set
    if (result["id"] not in existing_paper_ids_set) or disregard_existing_papers:
        try:
            extracted_result = {
                "id": result["id"],
                "doi": result["doi"],
                "title": result["display_name"],
                "isOpenAccess": result["primary_location"]["is_oa"],
                "abstract": Works()[result["id"].split("https://openalex.org/")[1]]["abstract"],
                "paperId": result["id"],
                "url": result["primary_location"].get("landing_page_url", ""),
                "citations": result.get("referenced_works", ""),
                "citationCount": result.get("cited_by_count", 0),
                "concepts": result.get("concepts", []),
                "publication_date": result.get("publication_date", ""),
                "relevance_score": result.get("relevance_score", ""),
                "language": result.get("language", ""),
                "year": result.get("publication_year", ""),
                "search_score": [[search_term, result.get("relevance_score", "")]]
            }
            
            try:
                extracted_result["authors"] = [[author["author"]["id"], author["author"]["display_name"]] for author in result["authorships"]]
            except (KeyError, TypeError):
                extracted_result["authors"] = []

            extracted_data.append(extracted_result)
        except Exception as e:
            print(f"Error processing result {i}: {e}")

    # Update "search_score" because maybe a new search query also resulted in this paper with a relevance score, which should appear in search_score because I want to classify all, even if some papers were added prev but not classified yet
    elif result["id"] in existing_paper_ids_set:
        # Select the matching row
        matching_row = existing_df.loc[existing_df['id'] == result["id"]]
        matching_row_search_score = matching_row['search_score'].values[0]

        # Check if 'search_score' is NaN or empty string
        if not matching_row_search_score:
            # If 'search_score' is NaN or "", replace it with new search score
            idx = existing_df.index[existing_df['id'] == result["id"]][0]
            existing_df.at[idx, 'search_score'] = [search_term, result.get("relevance_score", "")]
            
        elif isinstance(matching_row_search_score, list):
            # Extract all search terms in 'search_score'
            existing_search_terms = [item[0] for item in matching_row_search_score]

            # Check if 'search_term' is not in existing search terms
            if search_term not in existing_search_terms:
                # Append new search score
                new_search_score = matching_row_search_score + [[search_term, result.get("relevance_score", "")]]
                idx = existing_df.index[existing_df['id'] == result["id"]][0]
                existing_df.at[idx, 'search_score'] = new_search_score


# Write the list of extracted results to the JSON file
output_file_path = folder_path + f"/{time_str}_{len(extracted_data)}_new_papers_relevance_ranked_{SEARCH_TERM_SAVE_SAFE}_{num_papers}.json"

with open(output_file_path, "w") as output_file:
    json.dump(extracted_data, output_file)
with open('openalex/latest_unique_papers_downloaded.json', "w") as output_file:
    json.dump(extracted_data, output_file)

# Update latest papers
now = datetime.now()
date_str = now.strftime('%y-%m-%d')
time_str = now.strftime('%H-%M-%S')
if not os.path.exists(f'../knowledge_base/papers/{date_str}'):
    os.makedirs(f'../knowledge_base/papers/{date_str}')

# save the taxonomy and df to a txt and csv file
note = f"data_collection_retrieve_data_update_search_score_{SEARCH_TERM_SAVE_SAFE}"
existing_df.to_json(f'../knowledge_base/papers/{date_str}/{time_str}_{existing_df.shape[0]}_{note}.json', orient='records')
existing_df[['title', 'classification_ids', 'search_score']].to_json(f'../knowledge_base/papers/{date_str}/{time_str}_{existing_df.shape[0]}_{note}_manual_inspection.json', orient='records', indent=2)
# save to main
existing_df.to_json(f'../knowledge_base/papers/latest_papers.json', orient='records')

print("Completed extracting results! # of new results: ", len(extracted_data))