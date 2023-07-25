import pyalex
from pyalex import Works
import os
from datetime import date
import json
import pandas as pd

# Arguments
pyalex.config.email = "1kevin.huang@gmail.com"
search_term = "carbon capture"
num_papers = 200
existing_complete_papers_json_path = r'C:\Users\1kevi\Desktop\projects\Research\autoscious-carbon-capture\knowledge_base\papers\23-07-25_11935_database_update.json'
disregard_existing_papers = False # If you want to replace existing papers in the knowledge base

# Search by relevance for search term
results_pager = Works().search(search_term) \
    .paginate(per_page=200)

top_results = []
for i, page in enumerate(results_pager):
    top_results += page
    print("retrieving # results: ", len(page), "total results retrieved: ", len(top_results))

    if len(top_results) >= num_papers:
        break

# Get today's date
today = date.today()
date_string = today.strftime('%y-%m-%d')
folder_path = "openalex/" + date_string
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
    # Only extract results not already in existing paper id set
    if (result["id"] not in existing_paper_ids_set) or disregard_existing_papers:
        print("Extracting results from paper index: ", i, "paper id: ", result["id"])
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
            }
            
            try:
                extracted_result["authors"] = [[author["author"]["id"], author["author"]["display_name"]] for author in result["authorships"]]
            except (KeyError, TypeError):
                extracted_result["authors"] = []

            extracted_data.append(extracted_result)
        except Exception as e:
            print(f"Error processing result {i}: {e}")

# Write the list of extracted results to the JSON file
output_file_path = folder_path + f"/extracted_output_relevance_{len(extracted_data)}.json"

with open(output_file_path, "w") as output_file:
    json.dump(extracted_data, output_file)

print("Completed extracting results! # of new results: ", len(extracted_data))