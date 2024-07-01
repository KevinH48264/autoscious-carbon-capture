import requests
import json

def search_semantic_scholar(query, offset=0, limit=10, fields=[]):
    base_url = "http://api.semanticscholar.org/graph/v1/paper/search"
    
    # URL parameters
    params = {
        'query': query,
        'offset': offset,
        'limit': limit,
        'fields': ','.join(fields),
    }
    
    # Send GET request
    response = requests.get(base_url, params=params)
    
    # Raise an exception if the GET request fails
    response.raise_for_status()
    
    return response.json()

# Call the function, fields found here: https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/post_graph_get_papers
result = search_semantic_scholar("carbon capture", offset=10, limit=100, fields=["title", "citationCount", "abstract", "url", "year", "isOpenAccess", "fieldsOfStudy", "s2FieldsOfStudy", "tldr", "citations", "references", "embedding"]) 

# Convert to Latent Lab keys
def preprocess_latent_lab(data):
    for entry in data:
        if 'abstract' in entry:
            entry['description_cleaned'] = entry.pop('abstract')
        if 'year' in entry:
            entry['created'] = entry.pop('year')

    # If you need to convert it back to a JSON string:
    new_json_string = json.dumps(data, indent=2)
    return new_json_string

# Open the file in append mode
with open('output_100.json', 'w') as file:
    json_string = json.dumps(result['data'], indent=2)
    # json_string = preprocess_latent_lab(result['data'])
    file.write(json_string + '\n')