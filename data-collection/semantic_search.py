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

# Call the function
result = search_semantic_scholar("carbon capture", offset=10, limit=10, fields=["title", "citationCount", "abstract", "url", "year", "isOpenAccess", "fieldsOfStudy", "embedding", "tldr"])

# Open the file in append mode
with open('output.txt', 'w') as file:
    json_string = json.dumps(result['data'], indent=2)
    file.write(json_string + '\n')