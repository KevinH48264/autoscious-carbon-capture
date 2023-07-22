from flask import Flask, request, jsonify

from sentence_transformers import SentenceTransformer, util
import json
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the Sentence Transformer model
model = SentenceTransformer('allenai-specter')

# Sample document embeddings data (Replace this with your actual document embeddings)
dataset_file = 'C:/Users/91787/PycharmProjects/autoscious-carbon-capture/react-app/public/2000/gpt_classified_embed_hf.json'

with open(dataset_file) as fIn:
  papers = json.load(fIn)

embeddings_list=[]
for row in papers:
  embeddings_list.append(row["corpus_embed"])
corpus_embeddings = torch.tensor(embeddings_list)
@app.route("/", methods=["GET", "POST"])

@app.route('/search', methods=['GEY'])
def search():
    if request.method == "GET":
        data = request.get_json()
        query = data['query']

        # Calculate the query embedding
        query_embedding = model.encode(query, convert_to_tensor=True)

        # Perform the semantic search with the query embedding and your document embeddings
        search_hits = util.semantic_search(query_embedding, corpus_embeddings)
        search_hits = search_hits[0]  # Get the hits for the first query

        # Prepare the search results and return as JSON
        results = []
        for hit in search_hits[:10]:
            related_paper = papers[hit['corpus_id']]
            result = {
                'title': related_paper['title'],
                'score': hit['score']
            }
            results.append(result)

        return jsonify(results)
    else:

        return jsonify({"error": "Method not allowed"}), 405

if __name__ == '__main__':
    app.run()
