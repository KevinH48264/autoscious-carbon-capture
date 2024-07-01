from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from fastapi.middleware.cors import CORSMiddleware
import torch
from pathlib import Path
import json
from rank_bm25 import BM25Okapi
from fastapi.responses import JSONResponse

app = FastAPI()
origins = [
    "http://localhost:3000",
    "localhost:3000"
]


app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
)

@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Search connection works."}

class SearchQuery(BaseModel):
    query: str


@app.post("/search")
async def search(data: SearchQuery):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        def bm25_ranking(query, corpus):
            tokenized_corpus = [doc.split(" ") for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.split(" ")
            doc_scores = bm25.get_scores(tokenized_query)
            sorted_indices = sorted(range(len(doc_scores)), key=lambda k: doc_scores[k], reverse=True)
            return sorted_indices

        model = SentenceTransformer('allenai-specter').to(device)
        current_folder = Path(__file__).parent
        dataset_file = current_folder.parent / "data" / "gpt_classified_embed_hf.json"
        with open(dataset_file) as fIn:
            papers = json.load(fIn)

        embeddings_list = []
        titles=[]
        abstracts=[]
        for row in papers:
            embeddings_list.append(row["corpus_embed"])
            titles.append(row["title"])
            abstracts.append(row["abstract"])
        corpus_embeddings = torch.tensor(embeddings_list).to(device)
        combined_corpus = [title + " " + (abstract or "") for title, abstract in zip(titles, abstracts)]

        query = data.query
        print(query)
        bm25_ranked_indices = bm25_ranking(query, combined_corpus)
        top_100_indices = bm25_ranked_indices[:100]

        top_100_embeddings = corpus_embeddings[top_100_indices].to(device)
        top_100_titles = [titles[i] for i in top_100_indices]
        top_100_abstracts = [abstracts[i] for i in top_100_indices]
        print("bm2 docs\n")
        print(top_100_titles[0:10])
        # allenai query embedding
        query_embedding = model.encode(query, convert_to_tensor=True).to(device)

        # semantic search_utils
        #search_hits = util.semantic_search(query_embedding, corpus_embeddings)
        #search_hits = search_hits[0]

        cos_scores = util.pytorch_cos_sim(query_embedding, top_100_embeddings)[0]
        reranked_indices = [{"idx": top_100_indices[i], "score": cos_scores[i].tolist()} for i in
                            sorted(range(len(cos_scores)), key=lambda k: cos_scores[k], reverse=True)]

        # top 10
        results = []
        for hit in reranked_indices[:20]:
            related_paper = papers[hit['idx']]
            result = {
                'id': related_paper['id'],
                'title': related_paper['title'],
                'score': hit['score']
            }
            results.append(result)
            print(results)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
