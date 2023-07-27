# Selecting edges within each classification id group: keeping the paperId's most similar paper in their cluster, and also constructing the MST for each cluster to ensure it's a connected graph
import json
import numpy as np
from scipy.spatial.distance import cosine
from itertools import combinations
from scipy.spatial.distance import euclidean
from collections import defaultdict
import networkx as nx
import os
from datetime import datetime


# 1. Create a dictionary that maps each classification_id to a list of papers in that cluster.
def traverse_clusters(cluster, classification_to_papers):
    values = []

    # Recursively traverse this cluster's child clusters
    for child in cluster['children']:
        if 'value' in child:
            values.append([child['name'], child['value'][0]])
        elif 'children' in child:
            traverse_clusters(child, classification_to_papers)

    # Add this cluster's classification_id and papers to the dictionary
    if 'classification_id' in cluster:
        classification_to_papers[cluster['classification_id']] = values

def generate_edges():
    num_most_similar = 1

    # Open the file and load the JSON
    with open('clusters/latest_taxonomy.json', 'r') as f:
        parsed_taxonomy = json.load(f)

    # Initialize an empty dictionary
    classification_to_papers = {}

    # Traverse each top-level category in the parsed taxonomy
    for category in parsed_taxonomy:
        traverse_clusters(category, classification_to_papers)


    # 2. Iterate through this dictionary, and for each category, compute the cosine similarity between every pair of papers in that category.
    # 3. Store the results in a list of dictionaries, as you've described.

    # A dictionary mapping each paper_id to its list of highest weight edge
    max_edges = defaultdict(list)

    # A dictionary mapping each classification_id to its graph
    classification_to_graph = defaultdict(nx.Graph)

    for classification_id, paper_ids in classification_to_papers.items():
        paper_pairs = list(combinations(paper_ids, 2))
        print("On classification id: ", classification_id, ", # pairs: ", len(paper_pairs))

        for paper_id1_obj, paper_id2_obj in paper_pairs:
            # print("paper_id1", paper_id1_obj[0], "paper_id2", paper_id2_obj[0])

            # Only use paperIds
            paper_id1, paper_id2 = paper_id1_obj[0], paper_id2_obj[0]

            if (paper_id1_obj[1]["tsne_x"] == None or paper_id1_obj[1]["tsne_y"] == None or paper_id2_obj[1]["tsne_x"] == None or paper_id2_obj[1]["tsne_y"] == None):
                # print("skipping because of None tsne_x or tsne_y")
                continue

            x1, y1 = paper_id1_obj[1]["tsne_x"], paper_id1_obj[1]["tsne_y"]
            x2, y2 = paper_id2_obj[1]["tsne_x"], paper_id2_obj[1]["tsne_y"]
            
            distance = euclidean([x1, y1], [x2, y2])
            weight = 1 / (1 + distance)

            # Add the edge to the graph for this classification_id
            classification_to_graph[classification_id].add_edge(paper_id1, paper_id2, weight=weight)

            # Create a new edge
            new_edge = {"source": paper_id1, "target": paper_id2, "weight": weight}
            
            # Add the new edge to the list for paper_id1 if it's one of the top num_most_similar
            if len(max_edges[paper_id1]) < num_most_similar or weight > min(edge['weight'] for edge in max_edges[paper_id1]):
                if len(max_edges[paper_id1]) == num_most_similar:
                    # Remove the edge with the lowest weight
                    max_edges[paper_id1].remove(min(max_edges[paper_id1], key=lambda edge: edge['weight']))
                max_edges[paper_id1].append(new_edge)
            
            # Do the same for paper_id2, but reverse the source and target
            new_edge = {"source": paper_id2, "target": paper_id1, "weight": weight}
            if len(max_edges[paper_id2]) < num_most_similar or weight > min(edge['weight'] for edge in max_edges[paper_id2]):
                if len(max_edges[paper_id2]) == num_most_similar:
                    max_edges[paper_id2].remove(min(max_edges[paper_id2], key=lambda edge: edge['weight']))
                max_edges[paper_id2].append(new_edge)

    # Combine the max weight edges for each node with the MSTs for each classification_id
    final_edges = []

    # Create a set to keep track of edges that have been added
    added_edges = set()

    # Flatten max_edges.values()
    flattened_max_edges = [edge for edges in max_edges.values() for edge in edges]

    # Add an index as the id for each edge in final_edges
    for idx, edge in enumerate(flattened_max_edges):
        edge_tuple = (edge["source"], edge["target"])
        if edge_tuple not in added_edges:
            # Add the edge id first so it appears first
            edge_with_id = {"id": idx}
            edge_with_id.update(edge)
            final_edges.append(edge_with_id)
            added_edges.add(edge_tuple)

    # Now add the MST edges, continuing the ids from where they left off
    # for graph in classification_to_graph.values():
    #     mst_edges = nx.algorithms.tree.maximum_spanning_edges(graph, data=False)
    #     for source, target in mst_edges:
    #         edge_tuple = (source, target)
    #         if edge_tuple not in added_edges:
    #             weight = graph[source][target]['weight']
    #             final_edges.append({
    #                 "id": len(final_edges),  # The next id is the current length of final_edges
    #                 "weight": weight,
    #                 "source": source,
    #                 "target": target
    #             })
    #             added_edges.add(edge_tuple)
    
    now = datetime.now()
    date_str = now.strftime('%y-%m-%d')
    time_str = now.strftime('%H-%M-%S')
    if not os.path.exists(f'edges/{date_str}'):
        os.makedirs(f'edges/{date_str}')

    with open(f'edges/{date_str}/{time_str}_edges_gen_edges.json', 'w') as f:
        json.dump(final_edges, f, indent=4)
    with open('edges/latest_edges.json', 'w') as f:
        json.dump(final_edges, f, indent=4)

    return

generate_edges()