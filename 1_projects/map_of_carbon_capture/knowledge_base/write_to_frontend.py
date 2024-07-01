import os
import shutil
import json

taxonomy_source_path = "clusters/latest_taxonomy.json"
taxonomy_destination_path = "../react-app/public/latest/latest_taxonomy.json"
papers_source_path = "papers/latest_papers.json"
papers_destination_path = "../react-app/public/latest/latest_papers.json"
edges_source_path = "edges/latest_edges.json"
edges_destination_path = "../react-app/public/latest/latest_edges.json"

print("Transferring taxonomy JSON file to frontend")
if os.path.exists(taxonomy_source_path):
    try:
        # Open the file to check if it's a valid JSON
        with open(taxonomy_source_path, 'r') as file:
            json.load(file)
        # If it's valid, then copy it to the destination path
        shutil.copy(taxonomy_source_path, taxonomy_destination_path)
        print("File has been copied successfully.")
    except ValueError as e:
        print("Invalid JSON file:", e)
else:
    print("Source file does not exist.")

print("Transferring papers JSON file to frontend")
if os.path.exists(papers_source_path):
    try:
        # Open the file to check if it's a valid JSON
        with open(papers_source_path, 'r') as file:
            json.load(file)
        # If it's valid, then copy it to the destination path
        shutil.copy(papers_source_path, papers_destination_path)
        print("File has been copied successfully.")
    except ValueError as e:
        print("Invalid JSON file:", e)
else:
    print("Source file does not exist.")


print("Transferring edges JSON file to frontend")
if os.path.exists(edges_source_path):
    try:
        # Open the file to check if it's a valid JSON
        with open(edges_source_path, 'r') as file:
            json.load(file)
        # If it's valid, then copy it to the destination path
        shutil.copy(edges_source_path, edges_destination_path)
        print("File has been copied successfully.")
    except ValueError as e:
        print("Invalid JSON file:", e)
else:
    print("Source file does not exist.")

