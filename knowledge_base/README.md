## File organization
1. checkpoints -- for saving backups and checkpoints for each date for clusters, edges, and papers
    1. folder format = "year-month-day"
    2. file_format = "type_model_size_notes" under the respective folder
2. clusters
    1. complete clusters files in "year-month-day_size_notes"
3. edges
    1. complete edges file in "year-month-day_size_notes"
4. papers
    1. complete papers file in "year-month-day_size_notes"
5. old -- archive

## 1. For updating knowledge base with new papers: 
- Edit and run update_database.ipynb or run update_database.py

## 2. For adding embeddings
- Edit and run get_embeddings.py
- Alternatively, edit and run generate_viz.ipynb Pre-processing step 1

## 3. For generating latest t-SNE x and y coordinates:
- Edit and run get_tsne.py
- Alternatively, edit and run generate_viz.ipynb Pre-processing step 2

## 4. For generating a seeded initial taxonomy
- Edit the prompt and run "python seed_initial_taxonomy.py"
- Optionally, edit and run generate_viz.ipynb

## 5. For updating taxonomy and extracting keywords and classifying papers without extracted keywords
- Edit and run update_taxonomy_new_classify.py
- Alternatively, edit and run generate_viz.ipynb

## 6. For updating taxonomy and reclassifying extracted keywords
- Edit and run update_taxonomy_reclassify_keywords.py (under development, currently just reclassifies from the start instead of from lowest confidence score)
- Alternatively, edit and run generate_viz.ipynb

## 7. For reorganizing taxonomy
- Edit and run reorganize_taxonomy.py
- Alternatively, edit and run generate_viz.ipynb

## 8. For adding confidence scores to classification ids (O(# classified_ids))
- Edit and run add_keyword_class_scores.py
- Alternatively, edit and run generate_viz.ipynb

### Improvements:
1. A column in df to signal if a row has already had their confidence score calculated could decrease runtime drastically by filtering. 
2. This could be an error catching step where if classification id was not found, then the format was likely wrong and the keywords need to be reclassified. It could be set to None, and then this script and work with update_taxonomy_reclassify_keywords to just filter and rank based on where 1) classification_id = None and then 2) confidence_score is low.
3. Currently slow because of ast.literal_eval. Could be improved by using a different method to convert string to list. Perhaps regex.

## 9. For generating taxonomy JSON (O(# papers))
- Edit and run gen_taxonomy_json.py
- Alternatively, edit and run generate_viz.ipynb

## 10. For generating edges (O(n choose 2))
- Edit and run gen_edges.py
- Alternatively, edit and run generate_viz.ipynb

Improvements:
1. Currently unscalable if there are too many papers in one subcluster.
1.1 Improvement 1: Restrict amount of papers in one subcluster by updating the taxonomg and reclassifying as appropriate.