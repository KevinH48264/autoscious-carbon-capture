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
- Edit and run update_database.ipynb.

## 2. For adding embeddings
- Edit and run generate_viz.ipynb Pre-processing step 1 or edit and run get_embeddings.py

## 3. For generating latest t-SNE x and y coordinates:
- Edit and run generate_viz.ipynb Pre-processing step 2 or edit and run get_tsne.py

## 4. For generating a seeded initial taxonomy
- Edit the prompt and run "python seed_initial_taxonomy.py"

## 5. For updating taxonomy and extracting keywords and classifying papers without extracted keywords
- Edit and run generate_viz.ipynb Pre-processing step 3 or edit and run update_taxonomy_classify_new.py

## 6. For updating taxonomy and reclassifying extracted keywords
- Edit and run generate_viz.ipynb Pre-processing step 3 and use process_keywords instead of process_papers or edit and run update_taxonomy_reclassify_keywords.py (under development, currently just reclassifies from the start instead of from lowest confidence score)

## 7. For reorganizing taxonomy

## 8. For adding confidence scores to classification ids

## 8. For generating edges

## 9. For adapting taxonomy to be more structured