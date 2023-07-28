def retrieve_update_taxonomy_extract_keywords_prompt(taxonomy, papers):
    update_taxonomy_extract_keywords_prompt = f'''
Task:
The task is to 1) update the taxonomy of all carbon capture research by re-arranging or adding new categories and levels as appropriate and 2) correctly extract the most relevant and important paper text keywords to use them to classify each paper below into its top 3 matching categories in the updated taxonomy. 

Rules and Instructions:
A. Because there are previous papers classified with the taxonomy, you must ensure that the updated taxonomy category names are backwards compatible with the input taxonomy. This means that you must ensure that all papers that were previously classified with the previous taxonomy are still classified with the updated taxonomy, even if the category names are rearranged.
B. For the taxonomy, be as mutually exclusive, completely exhaustive (MECE) and concise as possible. Try to avoid repetition and overlap. 
C. Ensure that the taxonomy is readable and not overwhelming. Try to model the taxonomy to reflect the usabilty and usefulness of great classification systems like Dewey Decimal System and Library of Congress Classification. Use layers when appropriate and to prevent there from being more than 10 categories at any level.
D. Use a hierarchical structure to manage the breadth and depth of the categories effectively, with the broadest categories at top and these categories becoming more specific as you go down the hierarchy.
E. For paper keyword extraction and classification, be as accurate and grounded in the extracted paper text keywords as possible. 

Papers (id : text): 
{papers}

Input Taxonomy (category id : category name):
{taxonomy}

The output should be in the format of: 
1. "UPDATED TAXONOMY: " -- a readable, full, and updated MECE multilevel taxonomy with any re-arrangement of categories or new categories and levels added.

2. "PAPER CLASSIFICATION: 
    [
        paper id : [[paper text keywords, corresponding category id], [paper text keywords, corresponding category id], etc.], 
        paper id : [[paper text keywords, corresponding category id], etc.], 
        etc.
    ]" 
    -- a JSON of each paper id with a relevance ranked list of paper text keywords and corresponding category id, with everything being strings. Rank by most to least relevant category to so that anyone looking for all papers about a category can find the most relevant papers to the category. Use single quotations instead of double quotations for strings.
'''
    return update_taxonomy_extract_keywords_prompt

def retrieve_taxonomy_mapping_prompt(old_taxonomy, new_taxonomy):
    retrieve_taxonomy_mapping_prompt = f'''
Task:
The task is to match each Input Taxonomy category id to its closest Updated Taxonomy category id based on category names.

Rules:
1. Be as clear and correct as possible.

Input Taxonomy (id : name):
{old_taxonomy}

Updated Taxonomy (id : name):
{new_taxonomy}

The output should be in the following JSON format: 
"UPDATED CATEGORY IDS: [
{{ "Input Taxonomy category id" : "Updated Taxonomy category id"}},
{{ "Input Taxonomy category id" : "Updated Taxonomy category id"}},
etc.]" 
-- List every category id in Input Taxonomy and its closest category id in Updated Taxonomy based on category name. Use double quotes around each id.
'''
    return retrieve_taxonomy_mapping_prompt

# Trying token-optimizing keyword classification instead of paper
def retrieve_classify_keywords_prompt(taxonomy, keywords):
    classify_keywords_prompt = f'''
Task:
The task is to 1) update the taxonomy of all carbon capture research by re-arranging or adding new categories and levels as appropriate and 2) use the list of paper keywords to classify each paper id and keywords below into its keywords matching category ids in the updated taxonomy. 

Rules and Instructions:
1. For the taxonomy, be as mutually exclusive, completely exhaustive (MECE) and concise as possible. Try to avoid repetition and overlap. 
2. Ensure that the taxonomy is readable and not overwhelming. Try to model the taxonomy to reflect the usability and usefulness of great classification systems like Dewey Decimal System and Library of Congress Classification.
3. Use a hierarchical structure to manage the breadth and depth of the categories effectively, with the broadest categories at top and these categories becoming more specific as you go down the hierarchy. A general rule of thumb is to have around 10 top-level categories and 10 sub-categories for every parent category. Feel free to use as many depth levels as appropriate.
4. For paper keyword classification, be as accurate and grounded in the paper keywords as possible. 

Papers (id : keywords): 
{keywords}

Input Taxonomy (category id : category name):
{taxonomy}

The output should be in the format of: 
1. "UPDATED TAXONOMY: " -- a readable and updated MECE multilevel taxonomy with any re-arrangement of categories or new categories and levels added.

2. "PAPER CLASSIFICATION: 
[
    paper id : [[paper keywords, corresponding category id], [paper keywords, corresponding category id], etc.], 
    paper id : [[paper keywords, corresponding category id], etc.], 
    etc.
]" 
-- a JSON of each paper id with a list of its paper keywords and corresponding Updated Taxonomy category id, with everything being strings.

List formatting example (content and assignments are arbitrary):
Input: 80 : ['carbon capture', 'biology', 'sand']
Output: 80 : [['carbon capture, 8], ['biology', 10], ['sand', 9.3]]
'''

    return classify_keywords_prompt

# Reorganize taxonomy - 1
def retrieve_organize_taxonomy(taxonomy):
    organize_taxonomy_prompt = f'''
Initial Taxonomy (id : name)
{taxonomy}

Task:
There are already papers classified under each category, but the taxonomy is potentially all over the place. Imagine that the taxonomy is going to be transformed into a map, and that the top level categories would represent a zoomed out view and the lower level categories would appear as a user zooms in.

You are trying to create a useful taxonomy for carbon capture researchers. Re-arrange the categories and their levels so that the more relevant categories are  at the top level, with non-relevant categories categorized as lower levels under Miscellaneous. Feel free to use as many depth levels as necessary. Do not change category names to make them more or less relevant.

Please take inspiration from the Dewey Decimal System and Library of Congress Classification, and ensure that no level has too many categories. Try to keep the number of categories at each level to less than 10, by rearranging, or grouping categories together under a new parent category that you create.
'''
    return organize_taxonomy_prompt