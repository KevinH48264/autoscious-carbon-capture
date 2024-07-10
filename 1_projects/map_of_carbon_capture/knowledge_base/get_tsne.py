import pandas as pd
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import ast
from datetime import datetime
import numpy as np
import os
from update_taxonomy_util import load_latest_taxonomy_papers, save_taxonomy_papers_note


search_term_filter = "" # "" to just process all rows, which you generally want to do for t-SNE visualization

# convert the data into a pandas DataFrame
numbered_taxonomy, df = load_latest_taxonomy_papers(search_term_filter)

print("Before # x not None: ", len(df[df['x'].notna()]), "# None: ", len(df[df['x'].isna()]))

# Create a temporary DataFrame with only rows where 'embedding' is not NaN
df_temp = df[df['embedding'].notna()].copy()

# Convert string of list to numpy array
# TODO: this is probably a very slow step to convert strings to lists
print("converting all embeddings to lists")
df_temp['embedding'] = df_temp['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
print("finished converting embedding strings to lists")

# Stack all embeddings into a numpy array
embeddings = np.vstack(df_temp['embedding'])

# Compute t-SNE
print("computing tsne")
tsne = TSNE(n_components=2, random_state=0)
embeddings_2d = tsne.fit_transform(embeddings)

# Update 'x' and 'y' in the original DataFrame for rows that have embeddings
print("updating df")
df.loc[df_temp.index, 'x'] = embeddings_2d[:, 0]
df.loc[df_temp.index, 'y'] = embeddings_2d[:, 1]

# Create a scatter plot of all the points with node sizes based on normalized citationCount
plt.figure(figsize=(10, 10))
plt.scatter(df['x'], df['y'], alpha=0.5, label='All papers')
plt.show()

save_taxonomy_papers_note(numbered_taxonomy, df, "tsne_output", search_term_filter)

print("# x not None: ", len(df[df['x'].notna()]), "# None: ", len(df[df['x'].isna()]))