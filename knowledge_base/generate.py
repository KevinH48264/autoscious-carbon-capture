import json

import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE

# load the data from your JSON file
with open(r'C:\Users\91787\PycharmProjects\autoscious-carbon-capture\data_collection\output_100.json', 'r') as f:
    data = json.load(f)

# convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# remove rows with null abstracts
# df = df[df['abstract'].notna()]

# print out the DataFrame to verify
print(df.head())

embeddings = np.vstack(df['embedding'].apply(lambda x: x['vector']))

# Going for a more interactive visualization
import plotly.graph_objects as go

# After obtaining the t-SNE 2D coordinates as mentioned in the previous steps...
umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
embeddings_2d = proj_2d = umap_2d.fit_transform(embeddings)

# Create a Plotly figure
fig = go.Figure(data=go.Scatter(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    mode='markers+text',
    text=df['title'],  # This will display the title below each point
    textposition='bottom center',
    hovertemplate=  # This will display title, abstract, and tldr (if you have this column) on hover
        '<b>Title:</b> %{text}' +
        '<br><b>Abstract:</b> %{customdata[0]}' +
        '<br><b>TL;DR:</b> %{customdata[1]}',
    customdata=df[['abstract', 'tldr']]  # Replace 'tldr' with the appropriate column name if it exists
))

# Set the title of the figure
fig.update_layout(title='u-map visualization of embedding vectors')

# Show the figure
fig.show()
