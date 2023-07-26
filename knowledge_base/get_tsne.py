import pandas as pd
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import ast
from datetime import datetime

# NOTE: Only need to edit this with latest papers JSON for pre-processing steps 1 and 2
# Load in latest JSON files
papers_json_path = r'C:\Users\1kevi\Desktop\projects\Research\autoscious-carbon-capture\knowledge_base\papers\23-07-25_11935_database_update.json'

# load the data from your JSON file
with open(papers_json_path, 'r') as f:
    data = json.load(f)

# convert the data into a pandas DataFrame
df = pd.DataFrame(data)

print("Before # x not None: ", len(df[df['x'].notna()]), "# None: ", len(df[df['x'].isna()]))

# Create a temporary DataFrame with only rows where 'embedding' is not NaN
df_temp = df[df['embedding'].notna()]

# Convert string of list to numpy array
# TODO: this is probably a very slow step to convert strings to lists
print("converting all embeddings to lists")
df_temp['embedding'] = df_temp['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))

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

# Get today's date
now = datetime.now()
date_str = now.strftime('%y-%m-%d')
time_str = now.strftime('%H-%M-%S')
folder_path = f'papers/{date_str}'
n = len(papers.keys())
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

df.to_json(f'{folder_path}/{time_str}_tsne_output_{df.shape[0]}.json', orient='records')

print("# x not None: ", len(df[df['x'].notna()]), "# None: ", len(df[df['x'].isna()]))