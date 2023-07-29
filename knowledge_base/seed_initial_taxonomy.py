'''This file creates an initial seed taxonomy, but only adds it as a checkpoint in clusters/date/time instead of overwriting the latest_taxonomy.json because usually you don't want to replace latest_taxonomy.json'''

from datetime import datetime
import os
from llm import chat_openai

def create_initial_taxonomy_prompt(model="gpt-4"):
    seed_initial_taxonomy_prompt = '''
    Create a taxonomy of all carbon capture research areas. Be as mutually exclusive, completely exhaustive (MECE) and concise as possible. Be sure to also include a "General" category for information like literature reviews and updates, a "Miscellaneous" category for concepts that have yet to be covered by an appropriate category and non-carbon capture related concepts, and use multilevel numbering. Create as many levels breadth-wise and depth-wise as appropriate.
    '''

    res = chat_openai(seed_initial_taxonomy_prompt, model=model)
    initial_taxonomy = res[0]

    print("INITIAL TAXONOMY: ", initial_taxonomy)

    # save the taxonomy to a txt file
    now = datetime.now()
    date_str = now.strftime('%y-%m-%d')
    time_str = now.strftime('%H-%M-%S')
    folder_path = "clusters/" + date_str
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(f'clusters/{date_str}/{time_str}_initial_taxonomy.txt', 'w') as f:
        f.write(initial_taxonomy)

create_initial_taxonomy_prompt("gpt-3.5-turbo")