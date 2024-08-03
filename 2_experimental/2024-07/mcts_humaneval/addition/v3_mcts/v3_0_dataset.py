'''
Main Dataset class and helper functions for MCTS training examples from self-play games
'''

import json
import torch
from torch.utils.data import Dataset
import os
import random

# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# --- DATA ---
# Load and preprocess data
class PolicyValueDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''Support next token prediction and value prediction. 
        
        Returns: input_ids = parent_state + child_token_seq, labels = next token prediction labels of length child_token_seq, q = value prediction'''
        parent_state = self.data[idx]['state']
        child_token_seq = self.data[idx]['generated_sequence']
        q = self.data[idx]['value']
        input_ids = self.tokenizer(parent_state + child_token_seq, return_tensors='pt', padding=False, add_special_tokens=True) # Tokenize parent_state + child_token_seq for next token prediction # include BOS for deepseekmath
        
        # Tokenize parent_state and child_token_seq separately to get their lengths
        child_length = self.tokenizer(child_token_seq, return_tensors='pt', padding=False, add_special_tokens=True)['input_ids'].size(1) - 1 # exclude BOS
        labels = -100 * torch.ones_like(input_ids['input_ids'])
        labels[0, -child_length-1:-1] = input_ids['input_ids'][0, -child_length:] # Copy the child_token_seq to labels, shifted by 1 to the left

        return input_ids, labels, q

def collate_fn(batch):
    batch_size = len(batch)
    max_len = max(item[0]['input_ids'].size(1) for item in batch)  # Find the maximum length of sequences in the batch

    # Initialize tensors with appropriate sizes
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = -100 * torch.ones((batch_size, max_len), dtype=torch.long)  # Initialize labels with -100
    state_values = torch.zeros(batch_size, dtype=torch.float)

    # Fill the tensors with the actual values from the batch
    for i, (input_id_dict, label, q) in enumerate(batch):
        input_id = input_id_dict['input_ids']
        att_mask = input_id_dict['attention_mask']
        seq_len = input_id.size(1)
        input_ids[i, :seq_len] = input_id
        attention_mask[i, :seq_len] = att_mask
        labels[i, :seq_len] = label
        state_values[i] = q

    return input_ids, attention_mask, labels, state_values

def get_latest_game_index(save_version):
    if not os.path.exists(f"{save_version}/latest"):
        with open(f"{save_version}/latest", "w") as file:
            file.write("0") # initial game index = 0

    with open(f"{save_version}/latest", "r") as file:
        latest_game_index = file.read().strip()
    return latest_game_index
    
def make_datasets(save_version):
    latest_game_index = get_latest_game_index(save_version)

    # define directories
    latest_games_dir = f"{save_version}/{latest_game_index}_self_play"
    curr_training_dir = f"{save_version}/{latest_game_index}_training"
    os.makedirs(curr_training_dir, exist_ok=True)

    # collect all examples from the latest self-play game
    all_examples = []
    for subdir in os.listdir(latest_games_dir):
        subdir_path = os.path.join(latest_games_dir, subdir)
        if os.path.isdir(os.path.join(latest_games_dir, subdir)):
            examples_path = os.path.join(subdir_path, 'examples.json')
            if os.path.isfile(examples_path):
                with open(examples_path, 'r') as f:
                    examples = json.load(f)
                    all_examples.extend(examples)

    # Save data examples into train/val sets
    random.shuffle(all_examples)
    split_index = int(0.8 * len(all_examples)) # 80/20 for train/val split

    with open(f"{curr_training_dir}/train_examples.json", "w") as file:
        json.dump(all_examples[:split_index], file, indent=4)
    with open(f"{curr_training_dir}/val_examples.json", "w") as file:
        json.dump(all_examples[split_index:], file, indent=4)

    return curr_training_dir, latest_game_index