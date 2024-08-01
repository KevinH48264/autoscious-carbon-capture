import random
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from math import exp
import re
import numpy as np
import torch.nn.functional as F
import graphviz
import json
from collections import deque
from peft import get_peft_model, LoraConfig, TaskType

torch.manual_seed(42) # Set seed for reproducibility

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


# hyperparameters
MAX_TOKENS = 256  # Maximum number of tokens to generate
SAVE_VERSION = "v2.011"  # for saving the filename

# Load model
print("Is CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "deepseek-ai/deepseek-math-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


batch_size = 2 # DEBUG: 4
# Data setup
train_dataset = PolicyValueDataset('trees_v2_011/game_0/examples.json', tokenizer) # 8K examples
val_dataset = PolicyValueDataset('trees_v2_011/game_1/examples.json', tokenizer) # 2K examples

# train_dataloader = model_engine.deepspeed_io(train_dataset, collate_fn=collate_fn) # for data sharding during distributed training
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)


# --- MODEL ---
# Define the ValueHead class
class ValueHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.summary_dropout_prob) if hasattr(config, 'summary_dropout_prob') else nn.Identity()
        hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else config.word_embed_proj_dim
        self.value_head = nn.Linear(hidden_size, 1) # Linear layer to predict the expected reward (Q) of the current state

    def forward(self, last_hidden_state):
        output = self.dropout(last_hidden_state)
        output = self.value_head(output)
        values = torch.tanh(output)
        return values

# Define the policy and value model class
class AutoModelForCausalLMWithValueHead(nn.Module):
    def __init__(self, model_name, tokenizer):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer

        # Base model for policy generation
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.base_model.generation_config = GenerationConfig.from_pretrained(model_name)
        self.base_model.generation_config.pad_token_id = self.base_model.generation_config.eos_token_id
        self.base_model = get_peft_model(self.base_model, LoraConfig(task_type="CAUSAL_LM", r=20, lora_alpha=40, lora_dropout=0.05, bias="none")) # Integrate LoRA for training. Make sure to call merge_lora_weights(model.base_model) before inferencing. Can look at configs here too: https://www.kaggle.com/code/jatinsinghsagoi/aimo-24-finetune-deepseek-math
        self.base_model.print_trainable_parameters() # Print trainable parameters

        # Value head for policy evaluation
        self.value_head = ValueHead(self.base_model.config)

    def forward(self, input_ids, attention_mask=None, labels=None, state_values=None, **kwargs):
        '''Takes in input_ids and returns logits and value prediction. It also returns loss if labels AND state_values are provided.'''
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None: attention_mask = attention_mask.to(self.device)
        if labels is not None: labels = labels.to(self.device)
        if state_values is not None: state_values = state_values.to(self.device)

        # forward policy and value heads
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        value_preds = self.value_head(outputs.hidden_states[-1])  # [batch_size, seq_len, 1]

        # compute loss
        total_loss, ce_loss, mse_loss = None, None, None
        if labels is not None and state_values is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100) # Cross-entropy loss for next token prediction
            state_values_pred = value_preds[:, -1, :].squeeze(-1)  # [batch_size] # Use the state values at the last position for each sequence
            mse_loss = F.mse_loss(state_values_pred, state_values) # Mean squared error loss for state value prediction
            total_loss = ce_loss + mse_loss

        return logits, value_preds, total_loss, ce_loss, mse_loss
    
    # def generate(self, input_ids, attention_mask=None, max_tokens=MAX_TOKENS): # potentially use for policy step
    #     '''Generate a sequence of tokens given input_ids.'''
    #     with torch.no_grad():
    #         # Generate sequence
    #         generated_sequence = self.base_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_tokens, pad_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=True, output_hidden_states=True)
    #         return generated_sequence
        
    def evaluate(self, state: str = ""):
        '''Essentially the forward reward function. Use for evaluation step (separate from policy step)'''
        with torch.no_grad():
            inputs = self.tokenizer(state, return_tensors="pt").to(self.device) # input_ids.shape = [1, seq_len]
            outputs = self.base_model.model(**inputs)
            last_hidden_state = outputs.last_hidden_state # shape = [1, seq_len, hidden_size]
            value = self.value_head(last_hidden_state) # shape = [1, seq_len, 1]
            return value[:, -1, :].item() # Return the value of the last token


model = AutoModelForCausalLMWithValueHead(model_name, tokenizer).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4) # weight_decay penalizes large weights. Adds a regularization term to the loss function that shrinks the weights during each update

with open('debug/model_named_parameters.txt', 'w') as f:
    for name, param in model.named_parameters():
        f.write(f"{name}\n") # does indeed see lm_head and value_head

# Ensure LoRA parameters are trainable
for name, param in model.named_parameters():
    if 'lora' in name or 'value_head' in name:
        param.requires_grad = True
with open('debug/trainable_parameters.txt', 'w') as f:
    for name, param in model.named_parameters():
        if param.requires_grad:
            f.write(f"{name}\n")

SAVE_VERSION = "v2.011"  # for saving the filename

# Training loop
import wandb
from tqdm import tqdm
import os

# initialization
wandb.init(project=f"mcts_v2.1_policy_value_mcts_training_v{SAVE_VERSION}")
model.train()
start_epoch = 0
epochs = 10 # 50
checkpoint_interval = max(1, epochs // 5) # Calculate checkpoint interval (20% of the total number of epochs)
checkpoint_dir = f'v2_1_policy_value_mcts_training_{SAVE_VERSION}'
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(start_epoch, epochs):
    total_loss = 0.0
    total_mse = 0.0
    total_ce = 0.0
    total_mae = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch in progress_bar:
        input_ids, attention_mask, labels, state_values_true = batch

        # Forward pass
        logits, value_preds, total_loss, ce_loss, mse_loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, state_values=state_values_true)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Accumulate losses for logging
        total_loss += total_loss.item()
        total_mse += mse_loss.item()
        total_ce += ce_loss.item()
        total_mae += torch.abs(value_preds[:, -1, :].squeeze(-1) - state_values_true.to(model.device)).mean().item()

        progress_bar.set_postfix({"loss": total_loss.item()})

    avg_loss = total_loss / len(train_dataloader)
    avg_mse = total_mse / len(train_dataloader)
    avg_ce = total_ce / len(train_dataloader)
    avg_mae = total_mae / len(train_dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, MSE: {avg_mse}, CE: {avg_ce}, MAE: {avg_mae}")
    wandb.log({"epoch": epoch+1, "loss": avg_loss, "mse": avg_mse, "ce": avg_ce, "mae": avg_mae})

    # Validation step
    model.eval()
    val_total_loss = 0.0  # Initialize val_total_loss
    val_total_mse = 0.0
    val_total_ce = 0.0
    val_total_mae = 0.0
    with torch.no_grad():
        for val_batch in val_dataloader:
            input_ids, attention_mask, labels, state_values_true = val_batch

            # Forward pass
            logits, value_preds, val_loss, val_ce_loss, val_mse_loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, state_values=state_values_true)

            # Accumulate validation losses
            val_total_loss += val_loss.item()  # Accumulate val_total_loss
            val_total_mse += val_mse_loss.item()
            val_total_ce += val_ce_loss.item()
            val_total_mae += torch.abs(value_preds[:, -1, :].squeeze(-1) - state_values_true.to(model.device)).mean().item()

    avg_val_loss = val_total_loss / len(val_dataloader)  # Calculate average validation total loss
    avg_val_mse = val_total_mse / len(val_dataloader)
    avg_val_ce = val_total_ce / len(val_dataloader)
    avg_val_mae = val_total_mae / len(val_dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}, Validation MSE: {avg_val_mse}, Validation CE: {avg_val_ce}, Validation MAE: {avg_val_mae}")
    wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss, "val_mse": avg_val_mse, "val_ce": avg_val_ce, "val_mae": avg_val_mae})  # Log avg_val_loss

    model.train()

    # Save checkpoint every 20% of the epochs
    if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs:
        print("Saving model checkpoint...")
        checkpoint = {
            'base_model_state_dict': model.base_model.state_dict(),
            'value_head_state_dict': model.value_head.state_dict(),
            'epoch': epoch + 1
        }
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(checkpoint, f'{checkpoint_dir}/model_checkpoint_{epoch + 1}.pth')
        torch.cuda.empty_cache()

print("Fine-tuning completed.")
wandb.finish()