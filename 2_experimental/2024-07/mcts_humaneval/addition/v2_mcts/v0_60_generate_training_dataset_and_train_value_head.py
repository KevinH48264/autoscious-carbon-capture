import os
import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
import deepspeed

SAVE_VERSION = "v0.61"

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

# Define the extended model class
class AutoModelForCausalLMWithValueHead(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = self.base_model.config
        self.value_head = ValueHead(self.config)
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        transformer_outputs = self.base_model.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = transformer_outputs.last_hidden_state
        logits = self.base_model.lm_head(last_hidden_state)
        reward = self.value_head(last_hidden_state)
        return logits, reward
    
    def evaluate(self, state: str = ""):
        with torch.no_grad():
            inputs = self.tokenizer(state, return_tensors="pt")
            outputs = self.base_model.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            value = self.value_head(last_hidden_state)
            return value

# Load and preprocess data
class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_context_window):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_context_window = max_context_window

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Input: state
        Output: next token, Q value of state
        '''

        item = self.data[idx]
        state = item['state']
        next_token = item['token']
        state_value = item['Q']
        
        inputs = self.tokenizer(state, return_tensors='pt', padding=False, truncation=True, max_length=self.max_context_window) # truncates to max_length if the context window is too large
        next_token_id = self.tokenizer.encode(next_token, add_special_tokens=False, max_length=1, truncation=True)

        return inputs, torch.tensor(next_token_id), torch.tensor(state_value, dtype=torch.float)

def collate_fn(batch):
    batch_size = len(batch) # batch_size
    max_len = max(item[0]['input_ids'].size(1) for item in batch) # standardize all sequences to this length, for memory efficiency. We can also just use max_tokens, but this is more efficient

    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    next_token_ids = torch.zeros(batch_size, dtype=torch.long)
    state_values = torch.zeros(batch_size, dtype=torch.float)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)  # Track the actual lengths

    # Fill in the zero tensors with the actual values
    for i, (inputs, next_token_id, state_value) in enumerate(batch):
        seq_len = inputs['input_ids'].size(1)
        
        input_ids[i, :seq_len] = inputs['input_ids'].squeeze()
        attention_mask[i, :seq_len] = inputs['attention_mask'].squeeze()
        next_token_ids[i] = next_token_id
        state_values[i] = state_value
        seq_lens[i] = seq_len # track sequences to index appropriately into state_values_pred

    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'seq_lens': seq_lens}
    return inputs, next_token_ids, state_values


# Load model and tokenizer
print("Is CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "deepseek-ai/deepseek-math-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
policy_value_model = AutoModelForCausalLMWithValueHead(model_name)

train_dataset = CustomDataset('v0_6_value_head_training/train_80000.json', tokenizer, 4096) # 8K examples
val_dataset = CustomDataset('v0_6_value_head_training/val_20000.json', tokenizer, 4096) # 2K examples

# DeepSpeed configuration to run on multiple GPUs in one node / instance
batch_size = 4
ds_config = {
    "train_batch_size": batch_size * 8, # * # of GPUs
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "reduce_scatter": True,
        "allgather_bucket_size": 2e8,
        "reduce_bucket_size": 2e8
    },
    "gradient_accumulation_steps": 1,
    "steps_per_print": 2000,
    "wall_clock_breakdown": False
}

# Training setup
optimizer = torch.optim.AdamW(policy_value_model.value_head.parameters(), lr=5e-4, weight_decay=1e-4) # weight_decay penalizes large weights. Adds a regularization term to the loss function that shrinks the weights during each update
model_engine, optimizer, __, __ = deepspeed.initialize(
    model=policy_value_model,
    optimizer=optimizer,
    model_parameters=policy_value_model.value_head.parameters(), # Only optimize the value head
    config=ds_config
)
train_dataloader = model_engine.deepspeed_io(train_dataset, collate_fn=collate_fn) # for data sharding during distributed training
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

# Check for existing checkpoints
def get_latest_checkpoint(directory):
    checkpoints = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    checkpoints = sorted(checkpoints, key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else -1)
    return checkpoints[-1] if checkpoints else None

checkpoint_dir = f'v0_6_value_head_training_{SAVE_VERSION}'
latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

start_epoch = 0
if latest_checkpoint:
    print(f"Resuming from checkpoint: {latest_checkpoint}")
    model_engine.load_checkpoint(checkpoint_dir, latest_checkpoint)
    start_epoch = int(re.findall(r'\d+', latest_checkpoint)[-1]) + 1

# Complete an initial save with checkpointed model
if torch.distributed.get_rank() == 0:  # Ensure only one process saves the model
    print("Saving initial model checkpoint...")
    state_dict = model_engine.module.state_dict()  # Access the underlying model state_dict
    torch.save(state_dict, f'v0_6_value_head_training_{SAVE_VERSION}/model_checkpoint.pth')

# Training loop
epochs = 150
wandb.init(project="mcts_v0.6_value-head-finetuning")

for epoch in range(start_epoch, epochs):
    model_engine.train()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in progress_bar:
        inputs, next_token_ids, state_values_true = batch
        input_ids = inputs['input_ids'].to(model_engine.device) # [batch_size, batch_max_seq_len]
        attention_mask = inputs['attention_mask'].to(model_engine.device) # [batch_size, batch_max_seq_len]
        seq_lens = inputs['seq_lens'].to(model_engine.device) # [batch_size]
        state_values_true = state_values_true.to(model_engine.device).half() # [batch_size]

        logits, state_values_pred = model_engine(input_ids=input_ids, attention_mask=attention_mask) # logits.shape = [batch_size, batch_max_seq_len, vocab_size], state_values_pred.shape = [batch_size, batch_max_seq_len, 1]
        last_valid_indices = (seq_lens - 1).view(-1, 1, 1) # [batch_size, 1, 1]
        state_values_pred = state_values_pred.gather(1, last_valid_indices).squeeze(1).squeeze(-1).half()  # [batch_size], gather state_values_pred at the given sequence indices
        loss = nn.MSELoss()(state_values_pred, state_values_true)

        model_engine.backward(loss)
        model_engine.step()

        total_loss += loss.item()
        total_mse += loss.item()
        total_mae += torch.abs(state_values_pred - state_values_true).mean().item()

        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_dataloader)
    avg_mse = total_mse / len(train_dataloader)
    avg_mae = total_mae / len(train_dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, MSE: {avg_mse}, MAE: {avg_mae}")
    wandb.log({"epoch": epoch+1, "loss": avg_loss, "mse": avg_mse, "mae": avg_mae})

    # Validation step
    model_engine.eval()
    val_total_mse = 0.0
    val_total_mae = 0.0
    with torch.no_grad():
        for val_batch in val_dataloader:
            inputs, next_token_ids, state_values_true = val_batch
            input_ids = inputs['input_ids'].to(model_engine.device)
            attention_mask = inputs['attention_mask'].to(model_engine.device)
            seq_lens = inputs['seq_lens'].to(model_engine.device)
            state_values_true = state_values_true.to(model_engine.device).half()  # For DeepSpeed FP16

            logits, state_values_pred = model_engine(input_ids=input_ids, attention_mask=attention_mask) # logits.shape = [batch_size, batch_max_seq_len, vocab_size], state_values_pred.shape = [batch_size, batch_max_seq_len, 1]
            last_valid_indices = (seq_lens - 1).view(-1, 1, 1) # [batch_size, 1, 1]
            state_values_pred = state_values_pred.gather(1, last_valid_indices).squeeze(1).squeeze(-1).half()  # [batch_size], gather state_values_pred at the given sequence indices
            
            val_total_mse += nn.MSELoss()(state_values_pred, state_values_true).item()
            val_total_mae += torch.abs(state_values_pred - state_values_true).mean().item()

    avg_val_mse = val_total_mse / len(val_dataloader)
    avg_val_mae = val_total_mae / len(val_dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Validation MSE: {avg_val_mse}, Validation MAE: {avg_val_mae}")
    wandb.log({"epoch": epoch+1, "val_mse": avg_val_mse, "val_mae": avg_val_mae})

    model_engine.train()

    # After training is complete
    if torch.distributed.get_rank() == 0:  # Ensure only one process saves the model
        print("Saving model checkpoint...")
        state_dict = model_engine.module.state_dict()  # Access the underlying model state_dict
        torch.save(state_dict, f'v0_6_value_head_training_{SAVE_VERSION}/model_checkpoint.pth')
    torch.cuda.empty_cache() # Free up memory before checkpoint
    model_engine.save_checkpoint(f'v0_6_value_head_training_{SAVE_VERSION}', epoch)

print("Fine-tuning completed.")
wandb.finish()
