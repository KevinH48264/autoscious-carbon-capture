'''
Main policy and value model class for MCTS
'''

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, GenerationConfig
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F

torch.manual_seed(42)

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
        if state_values is not None: state_values = state_values.to(self.device).half() # for mixed precision training with deepspeed

        # forward policy and value heads
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        value_preds = self.value_head(outputs.hidden_states[-1]).half()  # [batch_size, seq_len, 1]

        # compute loss
        total_loss, ce_loss, mse_loss = None, None, None
        if labels is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100) # Cross-entropy loss for next token prediction
        if state_values is not None:
            state_values_pred = value_preds[:, -1, :].squeeze(-1)  # [batch_size] # Use the state values at the last position for each sequence
            mse_loss = F.mse_loss(state_values_pred, state_values) # Mean squared error loss for state value prediction

        # Combine losses
        if ce_loss is not None and mse_loss is not None:
            total_loss = ce_loss + mse_loss
        elif ce_loss is None and mse_loss is not None:
            total_loss = mse_loss
        elif ce_loss is not None and mse_loss is None:
            total_loss = ce_loss

        return logits, value_preds, total_loss, ce_loss, mse_loss
    
    def evaluate(self, state: str = ""):
        '''Essentially the forward reward function. Use for evaluation step (separate from policy step)'''
        with torch.no_grad():
            inputs = self.tokenizer(state, return_tensors="pt").to(self.device) # input_ids.shape = [1, seq_len]
            outputs = self.base_model.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1] # shape = [1, seq_len, hidden_size]
            value = self.value_head(last_hidden_state) # shape = [1, seq_len, 1]
            return value[:, -1, :].item() # Return the value of the last token