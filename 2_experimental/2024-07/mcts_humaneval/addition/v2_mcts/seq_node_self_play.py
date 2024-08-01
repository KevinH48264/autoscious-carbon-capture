'''This implements MCTS 1 self-play trajectory like AlphaZero

TODOs:
1. I think I have a problem where some nodes are becoming terminal and it's not because they contain the answer submission, but just don't generate anything? IDK.
'''


import random
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import pickle
import os
from math import exp
import re
import numpy as np
import torch.nn.functional as F
import graphviz
import json
from collections import deque
from peft import get_peft_model, LoraConfig

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
        '''Takes in input_ids token ids and returns logits and value prediction. It also returns loss if labels AND state_values are provided.'''
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
        '''Essentially the forward reward function. Use for evaluation step (separate from policy step). In the future, may be able to combine with forward pass.'''
        with torch.no_grad():
            inputs = self.tokenizer(state, return_tensors="pt").to(self.device) # input_ids.shape = [1, seq_len]
            outputs = self.base_model(**inputs, output_hidden_states=True)
            value = self.value_head(outputs.hidden_states[-1]) # value.shape = [1, seq_len, 1] # outputs.last_hidden_state = # shape = [1, seq_len, hidden_size]
            return value[:, -1, :].item() # Return the value of the last token


# --- TREE SEARCH ---
# Prepare Search Tree
class Node:
    '''Node class for MCTS. Each node represents a possible token (character) that could come next in the sequence of generated text.'''

    def __init__(self, id: str = "", prob: float = 1.0, state: str = "", parent: 'Node' = None, token: str = "", depth: int = 0):
        self.id = id # unique identifier for the node
        self.Q = 0 # self.value, average so it's not a high risk high reward and we can use the traces for training. expected value encourages more robust reasoning)? TODO: 1) play around with max vs average to think about test time calculations without knowing ground truth. max reward is kind of weird. 2) play around with a reward model rating the prompt itself on how "solvable" it is for if it's a high quality problem to tackle
        self.P = exp(prob) # input for the P_UCB calculation. Exponentiating allows for higher probabilities to have higher impact
        self.state = state # full generated text sequence up till node
        self._parent = parent
        self._children = []
        self.visits = 0 # IMPORTANT for UCB: self.visits = # of times you've updated the sample Q-value for that node. Only update if you've updated the Q value, otherwise a visit didn't help you update your confidence in the sample Q value being close to true Q value. Aka in backprop.
        self.token = token # current token
        self.depth = depth # depth of node in tree
        self.p_ucb = 0 # priority upper confidence bound value for this node
        self.solution = ""

        self.updated = False # field to indicate if the node was updated for visualization purposes
        self.player_node = False # field to indicate if the node is a player node
        self.selected = False # field to indicate if the node was selected for visualization purposes
        self.is_terminal = False # field to indicate if the node is a terminal node

    def __str__(self) -> str:
        child_ids = ", ".join([child.id for child in self._children])
        parent_id = self._parent.id if self._parent else None
        return f'''Node(
    self.id = "{self.id}",
    self.Q = {self.Q},
    self.P = {self.P},
    self.state = "{self.state}",
    self._parent = {parent_id},
    self._children = [{child_ids}],
    self.visits = {self.visits},
    self.token = "{self.token}",
    self.depth = {self.depth},
    self.p_ucb = {self.p_ucb},
    self.visits = {self.visits}
)'''

    def backprop(self) -> None:
        '''Assuming the children nodes are just expanded and evaluated child node with its Q value updated, this backpropagates the updated Q value to recalculate the Q value of the current and parent nodes weighted on visits.'''
        node = self
        while node:
            # Update Q value as a weighted average of the Q values of the children based on the number of visits. visits + 1 because we have visited it, we just don't consider it a selected_visit
            total_visits = sum((child.visits + 1) for child in node._children)
            if total_visits > 0:
                node.Q = sum(child.Q * (child.visits + 1) for child in node._children) / total_visits         
            
            node.updated = True  # Mark the node as updated for visualization purposes
            node = node._parent


    def visualize_tree(self, filename: str, dpi: int = 300):
        """
        Visualize the tree and save it as an PNG file.
        """
        # dot = graphviz.Digraph(format='png')
        # dot.attr(dpi=str(dpi))  # Set the resolution
        # self._add_to_graph(dot)
        # dot.render(filename, view=False)

        dot = graphviz.Digraph(format='png')
        dot.attr(dpi=str(dpi))
        dot.attr(rankdir='TB')  # Change layout direction to top-bottom
        dot.attr(nodesep='0.5')  # Increase node separation
        dot.attr(ranksep='1')  # Increase rank separation
        dot.node_attr.update(shape='record', width='2', height='2')  # Adjust node width and height
        self._add_to_graph(dot)

        # Add a legend
        with dot.subgraph(name='cluster_legend') as legend:
            legend.attr(label='Legend', fontsize='16', fontname='Helvetica')
            legend.node_attr.update(shape='record', width='1.5', height='0.75', fontsize='14')  # Adjust legend node size
            legend.node('updated', label='Updated Q / P-UCB', style='filled,bold', fillcolor='lightgrey', shape='record')
            legend.node('terminal', label='Terminal / True Reward', style='filled,bold', fillcolor='lightcoral', shape='record')
            legend.node('selected', label='Selected from MCTS Selection', style='filled,bold', fillcolor='lightblue', shape='record')
            legend.node('player', label='Player Node / Current State', style='filled,bold', fillcolor='lightgreen', shape='record')

        dot.render(filename, view=False)
        
    def _insert_line_breaks(self, text, max_length):
        """
        Insert line breaks into the text at the given max_length.
        """
        lines = []
        while len(text) > max_length:
            space_index = text.rfind(' ', 0, max_length)
            if space_index == -1:
                space_index = max_length
            lines.append(text[:space_index])
            text = text[space_index:].strip()
        lines.append(text)
        return '\n'.join(lines)

    def _add_to_graph(self, dot, parent_id=None):
        """
        Helper function to add nodes and edges to the graph.
        """
        token_label = self._insert_line_breaks(self.token, 50)
        token_label = graphviz.escape(token_label).replace('{', '\\{').replace('}', '\\}').replace('|', '\\|').replace('<', '\\<').replace('>', '\\>').replace('\n', '\\n')

        node_label = f"{{ id {self.id} | token '{token_label}' | Q {self.Q:.4f} | p_ucb {self.p_ucb:.4f} | visits {self.visits} | P {self.P:.4f} }}"
        
        # Highlight updated nodes
        fill_color = None
        if self.updated:
            self.updated = False  # reset updated status
            fill_color = "lightgrey"
        if self.is_terminal:
            fill_color = "lightcoral"
        if self.selected:
            self.selected = False  # reset updated status
            fill_color = "lightblue"
        if self.player_node:
            self.player_node = False  # reset player node status
            fill_color = "lightgreen"

        if fill_color:
            dot.node(
                self.id, 
                node_label, 
                shape="record",
                style="filled,bold", 
                fillcolor=fill_color
            )
        else:
            dot.node(
                self.id, 
                node_label,
                shape="record"
            )

        if parent_id:
            dot.edge(parent_id, self.id)

        for child in self._children:
            child._add_to_graph(dot, self.id)

    def save_tree(self, filename: str):
        """Save the tree to a file."""
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

# Helper functions for Tree Search
def load_tree(filename: str = "") -> Node:
    """Load the tree from a file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)

def p_ucb_select(parent_node: Node = None, child_nodes: list = []):
    '''Priority Upper Confidence Bound Selection. Identifies the child node with the highest priority to select and expand out more.
    
    P-UCB(s, a) = Q(s, a) + beta(s) * P(a | s) * sqrt(log(s.visits) / (1 + s'.visits)
    -> where beta = weight for exploration. beta(s) = log((s.visits + c_base + 1) / c_base) + c
    P-UCB-SELECT(s, c) = argmax(a) P-UCB(s, a)
    c_base, and c are hyperparameters that can be tuned. c_base = 10, c = 2, 4, 6 in their experiments: https://arxiv.org/pdf/2303.05510#subsection.D.1

    Q(s, a) \in [0, 1] is the quality of a node
    P(a | s) \in exp([0, 1]) = [1, 2.718] is the prior probability of selecting a token a
    s.visits, s'.visits \in [0, inf) are the number of times the parent node has been visited, and the child node has been visited respectively
    beta(s) \in [0, inf) is the exploration weight

    Note: P-UCB is for relative comparison of nodes to select, so the absolute value of P-UCB doesn't matter. Only the relative values matter.
    '''
    # beta exploration terms hyperparameters
    c_base = 10 # as c_base increases, decreases the delta effect of s_visits
    c = 4 # as c increases, beta increases -> exploration increases globally

    if not parent_node: # root node case, you explore root_node times (t), and your child (root) has been visited t times too. So just set parent_visits to child_node.visits
        parent_node = child_node # denominator scales up faster so you'll have a smaller UCB term for the root node eventually (log numerator vs linear denominator)

    beta = np.log((parent_node.visits + c_base + 1) / c_base) + c # add-on exploration term

    for node in child_nodes:
        node.p_ucb = node.Q + beta * node.P * np.sqrt(np.log(parent_node.visits)) / (1 + node.visits) # we're not updating child node until it's selected, but its Q value has indeed been updated so we do + 1. parent_node was updated / selected though
    selected_node = max(child_nodes, key=lambda node: node.p_ucb)
    selected_node.updated = True  # mark the selected node as updated for visualization purposes

    return selected_node

# Function to get hidden states from the model. # TODO: can be made more efficient by getting hidden state, recording, then getting the output
def get_hidden_states(inputs, model):
    with torch.no_grad():
        outputs = model.base_model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1]  # Return the last layer hidden states

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1)

# Function to generate tokens until the sequence has a cosine similarity less than the threshold
def generate_until_threshold(model, tokenizer, prompt, threshold, branching_factor):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # inputs['input_ids'].shape = [1, seq_len]
    prompt_hidden_states = get_hidden_states(inputs, model)[0] # shape = [seq_len, hidden_size]
    
    generated_sequences = []
    
    for branch_i in range(branching_factor):
        print("\n\n- Generating child sequence", branch_i)
        generated_tokens = inputs['input_ids'].to(model.device)  # shape = [1, seq_len]
        attention_mask = inputs['attention_mask'].to(model.device) # [batch_size, batch_max_seq_len]
        current_hidden_states = prompt_hidden_states # we use this for cosime similarity
        new_tokens = []
        token_log_probs = []

        similarity = 0
        while True and generated_tokens.shape[1] <= MAX_TOKENS:
            with torch.no_grad():
                outputs = model.base_model.generate(
                    input_ids=generated_tokens,
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=1,
                    num_return_sequences=1,
                    output_scores=True,
                    renormalize_logits=True,
                    return_dict_in_generate=True,
                    attention_mask=attention_mask
                )

            next_token = outputs.sequences[:, -1].unsqueeze(0)  # Get the last token
            # End if it's the end of the sequence or empty token
            if next_token.item() == model.base_model.generation_config.eos_token_id or tokenizer.decode([next_token.item()], skip_special_tokens=True) == '':
                break

            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)), dim=1)
            new_tokens.append(next_token)
            token_log_probs.append(outputs.scores[-1][0, next_token.item()].item())

            extended_hidden_states = get_hidden_states({'input_ids': generated_tokens}, model)[0]
            current_final_hidden_state = current_hidden_states[-1]
            extended_final_hidden_state = extended_hidden_states[-1]
            similarity = cosine_similarity(current_final_hidden_state, extended_final_hidden_state)

            if similarity < threshold: # If the similarity is less than the threshold (different from prev state), break
                break

            current_hidden_states = extended_hidden_states  # Update the current hidden states

        if len(new_tokens) != 0:
            generated_sequence = torch.cat(new_tokens, dim=1)
            avg_log_prob = sum(token_log_probs) / len(token_log_probs)  # Average log probability
            generated_sequences.append((generated_sequence, avg_log_prob))

            print(f"Similarity: {similarity.item():4f}\nGenerated tokens: '{tokenizer.decode(generated_sequence.squeeze(), skip_special_tokens=True)}'")
    
    return generated_sequences

def expand_seq_node_children_only(curr_node: Node = None, branching_factor: int = 3, node_idx: int = 0, threshold: float = 0.35): # Cosine similarity threshold for generated tokens, 0.35 from testing what a "distinct" idea / sequence of tokens is. If above this threshold, we merge.

    # Generate branching_factor next tokens
    prompt = curr_node.state
    print("Threshold: ", threshold)
    print("Prompt: ", prompt)
    generated_sequences = generate_until_threshold(model, tokenizer, prompt, threshold, branching_factor)

    # Create nodes for each generated token
    parent_node = curr_node
    child_nodes = []

    print("\n\n--- Printing full generated sequences ---")
    for idx, (generated_sequence, avg_log_prob) in enumerate(generated_sequences):
        generated_tokens = tokenizer.decode(generated_sequence.squeeze(), skip_special_tokens=True)
        new_state = parent_node.state + generated_tokens
        print(f"\nGenerated sequence: {idx}\n'{new_state}'")

        # Check cosine similarity with other child nodes
        add_node = True
        for existing_node in child_nodes:
            existing_hidden_states = get_hidden_states(tokenizer(existing_node.state, return_tensors="pt").to(model.device), model)[0]
            new_hidden_states = get_hidden_states(tokenizer(new_state, return_tensors="pt").to(model.device), model)[0]

            existing_final_hidden_state = existing_hidden_states[-1]
            new_final_hidden_state = new_hidden_states[-1]

            similarity = cosine_similarity(existing_final_hidden_state, new_final_hidden_state)
            if similarity >= threshold: # If the similarity is less than the threshold (different enough), add the node
                add_node = False
                break
        if not add_node:
            continue

        # Create a new node
        new_node = Node(
            id=f"{str(node_idx)}",
            prob=avg_log_prob,
            state=new_state,
            parent=parent_node,
            token=generated_tokens,
            depth=parent_node.depth + 1
        )
        node_idx += 1

        child_nodes.append(new_node)
        parent_node._children.append(new_node)

    return child_nodes, node_idx

def llm_generate(prompt: str = ""):
    '''Greedy decoding from the prompt. Returns the generated text.'''
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids = inputs['input_ids'], 
                             max_new_tokens=256, # 1024 > 30s, 256 ~ 10s
                             top_k=1, 
                             do_sample=True) # Greedy decoding
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


def calculate_reward(node: Node = None, true_answer: str = "", root: Node = None, terminal: bool = False):
    '''Calculate the reward for the generated solution. Returns -1 for incorrect submission, 0 for no submission, 1 for correct submission.'''

    solution = node.state
    match = re.search(r'\\boxed\{(\d+)\}', solution[len(root.state):]) # If terminal node, return ground truth reward
    if match or terminal:
        node.is_terminal = True # for visualization purposes
        if match:
            if match.group(1) == true_answer:
                return 1
            else:
                return -1
        
    # If not terminal, evaluate the solution using the value head
    return model.evaluate(solution) # Expansion should include a value head prediction

def generate_problem(num_digits=-1):
    '''Generate a random addition problem.'''
    if num_digits == -1:
        index = random.randint(1, 100)
    else:
        index = num_digits
    a = random.randint(10**(index-1), 10**index - 1)
    b = random.randint(10**(index-1), 10**index - 1)
    prompt = f"{a} + {b} = " + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    true_answer = a + b
    return prompt, str(true_answer)

def collect_training_examples(root, tokenizer, output_file):
    examples = []
    
    def get_generated_sequences(node):
        """Collect the generated sequences for the node's children."""
        sequences = []
        for child in node._children:
            sequences.append(child.token)
        return sequences

    def bfs_collect(node):
        """Use BFS to traverse the tree and collect training examples.
        
        Desired format: 
        {
            "state": <node.state - node.token>,
            "generated_sequence": <node.token>,
            "value": <node.Q>
        } * node.visits

        """
        queue = deque([node])
        while queue:
            current_node = queue.popleft()
            if current_node._parent:  # if it's not a root node
                # generated_sequences = get_generated_sequences(current_node)
                for _ in range(current_node.visits):
                    example = {
                        "state": current_node.state[:-len(current_node.token)], # state before token was generated
                        "generated_sequence": current_node.token,
                        "value": current_node.Q
                    }
                    examples.append(example) # add the example visit times
            for child in current_node._children:
                queue.append(child)

    # Start BFS from the root node
    bfs_collect(root)

    # Save examples to JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)

    print(f"Training examples saved to {output_file}")


# --- ROLLOUTS / GAMEPLAY ---

# hyperparameters
# MODEL_CHECKPOINT_DIR = 'v0_6_value_head_training_v0.61' # Checkpoint for the policy and value model
MAX_TOKENS = 256  # Maximum number of tokens to generate
NUM_ROLLOUTS = 5 # 20
BRANCHING_FACTOR = 10  # 20
SAVE_VERSION = "v2.011"  # for saving the filename
start_time = time.time()
TOTAL_GAMES = 2
STARTING_GAME_NUMBER = 0  # for saving the filename


# Load model
print("Is CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "deepseek-ai/deepseek-math-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLMWithValueHead(model_name, tokenizer).to(device)
model.eval() # Set the model to evaluation mode, no training

# Load the model checkpoint # DEBUG: not including the value head checkpoint for now
# if os.path.exists(MODEL_CHECKPOINT_DIR):
#     checkpoint_path = f'{MODEL_CHECKPOINT_DIR}/model_checkpoint.pth'
#     state_dict = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(state_dict, strict=False)  # strict=False to allow missing keys if any. 
#     model.eval()  # Set the model to evaluation mode

num_digits = 1
for game_number in range(STARTING_GAME_NUMBER, STARTING_GAME_NUMBER + TOTAL_GAMES):
    # Generate a random addition problem
    if TOTAL_GAMES >= 10 and game_number % (TOTAL_GAMES // 10) == 0: # DEBUG: (TOTAL_GAMES // 10) games per digit, otherwise all 
        num_digits += 1
    prompt, true_answer = generate_problem(num_digits=num_digits)
    print(f"\nPrompt: {prompt}\nTrue Answer: {true_answer}\n")

    # initialize tree with addition prompt
    os.makedirs(f"trees_{SAVE_VERSION.replace('.', '_')}", exist_ok=True)
    tree_save_dir = f"trees_{SAVE_VERSION.replace('.', '_')}/game_{game_number}"
    node_id = 0
    root = Node(id = str(node_id), 
                prob = 0.0, # this is log_prob --> prob = e^log_prob 
                state = prompt, 
                parent = None, 
                token = prompt, 
                depth = 0)
    node_id += 1
    print(f"\nCreated new root node:\n{root}")

    # START ITERATIONS HERE
    player_node = root
    num_actions = 0
    while not player_node.is_terminal: # Continue selecting player nodes with the highest Q value
        print(f"\n--- player_node {num_actions} ---\n{player_node}")

        # Start MCTS rollouts
        total_rollouts = num_actions * NUM_ROLLOUTS
        for rollout_idx in range(total_rollouts, total_rollouts + NUM_ROLLOUTS):

            # begin first rollout
            print(f"\n--- Starting rollout {rollout_idx}, with node_id {node_id} ---")
            curr_node = player_node
            curr_node.player_node = True
            curr_node.visits += 1
            print(f"\ncurr_node:\n{curr_node}")

            # selection
            print(f"\n--- Selection {rollout_idx} ---")
            while len(curr_node._children) > 0:
                curr_node = p_ucb_select(curr_node, curr_node._children)
                curr_node.visits += 1
            curr_node.selected = True
            print(f"Selected Node: \n{curr_node}")

            # EXPANSION
            print(f"\n--- Expansion {rollout_idx} ---")
            leaf_nodes = []
            if not curr_node.is_terminal: # only expand if not terminal
                leaf_nodes, node_id = expand_seq_node_children_only(curr_node, branching_factor = BRANCHING_FACTOR, node_idx = node_id, threshold = 0.35) # Empirically seems to not be word level (0.4) and not full answer level (0.3) but somewhere in between
            print(f"\nNumber of expanded nodes: {len(leaf_nodes)}")

            # Evaluation
            print(f"\n--- Evaluation {rollout_idx} ---")
            if leaf_nodes:
                for node_idx, node in enumerate(leaf_nodes):
                    node.Q = calculate_reward(node, true_answer, root)
                    node.updated = True
                    print(f"Reward for child {node_idx}: {node.Q}")
            else: # No leaf nodes, we should just calculate the reward for the current node as a terminal state
                curr_node.Q = calculate_reward(curr_node, true_answer, root, terminal=True)
                curr_node.updated = True
                print(f"Reward for current node: {curr_node.Q}")

            # Backpropagation
            print(f"\n--- Backpropagation {rollout_idx} ---")
            if leaf_nodes:
                curr_node.backprop()
            else:
                curr_node._parent.backprop()

            # visualize and save the tree
            os.makedirs(tree_save_dir, exist_ok=True)
            root.visualize_tree(f"{tree_save_dir}/tree_{rollout_idx}")
            print(f"Tree visualization saved as {tree_save_dir}/tree_{rollout_idx}.png")
            tree_filename = f"{tree_save_dir}/tree_{rollout_idx}.pkl"
            root.save_tree(tree_filename)
            print(f"Tree saved as {tree_filename}")

            # input(f"\nFinished round {rollout_idx}. Press Enter to continue...\n") # DEBUG

        # Select the child node with the highest Q value and your next action
        if not player_node._children:
            print("\nPlayer node is a terminal node", player_node)
            break
        player_node = max(player_node._children, key=lambda node: (node.Q, node.visits, node.P))
        num_actions += 1

        print(f"\nnew player_node: \n{player_node}")

    # visualize and save the tree
    os.makedirs(tree_save_dir, exist_ok=True)
    player_node.player_node = True
    root.visualize_tree(f"{tree_save_dir}/tree_final")
    print(f"Tree visualization saved as {tree_save_dir}/tree_final.png")
    tree_filename = f"{tree_save_dir}/tree_final.pkl"
    root.save_tree(tree_filename)
    print(f"Tree saved as {tree_filename}")

    with open(f"{tree_save_dir}/time.txt", "w") as file:
        file.write(f"Time taken: {time.time() - start_time:.2f} seconds")


    # Collect training examples of the form: Training example = (s_t, \pi, z) = (s_t, [<visit_count_child_1_percentage>, etc.], value)
    collect_training_examples(root, tokenizer, os.path.join(tree_save_dir, "examples.json"))