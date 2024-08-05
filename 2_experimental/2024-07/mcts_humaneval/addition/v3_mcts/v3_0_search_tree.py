'''
Main Tree Search class and helper functions for MCTS self play
'''

import os
from math import exp
import subprocess
import graphviz
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import random
import json
from v3_0_policy_value_model import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import re
from collections import deque

# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)


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
            if node._children:
                total_visits = sum((child.visits + 1) for child in node._children) # child.visits start off at 0 visits even though their Q value is already updated, so we add 1
                node.Q = sum(child.Q * (child.visits + 1) for child in node._children) / total_visits         
            
            node.updated = True  # Mark the node as updated Q value for visualization purposes
            node = node._parent


    def visualize_tree(self, filename: str, dpi: int = 300):
        """
        Visualize the tree and save it as an PNG file.
        """
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

    return selected_node

# Function to get hidden states from the model. 
def get_hidden_states(inputs, model):
    with torch.no_grad():
        outputs = model.base_model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1]  # Return the last layer hidden states # Shape: [batch_size, seq_len, hidden_size]

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1)

# Function to generate tokens until the sequence has a cosine similarity less than the similarity threshold (that are different enough in state)
def generate_seqs_under_sim_threshold(model, tokenizer, prompt, similarity_threshold, branching_factor, MAX_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # inputs['input_ids'].shape = [1, seq_len]
    parent_hidden_states = get_hidden_states(inputs, model)[0] # shape = [seq_len, hidden_size]
    
    generated_sequences = []
    for branch_i in range(branching_factor): # try to generate up to branching_factor different child sequences
        print("\n\n- Generating child sequence", branch_i)
        generated_tokens = inputs['input_ids'].to(model.device)  # shape = [1, seq_len]
        attention_mask = inputs['attention_mask'].to(model.device) # [batch_size, batch_max_seq_len]
        new_tokens = []
        token_log_probs = []

        similarity = torch.tensor(0.0).to(model.device)
        while True and generated_tokens.shape[1] < MAX_TOKENS: # Only break if >= MAX_TOKENs, or if the generated tokens are different enough, or if the sequence is complete (end token or '')
            with torch.no_grad():
                outputs = model.base_model.generate(
                    input_ids=generated_tokens, # tokenized
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=1,
                    num_return_sequences=1, 
                    output_scores=True,
                    renormalize_logits=True, # softmax of logits / probabilities will sum to 1
                    return_dict_in_generate=True
                )
            next_token = outputs.sequences[:, -1].unsqueeze(0)  # Get the last token

            # Add the token to the generated sequence
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)), dim=1)
            new_tokens.append(next_token)
            token_log_probs.append(outputs.scores[-1][0, next_token.item()].item()) # get the -1 (last) token vocab probabilities, and then get the 0th return_sequence, and then the next_token.item() token's probability

            if next_token.item() == model.base_model.generation_config.eos_token_id or tokenizer.decode([next_token.item()], skip_special_tokens=True) == '': # End if it's the end of the sequence or empty token
                break

            # Calculate cosine similarity between the prompt hidden states and the extended hidden states using last token representation
            child_hidden_states = get_hidden_states({'input_ids': generated_tokens}, model)[0]
            similarity = cosine_similarity(parent_hidden_states[-1], child_hidden_states[-1]) # compare the last hidden states of the parent and child # TODO: tunable hyperparameter of 1) last token representation (current bc value head updates this) or 2) mean pooling
            if similarity < similarity_threshold: # If the similarity is less than the similarity_threshold (different from prev state), break because the state is now different enough
                break
            # if too similar, continue generating tokens to expand the child state

        if len(new_tokens) != 0: # If we generated tokens, add the sequence to the list
            generated_sequence = torch.cat(new_tokens, dim=1)
            avg_log_prob = sum(token_log_probs) / len(token_log_probs)  # Average log probability
            generated_sequences.append((generated_sequence, avg_log_prob))

            print(f"Similarity with parent: {similarity.item():4f}\nGenerated tokens: '{tokenizer.decode(generated_sequence.squeeze(), skip_special_tokens=True)}'")
    
    return generated_sequences

def expand_seq_node_children_only(curr_node: Node = None, branching_factor: int = 3, node_idx: int = 0, similarity_threshold: float = 0.35, model: AutoModelForCausalLMWithValueHead = None, tokenizer: AutoTokenizer = None, MAX_TOKENS: int = 0): # Cosine similarity threshold for generated tokens, 0.35 from testing what a "distinct" idea / sequence of tokens is. If above this similarity threshold, we merge because they're too similar.

    # Generate branching_factor next tokens
    prompt = curr_node.state
    print("Similarity threshold to merge: ", similarity_threshold)
    print("Prompt: ", prompt)
    generated_sequences = generate_seqs_under_sim_threshold(model, tokenizer, prompt, similarity_threshold, branching_factor, MAX_TOKENS) # Generate tokens until the sequence has 1) a cosine similarity less than the similarity threshold (or 2) different enough or complete or 3) too long)

    # Create nodes for each generated token
    parent_node = curr_node
    child_nodes = []
    print("\n\n--- Printing full generated sequences ---")
    for idx, (generated_sequence, avg_log_prob) in enumerate(generated_sequences):
        generated_tokens = tokenizer.decode(generated_sequence.squeeze(), skip_special_tokens=True)
        new_state = parent_node.state + generated_tokens
        print(f"\nGenerated sequence: {idx}\n'{new_state}'")

        # Check cosine similarity with other child nodes to make sure it's different enough from other child nodes
        add_node = True
        for existing_child_node in child_nodes:
            existing_child_hidden_states = get_hidden_states(tokenizer(existing_child_node.state, return_tensors="pt").to(model.device), model)[0, -1, :] # get the first batch, last token hidden state representation
            new_child_hidden_states = get_hidden_states(tokenizer(new_state, return_tensors="pt").to(model.device), model)[0, -1, :]
            similarity = cosine_similarity(existing_child_hidden_states, new_child_hidden_states)
            
            if not (similarity < similarity_threshold): # If the similarity is not different enough, do not add the node as a new child
                add_node = False
                break
        if not add_node:
            continue # skip adding this child

        # Create a new child node
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

def check_if_terminal(node: Node = None, root: Node = None, model: AutoModelForCausalLMWithValueHead = None, tokenizer: AutoTokenizer = None, MAX_TOKENS: int = 0):
    '''
    Terminal node criteria: 
        1) tokenizer(node.state) >= MAX_TOKENs
        2) if last_token.item() == model.base_model.generation_config.eos_token_id or tokenizer.decode([last_token.item()], skip_special_tokens=True) == ''
        3) if it contains a submission (boxed)
    '''
    # Criteria 1
    if len(tokenizer(node.state, return_tensors="pt")['input_ids'][0]) >= MAX_TOKENS:
        print("Terminal node: max tokens")
        print(node)
        return True
    
    # Criteria 2
    last_token = tokenizer(node.state, return_tensors="pt")['input_ids'][0, -1]
    if last_token.item() == model.base_model.generation_config.eos_token_id or tokenizer.decode([last_token.item()], skip_special_tokens=True) == '':
        print("Terminal node: EOS token or empty token")
        print(node)
        return True
    
    # Criteria 3
    match = re.search(r'\\boxed\{(\d+)\}', node.state[len(root.state):])
    if match:
        print("Terminal node: submission")
        print(node)
        return True

    return False    


def evaluate_state(node: Node = None, true_answer: str = "", root: Node = None, model: AutoModelForCausalLMWithValueHead = None):
    '''Calculate the reward for the generated solution. 
    
    If terminal: Returns -1 for incorrect submission, 0 for no submission, 1 for correct submission.
    
    If not terminal: Returns the value head prediction.
'''    
    # If node.is_terminal, evaluate the solution according to true answer
    if node.is_terminal:
        match = re.search(r'\\boxed\{(\d+)\}', node.state[len(root.state):])
        if match:
            if match.group(1) == true_answer:
                return 1
            else:
                return -1
        else:
            return 0 # No answer
    
    # If not terminal, evaluate the solution using the value head
    return model.evaluate(node.state) # Expansion should include a value head prediction

def generate_problem(num_digits=-1):
    '''Generate a random addition problem. Currently chooses a random digit problem from 1 - 100'''
    if num_digits == -1:
        index = random.randint(1, 100)
    else:
        index = num_digits
    a = random.randint(10**(index-1), 10**index - 1)
    b = random.randint(10**(index-1), 10**index - 1)
    prompt = f"{a} + {b} = ?" + "\nPlease reason step by step, and put your final answer within \\boxed{}." # Needed for evaluation to parse from boxed
    true_answer = a + b
    return prompt, str(true_answer)

def collect_training_examples(root: Node = None, output_file: str = ""):
    examples = []

    def bfs_collect(node: Node = None):
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

def get_latest_game_index(save_dir):
    if not os.path.exists(f"{save_dir}/latest"):
        with open(f"{save_dir}/latest", "w") as file:
            file.write("0") # initial game index = 0

    with open(f"{save_dir}/latest", "r") as file:
        latest_game_index = file.read().strip()
    return latest_game_index
    
def make_datasets(save_version):
    latest_game_index = get_latest_game_index(save_version)

    # get all examples from latest self-play game
    latest_games_dir = f"{save_version}/{latest_game_index}_self_play"
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

    with open(f"{latest_games_dir}/train_examples.json", "w") as file:
        json.dump(all_examples[:split_index], file, indent=4)
    with open(f"{latest_games_dir}/val_examples.json", "w") as file:
        json.dump(all_examples[split_index:], file, indent=4)

    return latest_games_dir, latest_game_index

def get_available_gpu(threshold=1000):
    '''# Function to find the next available GPU with less than 1GB usage'''
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']
        ).decode('utf-8')
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        available_gpus = [i for i, mem in enumerate(gpu_memory) if mem < threshold]
        if available_gpus:
            return available_gpus[0]
        else:
            return None
    except Exception as e:
        print(f"Error getting GPU memory usage: {e}")
        return None