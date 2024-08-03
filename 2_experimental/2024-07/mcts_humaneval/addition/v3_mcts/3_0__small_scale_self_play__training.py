'''Main self-play and training program for MCTS

How to run:
1) Run "python 3_0__small_scale__self_play__training.py"
'''

# libraries
import random
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import wandb
from tqdm import tqdm

# classes and functions
from v3_0_policy_value_model import AutoModelForCausalLMWithValueHead
from v3_0_search_tree import Node, generate_problem, p_ucb_select, expand_seq_node_children_only, calculate_reward, collect_training_examples, get_latest_game_index
from v3_0_dataset import PolicyValueDataset, collate_fn, get_latest_game_index, make_datasets

# set seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# hyperparameters
MAX_TOKENS = 64 # DEBUG: 256  # Maximum number of tokens to generate during self-play + training
NUM_ROLLOUTS = 2 # DEBUG: 20 # Number of rollouts per iteration during self-play
BRANCHING_FACTOR = 2 # DEBUG: 20 # Number of children to expand per node during self-play
TOTAL_GAMES = 2 # DEBUG: ? # Number of games to play during self-play
MODEL_GENERATIONS = 2 # DEBUG: 20 # Number of model generations from self-play + training
BATCH_SIZE = 2 # DEBUG: 4 # Batch size for training and validation
SAVE_VERSION = "v3_0__small_scale__self_play__training" # for saving the filename
os.makedirs(SAVE_VERSION, exist_ok=True)
full_program_start_time = time.time()

# load model for both self-play and training
print("Is CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "deepseek-ai/deepseek-math-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLMWithValueHead(model_name, tokenizer).to(device)

# --- MAIN LOOP ---
for model_generation_idx in range(MODEL_GENERATIONS):
    # load model checkpoint for self-play and training
    print("Loading model checkpoint...")
    checkpoint_path =  f'{SAVE_VERSION}/latest_checkpoint.pt'
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)  # strict=False to allow missing keys if any. 
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4) # weight_decay penalizes large weights. Adds a regularization term to the loss function that shrinks the weights during each update. 
    model.eval()  # Set the model to evaluation mode


    '''1. SELF PLAY

    This implements MCTS 1 self-play trajectory like AlphaZero'''
    print(f"\n--- SELF PLAY {model_generation_idx} ---")
    start_time = time.time()

    # --- GAMEPLAY ---
    # logging
    STARTING_GAME_NUMBER = 0  # for saving the filename
    curr_game_index = int(get_latest_game_index(SAVE_VERSION)) + 1
    save_path = SAVE_VERSION + f"/{curr_game_index}_self_play" # for saving the filename
    os.makedirs(save_path, exist_ok=True)

    num_digits = 2 # Number of digits in the addition problem to start off with #DEBUG: so we don't get lower than 1 example < batch_size if the question is too easy
    for game_number in range(STARTING_GAME_NUMBER, STARTING_GAME_NUMBER + TOTAL_GAMES):
        # Generate a random addition problem
        if TOTAL_GAMES >= 10 and game_number % (TOTAL_GAMES // 10) == 0: # DEBUG: (TOTAL_GAMES // 10) games per digit, otherwise all 
            num_digits += 1
        prompt, true_answer = generate_problem(num_digits=num_digits)
        print(f"\nPrompt: {prompt}\nTrue Answer: {true_answer}\n")

        # initialize tree with addition prompt
        os.makedirs(f"{save_path}/", exist_ok=True)
        tree_save_dir = f"{save_path}/game_{game_number}"
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
                    leaf_nodes, node_id = expand_seq_node_children_only(curr_node, branching_factor = BRANCHING_FACTOR, node_idx = node_id, threshold = 0.35, model = model, tokenizer = tokenizer, MAX_TOKENS = MAX_TOKENS) # Empirically seems to not be word level (0.4) and not full answer level (0.3) but somewhere in between
                print(f"\nNumber of expanded nodes: {len(leaf_nodes)}")

                # Evaluation
                print(f"\n--- Evaluation {rollout_idx} ---")
                if leaf_nodes:
                    for node_idx, node in enumerate(leaf_nodes):
                        node.Q = calculate_reward(node, true_answer, root, model=model, tokenizer=tokenizer, MAX_TOKENS=MAX_TOKENS)
                        node.updated = True
                        print(f"Reward for child {node_idx}: {node.Q}")
                else: # No leaf nodes, we should just calculate the reward for the current node as a terminal state
                    curr_node.Q = calculate_reward(curr_node, true_answer, root, terminal=True, model=model, tokenizer=tokenizer, MAX_TOKENS=MAX_TOKENS)
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

        with open(f"{tree_save_dir}/time.txt", "w") as file:
            file.write(f"Time taken: {time.time() - start_time:.2f} seconds")


        # Collect training examples of the form: Training example = (s_t, \pi, z) = (s_t, [<visit_count_child_1_percentage>, etc.], value)
        collect_training_examples(root, tokenizer, os.path.join(tree_save_dir, "examples.json"))

    # Save the latest game index and log time taken
    with open(f"{SAVE_VERSION}/latest", "w") as file:
        file.write(str(curr_game_index))

    with open(f"{save_path}/time.txt", "w") as file:
        file.write(f"Time taken: {time.time() - start_time:.2f} seconds")



    '''2. TRAINING'''
    print(f"\n--- TRAINING {model_generation_idx} ---")
    start_time = time.time()

    # data setup
    curr_training_dir, latest_game_index = make_datasets(SAVE_VERSION)
    train_dataset = PolicyValueDataset(f'{curr_training_dir}/train_examples.json', tokenizer) # 8K examples
    val_dataset = PolicyValueDataset(f"{curr_training_dir}/val_examples.json", tokenizer) # 2K examples
    # train_dataloader = model_engine.deepspeed_io(train_dataset, collate_fn=collate_fn) # for data sharding during distributed training
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)

    # --- TRAINING ---
    # initialization
    save_path = SAVE_VERSION + f"/{latest_game_index}_training" # for saving the filename
    os.makedirs(SAVE_VERSION, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    wandb.init(project=f"mcts_{SAVE_VERSION}_{latest_game_index}_training")
    model.train()

    start_epoch = 0
    epochs = 2
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

        num_batches = len(train_dataloader) if len(train_dataloader) > 0 else 1
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_ce = total_ce / num_batches
        avg_mae = total_mae / num_batches
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

        num_batches = len(val_dataloader) if len(val_dataloader) > 0 else 1 # prevent divide by 0 issues
        avg_val_loss = val_total_loss / num_batches  # Calculate average validation total loss
        avg_val_mse = val_total_mse / num_batches
        avg_val_ce = val_total_ce / num_batches
        avg_val_mae = val_total_mae / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}, Validation MSE: {avg_val_mse}, Validation CE: {avg_val_ce}, Validation MAE: {avg_val_mae}")
        wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss, "val_mse": avg_val_mse, "val_ce": avg_val_ce, "val_mae": avg_val_mae})  # Log avg_val_loss

        model.train()

    # Save checkpoint after 2 epochs
    print("Saving model checkpoint...")
    torch.save(model.state_dict(), f'{save_path}/checkpoint.pt') # save locally for evaluation in the future
    torch.save(model.state_dict(), f'{SAVE_VERSION}/latest_checkpoint.pt')
    torch.cuda.empty_cache()

    print("Fine-tuning completed.")
    wandb.finish()

    # Record time taken
    with open(f"{save_path}/time.txt", "w") as file:
        file.write(f"Time taken: {time.time() - start_time:.2f} seconds")


# Record total time taken and completion
print("All model generations completed.")
with open(f"{SAVE_VERSION}/time.txt", "w") as file:
    file.write(f"Total time taken to complete {MODEL_GENERATIONS} model generations: {time.time() - full_program_start_time:.2f} seconds")