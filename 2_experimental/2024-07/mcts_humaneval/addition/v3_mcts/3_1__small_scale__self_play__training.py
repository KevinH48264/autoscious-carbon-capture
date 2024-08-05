'''Main self-play and training program for MCTS to do learn strong reasoning patterns to solve addition problems.

How to run:
1) Review hyperparameters below
2) Run 'wandb login' if you aren't already logged in
3) Run "python 3_1__small_scale__self_play__training.py &> log.txt &"

Optional:
1) Preload a pre-trained value head / checkpoint by defining INITIAL_CHECKPOINT_PATH

Add-ons:
1. Curriculum learning in self-play based on evaluation
'''

# libraries
import random
import sys
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import wandb
from tqdm import tqdm
import json
import shutil

# classes and functions
from v3_0_policy_value_model import AutoModelForCausalLMWithValueHead, plot_validation_metrics
from v3_0_search_tree import Node, check_if_terminal, generate_problem, p_ucb_select, expand_seq_node_children_only, evaluate_state, collect_training_examples, get_latest_game_index, get_available_gpu
from v3_0_dataset import PolicyValueDataset, collate_fn, get_latest_game_index, make_datasets

# set seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Disable parallelism to disable warning of forking after parallelism and prevent deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# hyperparameters
MAX_TOKENS = 128 # DEBUG: 256  # Maximum number of tokens to generate during self-play + training # NOTE: this needs to be more than the longest prompt in the dataset otherwise there will be an error
NUM_ROLLOUTS = 2 # DEBUG: 20 # Number of rollouts per iteration during self-play
BRANCHING_FACTOR = 2 # DEBUG: 20 # Number of children to expand per node during self-play
TOTAL_GAMES = 2 # DEBUG: ? # Number of games to play during self-play
MODEL_GENERATIONS = 20 # DEBUG: 20 # Number of model generations from self-play + training
NUM_EVAL = 10 # Number of evaluations to run (for addition, evaluates from 1 to NUM_EVAL digits)
BATCH_SIZE = 2 # DEBUG: 4 # Batch size for training and validation
INITIAL_CHECKPOINT_PATH = './v2_012_value_head_training_v2.011__50_epochs__80000_train_20000_val__5e-4_lr__1e-4_wd__32_batch_size__8_gpus/model_checkpoint_30.pt' # Pretrained value head or previous checkpoint

# initialize GPU
available_gpu = get_available_gpu()
if available_gpu is not None:
    device = torch.device(f"cuda:{str(available_gpu)}" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU {available_gpu} with less than 1GB usage.")
else:
    print("No available GPU found with less than 1GB usage. Exiting.")
    sys.exit(1)
print("Is CUDA available:", torch.cuda.is_available())

# initialization
SAVE_DIR = f"v3_1__eval_{NUM_EVAL}__tokens_{MAX_TOKENS}__rollouts_{NUM_ROLLOUTS}__branch_{BRANCHING_FACTOR}__games_{TOTAL_GAMES}__gen_{MODEL_GENERATIONS}__batch_{BATCH_SIZE}" # for saving the filename
print(f"Save directory: {SAVE_DIR}")
os.makedirs(SAVE_DIR, exist_ok=True)
full_program_start_time = time.time()
with open(f"{SAVE_DIR}/info", "a") as file:
    file.write(f"Evaluations are for first 10 digits only for now.\n")

# load model for both self-play and training
model_name = "deepseek-ai/deepseek-math-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLMWithValueHead(model_name, tokenizer, device).to(device)

# Copy initial checkpoint to checkpoint_path
checkpoint_path =  f'{SAVE_DIR}/latest_checkpoint.pt'
if os.path.exists(INITIAL_CHECKPOINT_PATH):
    shutil.copy(INITIAL_CHECKPOINT_PATH, checkpoint_path)
    with open(f"{SAVE_DIR}/info", "a") as file:
        file.write(f"Initial checkpoint path: {INITIAL_CHECKPOINT_PATH}\n\n")


# --- MAIN LOOP ---
for model_generation_idx in range(MODEL_GENERATIONS):
    # load model checkpoint for self-play and training
    print("Loading model checkpoint...")
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)  # strict=False to allow missing keys if any. 
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4) # weight_decay penalizes large weights. Adds a regularization term to the loss function that shrinks the weights during each update. TODO: tunable hyperparameter
    model.eval()  # Set the model to evaluation mode


    '''1. SELF PLAY

    This implements MCTS 1 self-play trajectory like AlphaZero'''
    print(f"\n--- SELF PLAY {model_generation_idx + 1} ---")
    start_time = time.time() # game generation start time

    # --- GAMEPLAY ---
    # logging
    curr_game_index = int(get_latest_game_index(SAVE_DIR)) + 1
    self_play_save_path = SAVE_DIR + f"/{curr_game_index}_self_play" # for saving the filename
    os.makedirs(self_play_save_path, exist_ok=True)

    # curriculum learning based on evaluation results
    num_digits = 1
    if curr_game_index > 1:
        with open(f"{SAVE_DIR}/{curr_game_index - 1}_training/validation_metrics.json", "r") as file:
            metrics = json.load(file)
            correct_answers = metrics["correct_answers"]

            for i in range(0, 1000): # find the smallest from 0 to infinity that isn't in correct_answers
                if i not in correct_answers:
                    num_digits = i + 1 # +1 because we're 0-indexed
                    break
    with open(f"{SAVE_DIR}/info", "a") as file:
        file.write(f"Self-play Games Generation: {curr_game_index} - Number of digits: {num_digits}\n")

    # tree search / self-play
    for game_number in range(TOTAL_GAMES):
        prompt, true_answer = generate_problem(num_digits=num_digits) # DEBUG: hardcoding for now
        print(f"\nPrompt: {prompt}\nTrue Answer: {true_answer}\nPrompt tokens: {len(tokenizer(prompt)['input_ids'])}\n")

        # initialize tree with addition prompt
        tree_save_dir = f"{self_play_save_path}/game_{game_number}"
        os.makedirs(tree_save_dir, exist_ok=True)
        node_id = 0
        root = Node(id = str(node_id), 
                    prob = 0.0, # this is log_prob --> prob = e^log_prob # TODO: double check this range, but not critical because relative probabilities will persist
                    state = prompt, 
                    parent = None, 
                    token = prompt, 
                    depth = 0)
        node_id += 1
        print(f"\nCreated new root node:\n{root}")

        # START ITERATIONS HERE
        player_node = root
        num_actions = 0 # player moves (not simulation)
        while not player_node.is_terminal: # Continue selecting player nodes with the highest Q value. # Terminal = max tokens reached in state, or answer contains \boxed{} = correct | incorrect
            print(f"\n--- player_node {num_actions} ---\n{player_node}")

            # Start MCTS rollouts
            total_rollouts = num_actions * NUM_ROLLOUTS # for logging unique rollout_idxs
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
                    leaf_nodes, node_id = expand_seq_node_children_only(curr_node, branching_factor = BRANCHING_FACTOR, node_idx = node_id, similarity_threshold = 0.35, model = model, tokenizer = tokenizer, MAX_TOKENS = MAX_TOKENS) # Empirically seems to not be word level (0.4) and not full answer level (0.3) but somewhere in between. # Anything above similarity threshold are too similar and merged. # TODO: tunable hyperparameter for threshold
                print(f"\nNumber of expanded nodes: {len(leaf_nodes)}")

                # Evaluation
                print(f"\n--- Evaluation {rollout_idx} ---")
                if leaf_nodes: # evaluate if leaf nodes, else it's terminal and already has a Q value
                    for node_idx, node in enumerate(leaf_nodes):
                        # Check if the node is terminal
                        node.is_terminal = check_if_terminal(node, root, model, tokenizer, MAX_TOKENS)
                        node.Q = evaluate_state(node, true_answer, root, model=model) # Note: we include the true_answer as the "rule" of the game. We're trying to build the right "reasoning" to achieve the true answer / win the game.
                        node.updated = True # mark as updated Q value
                        print(f"Reward for child {node_idx}: {node.Q}")

                # Backpropagation
                print(f"\n--- Backpropagation {rollout_idx} ---")
                curr_node.backprop()

                # visualize and save the tree
                root.visualize_tree(f"{tree_save_dir}/tree_{rollout_idx}")
                print(f"Tree visualization saved as {tree_save_dir}/tree_{rollout_idx}.png")

            # Finished simulation rollouts, select the child node with the highest Q value and your next action
            if player_node.is_terminal:
                print("\nPlayer node is a terminal node", player_node)
                break

            player_node = max(player_node._children, key=lambda node: (node.Q, node.visits, node.P))
            num_actions += 1
            print(f"\nnew player_node: \n{player_node}")

        # visualize and save the final tree
        player_node.player_node = True
        root.visualize_tree(f"{tree_save_dir}/tree_final")
        print(f"Tree visualization saved as {tree_save_dir}/tree_final.png")

        with open(f"{tree_save_dir}/time.txt", "w") as file:
            elapsed_time = time.time() - start_time
            file.write(f"Time taken: {elapsed_time:.2f} seconds = {elapsed_time / 60:.2f} minutes = {elapsed_time / 3600:.2f} hours")


        # Collect training examples of the form: Training example = (s_t, \pi, z) = (s_t - token, token [generated sequence], value) for all visits <= MAX_TOKENS * NUM_ROLLOUTS
        collect_training_examples(root, os.path.join(tree_save_dir, "examples.json"))

    # Save the latest game index and log time taken
    with open(f"{SAVE_DIR}/latest", "w") as file:
        file.write(str(curr_game_index))

    with open(f"{self_play_save_path}/time.txt", "w") as file:
        elapsed_time = time.time() - start_time
        file.write(f"Time taken: {elapsed_time:.2f} seconds = {elapsed_time / 60:.2f} minutes = {elapsed_time / 3600:.2f} hours")


    '''2. TRAINING'''
    print(f"\n--- TRAINING {model_generation_idx} ---")
    training_start_time = time.time()

    # data setup
    curr_training_dir, latest_game_index = make_datasets(SAVE_DIR)
    train_dataset = PolicyValueDataset(f'{curr_training_dir}/train_examples.json', tokenizer) # 8K examples
    val_dataset = PolicyValueDataset(f"{curr_training_dir}/val_examples.json", tokenizer) # 2K examples
    # train_dataloader = model_engine.deepspeed_io(train_dataset, collate_fn=collate_fn) # for data sharding during distributed training
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)

    # --- TRAINING ---
    # initialization
    training_save_path = SAVE_DIR + f"/{latest_game_index}_training" # for saving the filename
    os.makedirs(training_save_path, exist_ok=True)
    wandb.init(project=f"mcts_{SAVE_DIR}_{latest_game_index}_training")

    num_epochs = 2 # TODO: tunable hyperparameter
    for epoch in range(num_epochs):
        # Training step
        model.train()

        total_loss = 0.0
        total_ce = 0.0 # cross entropy for policy
        total_mse = 0.0 # mean squared error for value
        total_mae = 0.0 # mean absolute error for value
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            input_ids, attention_mask, labels, state_values_true = batch

            # Forward pass
            logits, value_preds, total_loss, ce_loss, mse_loss, mae_loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, state_values=state_values_true)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate losses for logging
            total_loss += total_loss.item()
            total_mse += mse_loss.item()
            total_ce += ce_loss.item()
            total_mae += mae_loss.item()

            progress_bar.set_postfix({"loss": total_loss.item()})

        # Log average losses
        num_batches = len(train_dataloader) if len(train_dataloader) > 0 else 1
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_ce = total_ce / num_batches
        avg_mae = total_mae / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}, MSE: {avg_mse}, CE: {avg_ce}, MAE: {avg_mae}")
        wandb.log({"epoch": epoch+1, "loss": avg_loss, "mse": avg_mse, "ce": avg_ce, "mae": avg_mae})

        # Validation step
        model.eval()
        val_total_loss = 0.0  # Initialize val_total_loss
        val_total_ce = 0.0
        val_total_mae = 0.0
        val_total_mse = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                input_ids, attention_mask, labels, state_values_true = val_batch

                # Forward pass
                logits, value_preds, val_loss, val_ce_loss, val_mse_loss, val_mae_loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, state_values=state_values_true)

                # Accumulate validation losses
                val_total_loss += val_loss.item()  # Accumulate val_total_loss
                val_total_mse += val_mse_loss.item()
                val_total_ce += val_ce_loss.item()
                val_total_mae += val_mae_loss.item()

        # Log average validation losses
        num_batches = len(val_dataloader) if len(val_dataloader) > 0 else 1 # prevent divide by 0 issues
        avg_val_loss = val_total_loss / num_batches  # Calculate average validation total loss
        avg_val_mse = val_total_mse / num_batches
        avg_val_ce = val_total_ce / num_batches
        avg_val_mae = val_total_mae / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}, Validation MSE: {avg_val_mse}, Validation CE: {avg_val_ce}, Validation MAE: {avg_val_mae}")
        wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss, "val_mse": avg_val_mse, "val_ce": avg_val_ce, "val_mae": avg_val_mae})  # Log avg_val_loss

    # Save checkpoint after num_epochs (2 for now)
    print("Saving model checkpoint...")
    # torch.save(model.state_dict(), f'{save_path}/checkpoint.pt') # save locally for evaluation in the future # Uncomment when ready for full test evaluation (more than validation evaluation) otherwise it's too memory intensive
    torch.save(model.state_dict(), f'{SAVE_DIR}/latest_checkpoint.pt') # this saves LoRA weights too
    torch.cuda.empty_cache()

    print("Fine-tuning completed.")
    wandb.finish()

    # Record time taken
    with open(f"{training_save_path}/training_time.txt", "w") as file:
        elapsed_time = time.time() - training_start_time
        file.write(f"Time taken: {elapsed_time:.2f} seconds = {elapsed_time / 60:.2f} minutes = {elapsed_time / 3600:.2f} hours")

    
    '''3. VALIDATION'''
    # --- VALIDATION EVALUATION --- # necessary on top of training validation because this one DOES NOT use ground truth reward during self-play / simulations
    print(f"\n--- VALIDATION {model_generation_idx} ---")
    validation_time_start = time.time()

    # Load validation / test dataset
    with open("100_digit_addition_testset.json", "r") as file:
        data = json.load(file)
    data = data[:NUM_EVAL] # only grab the first NUM_EVAL examples for initial validation. Full test set will be for final evaluation
    total_validation_games = len(data)

    # Logging # save_path = _training/
    correct_answers, incorrect_answers, blank_answers = [], [], []
    game_time = []
    model.eval() # Set the model to evaluation mode, no training
    for game_number in tqdm(range(total_validation_games)):
        # pull the problem from data
        prompt = data[game_number]['question'] + "?" + "\nPlease reason step by step, and put your final answer within \\boxed{}." # Needed for evaluation to parse from boxed and this is how the training example starting prompts look
        true_answer = data[game_number]['answer']
        print(f"\nValidation Question: {data[game_number]['index']}\nPrompt: {prompt}\nTrue Answer: {true_answer}\n")

        # initialize tree with addition prompt
        tree_save_dir = f"{training_save_path}/validation_game_{game_number}"
        os.makedirs(tree_save_dir, exist_ok=True)
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
                    leaf_nodes, node_id = expand_seq_node_children_only(curr_node, branching_factor = BRANCHING_FACTOR, node_idx = node_id, similarity_threshold = 0.35, model = model, tokenizer = tokenizer, MAX_TOKENS = MAX_TOKENS) # Empirically seems to not be word level (0.4) and not full answer level (0.3) but somewhere in between
                print(f"\nNumber of expanded nodes: {len(leaf_nodes)}")

                # Evaluation
                print(f"\n--- Evaluation {rollout_idx} ---")
                if leaf_nodes: # evaluate if leaf nodes, else it's terminal and already has a Q value
                    for node_idx, node in enumerate(leaf_nodes):
                        # Check if the node is terminal
                        node.is_terminal = check_if_terminal(node, root, model, tokenizer, MAX_TOKENS)
                        node.Q = model.evaluate(curr_node.state) # NEW: only use model predictions for evaluation
                        node.updated = True # mark as updated Q value
                        print(f"Reward for child {node_idx}: {node.Q}")

                # Backpropagation
                print(f"\n--- Backpropagation {rollout_idx} ---")
                curr_node.backprop()

                # visualize and save the tree
                root.visualize_tree(f"{tree_save_dir}/tree_{rollout_idx}")
                print(f"Tree visualization saved as {tree_save_dir}/tree_{rollout_idx}.png")

            # Select the child node with the highest Q value and your next action
            if player_node.is_terminal:
                print("\nPlayer node is a terminal node", player_node)
                break
            player_node = max(player_node._children, key=lambda node: (node.Q, node.visits, node.P))
            num_actions += 1
            print(f"\nnew player_node: \n{player_node}")

        # visualize and save the final tree
        player_node.player_node = True
        root.visualize_tree(f"{tree_save_dir}/tree_final")
        print(f"Tree visualization saved as {tree_save_dir}/tree_final.png")

        with open(f"{tree_save_dir}/time.txt", "w") as file:
            elapsed_time = time.time() - start_time
            file.write(f"Time taken: {elapsed_time:.2f} seconds = {elapsed_time / 60:.2f} minutes = {elapsed_time / 3600:.2f} hours")
            

        # Record predictions and log time
        player_node.is_terminal = True # Mark as terminal for evaluation
        reward = evaluate_state(player_node, true_answer, root, model=model) # Calculate knowing true answer
        if reward == 1:
            correct_answers.append(game_number)
        elif reward == -1:
            incorrect_answers.append(game_number)
        else:
            blank_answers.append(game_number)

        game_time.append(time.time() - start_time)


        # Compute average game time and overall correct/blank/incorrect
        average_game_time = sum(game_time) / len(game_time)
        total = len(correct_answers) + len(incorrect_answers) + len(blank_answers)

        metrics = {
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,
            "blank_answers": blank_answers,
            "average_game_time": average_game_time,
            "percent_correct": len(correct_answers) / total,
            "percent_incorrect": len(incorrect_answers) / total,
            "percent_blank": len(blank_answers) / total
        }

        # Save these metrics into metrics.json
        with open(f"{training_save_path}/validation_metrics.json", "w") as file:
            json.dump(metrics, file, indent=4)

        with open(f"{training_save_path}/validation_time.txt", "w") as file:
            elapsed_time = time.time() - validation_time_start
            file.write(f"Time taken: {elapsed_time:.2f} seconds = {elapsed_time / 60:.2f} minutes = {elapsed_time / 3600:.2f} hours")

    plot_validation_metrics(SAVE_DIR)

# Delete latest checkpoint.pt to save on memory for now during experimentation # Uncomment to save and use the latest checkpoint
os.remove(checkpoint_path)

# Record total time taken and completion
print("All model generations completed.")
with open(f"{SAVE_DIR}/time.txt", "w") as file:
    elapsed_time = time.time() - full_program_start_time
    file.write(f"Total time taken to complete {MODEL_GENERATIONS} model generations: {elapsed_time:.2f} seconds = {elapsed_time / 60:.2f} minutes = {elapsed_time / 3600:.2f} hours")