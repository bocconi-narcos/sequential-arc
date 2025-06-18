"""
Generates a replay buffer of (state, action, reward, next_state, done) transitions
from the Abstract Reasoning Corpus (ARC).

This script interacts with an ARC environment (`env.ARCEnv`), loads ARC tasks
from specified JSON files, filters tasks based on grid dimensions, and then
collects interaction data by taking random actions in the environment.

The primary inputs are:
- Path to a challenges JSON file (e.g., `dataset/evaluation/challenges.json`).
- Path to a solutions JSON file (e.g., `dataset/evaluation/solutions.json`).

The script outputs a single pickle file containing the replay buffer,
which is a list of transition tuples. Each tuple typically contains:
(unpadded_state_list, action_int, reward_float, unpadded_next_state_list, done_bool).
"""
import argparse
import pickle
import numpy as np
import os
import random
import json
from collections import deque

from env import ARCEnv
from action_space import ARCActionSpace
from dsl.utils.padding import unpad_grid, pad_grid
from pathlib import Path

# adjust paths
DATA_DIR = Path(__file__).resolve().parent / "data"
CHALLENGES_JSON = DATA_DIR / "challenges.json"
SOLUTIONS_JSON  = DATA_DIR / "solutions.json"

# For the replay buffer, locally
OUTPUT_FILEPATH = Path(__file__).resolve().parent / "replay_buffer_joint.pkl"


def load_and_filter_grids(solutions_data, challenges_data, max_h, max_w):
    """
    Filters grids from solutions_data and challenges_data based on maximum height and width.
    Returns a list of (task_key, example_index) for valid grids.
    """
    valid_grids = []
    
    for task_key in challenges_data:
        if 'train' in challenges_data[task_key]:
            for i, example in enumerate(challenges_data[task_key]['train']):
                try:
                    # Check both input and output dimensions
                    input_grid = np.array(example['input'])
                    output_grid = np.array(solutions_data[task_key])  # Solution is directly in solutions_data
                    
                    if (input_grid.shape[0] <= max_h and input_grid.shape[1] <= max_w and
                        output_grid.shape[0] <= max_h and output_grid.shape[1] <= max_w):
                        valid_grids.append((task_key, i))
                except Exception as e:
                    print(f"Warning: Could not parse grid for task {task_key}, example {i}: {e}")
                    continue
    
    return valid_grids

def process_state(state: np.ndarray) -> np.ndarray:
    """
    Process a state array to ensure it's 2D. If it's 3D (from selection functions),
    combine all layers into a single 2D array.
    
    Args:
        state: Input state array (2D or 3D)
        
    Returns:
        2D numpy array
    """
    if state.ndim == 3:
        # If we have multiple layers (from selection functions),
        # combine them by taking the maximum value across layers
        return np.max(state, axis=0)
    return state

def main():
    """
    Main function to generate the ARC replay buffer.

    Parses command-line arguments, sets up the ARC environment and action space,
    filters initial grids based on specified dimensions, and then enters a data
    collection loop. In this loop, it repeatedly resets the environment to a
    valid initial grid, takes a series of random actions, and stores the
    resulting (state, action, reward, next_state, done) transitions.
    The collected transitions are saved to a pickle file.
    """
    parser = argparse.ArgumentParser(description="Generate a replay buffer for ARC.")

    parser.add_argument(
        "--challenges_json_path",
        type=str,
        default=str(CHALLENGES_JSON),      # ← use constant as default
        help=f"Path to ARC challenges JSON (default: {CHALLENGES_JSON}).",
    )

    parser.add_argument(
        "--solutions_json_path",
        type=str,
        default=str(SOLUTIONS_JSON),       # ← use constant as default
        help=f"Path to ARC solutions JSON (default: {SOLUTIONS_JSON}).",
    )

    parser.add_argument(
        "--output_filepath",
        type=str,
        default=str(OUTPUT_FILEPATH),      # ← use constant as default
        help=f"Where to save the replay buffer (default: {OUTPUT_FILEPATH}).",
    )
    parser.add_argument("--buffer_size", type=int, default=1000,
                        help="Max transitions in the buffer."
    )

    parser.add_argument("--max_grid_dim_h", type=int, default=10,
                        help="Max height for initial grids."
    )

    parser.add_argument("--max_grid_dim_w", type=int, default=10,
                        help="Max width for initial grids."
    )

    parser.add_argument("--num_steps_per_grid", type=int, default=5,
                        help="Number of random actions to take per initial grid."
    )

    parser.add_argument("--canvas_size", type=int, default=10,
                        help="Canvas size for ARCEnv."
    )

    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed."
    )

    parser.add_argument("--action_preset", type=str, default="default",
                        help="Action preset for ARCActionSpace."
    )


    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load solutions and challenges data
    with open(args.solutions_json_path, 'r') as f:
        solutions_data = json.load(f)
    with open(args.challenges_json_path, 'r') as f:
        challenges_data = json.load(f)

    action_space = ARCActionSpace(preset=args.action_preset, mode="joint")
    
    env = ARCEnv(
        challenges_json=args.challenges_json_path,
        solutions_json=args.solutions_json_path,
        action_space=action_space,
        canvas_size=args.canvas_size,
        seed=args.seed 
    )

    valid_initial_grids = load_and_filter_grids(solutions_data, challenges_data, args.max_grid_dim_h, args.max_grid_dim_w)

    if not valid_initial_grids:
        print(f"No initial grids found matching criteria (max_h={args.max_grid_dim_h}, max_w={args.max_grid_dim_w}). Exiting.")
        return

    print(f"Found {len(valid_initial_grids)} valid initial grids based on dimension constraints.")
    print(f"Replay buffer initialized with max size {args.buffer_size}.")
    print(f"Seed set to: {args.seed}")

    replay_buffer = deque(maxlen=args.buffer_size)
    transitions_added = 0

    while len(replay_buffer) < args.buffer_size:
        task_key, example_index = random.choice(valid_initial_grids)
        
        # Get the input from challenges and output from solutions
        example = challenges_data[task_key]['train'][example_index]
        input_grid = np.array(example['input'])
        output_grid = np.array(solutions_data[task_key])  # Solution is directly in solutions_data
        
        print(f"\n=== New Task ===")
        print(f"Task key: {task_key}")
        print(f"Example index: {example_index}")
        print(f"Input grid shape: {input_grid.shape}")
        print(f"Output grid shape: {output_grid.shape}")
        
        # Create the initial state
        grid = np.zeros((args.canvas_size, args.canvas_size, 2), dtype=np.int8)
        grid[..., 0] = pad_grid(input_grid, (args.canvas_size, args.canvas_size))
        grid[..., 1] = pad_grid(output_grid, (args.canvas_size, args.canvas_size))
        
        # Set the environment state directly
        env.state = grid
        env.info = {
            "key": task_key,
            "actions": [],
            "action_desc": [],
            "num_actions": 0,
            "solved": False,
        }
        
        current_grid_padded = env.state[..., 0]
        state_np_unpadded = unpad_grid(current_grid_padded)
        
        for _step in range(args.num_steps_per_grid):
            if len(replay_buffer) >= args.buffer_size:
                break

            action = action_space.sample()
            print(f"\n=== Step {_step} ===")
            print(f"Action: {action}")
            print(f"Action description: {action_space.action_to_str(action)}")
            
            try:
                next_observation, reward, terminated, truncated, _info = env.step(action)
                done = terminated or truncated

                next_grid_padded = next_observation[..., 0]
                # Process the state to handle potential 3D arrays from selection functions
                next_grid_padded = process_state(next_grid_padded)
                next_state_np_unpadded = unpad_grid(next_grid_padded)

                # Store the transition in the replay buffer
                replay_buffer.append((
                    state_np_unpadded.tolist(),  # Convert to list for pickle compatibility
                    action,
                    reward,
                    next_state_np_unpadded.tolist(),
                    done
                ))
                transitions_added += 1
                
                if transitions_added % 100 == 0:
                    print(f"Added {transitions_added} transitions to replay buffer.")
                
                state_np_unpadded = next_state_np_unpadded
                
            except ValueError as e:
                print(f"\nError in step {_step}:")
                print(f"Action: {action}")
                print(f"Action description: {action_space.action_to_str(action)}")
                print(f"Error message: {str(e)}")
                # Continue with next step instead of breaking
                continue

    # Save the replay buffer
    with open(args.output_filepath, 'wb') as f:
        pickle.dump(list(replay_buffer), f)
    print(f"\nReplay buffer saved to {args.output_filepath}")
    print(f"Total transitions added: {transitions_added}")

if __name__ == "__main__":
    main()